"""Memory traffic per window, from Spike's cache counters, and the least a window must move."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass
from pathlib import Path

# Internal
from score.profile.errors import ProfileError
from score.profile.weights import TensorSizes


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Spike's cache flags: the Milk-V Duo's C906, a 32 KB 2-way instruction cache and a 64 KB 4-way data cache, both with 64-byte lines
INSTRUCTION_CACHE = "--ic=256:2:64"
DATA_CACHE = "--dc=256:4:64"
LINE_BYTES = 64

# The counters read from Spike's report; its byte counters overflow and are never read
DATA_READ_MISSES = "D$ Read Misses"
DATA_WRITE_MISSES = "D$ Write Misses"
DATA_WRITEBACKS = "D$ Writebacks"
INSTRUCTION_READ_MISSES = "I$ Read Misses"
COUNTERS = (
    DATA_READ_MISSES,
    DATA_WRITE_MISSES,
    DATA_WRITEBACKS,
    INSTRUCTION_READ_MISSES,
)

# Tensors the model's two passes leave out, by name prefix
# `forecast_head` uses every tensor but the fft and time heads', and `reconstruction_heads` every tensor but the forecast mapping's
PLAIN_EXCLUDED = ("decoder_with_head.fft_head.", "decoder_with_head.time_head.")
HIDDEN_EXCLUDED = ("decoder_with_head.forecast_mapping.",)

# The hidden set is read once per copy
HIDDEN_COPIES = 12


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class CacheCounters:
    """The counters of one Spike run.

    Attributes
    ----------
    data_read_misses : int
        Data cache read misses.
    data_write_misses : int
        Data cache write misses, which fetch the line.
    data_writebacks : int
        Dirty data lines written back to memory.
    instruction_misses : int
        Instruction cache read misses.
    """

    data_read_misses: int
    data_write_misses: int
    data_writebacks: int
    instruction_misses: int


@dataclass(frozen=True)
class Traffic:
    """One window's memory traffic.

    Attributes
    ----------
    counters : CacheCounters
        The window's own misses and writebacks.
    data_bytes : int
        Data cache traffic: (read misses + write misses + writebacks) lines.
    instruction_bytes : int
        Instruction cache traffic: its misses' lines.
    total_bytes : int
        Both added up.
    """

    counters: CacheCounters
    data_bytes: int
    instruction_bytes: int
    total_bytes: int


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_cache_report(path: Path) -> CacheCounters:
    """Read the miss and writeback counters from a Spike run's stdout.

    Parameters
    ----------
    path : Path
        The run's stdout, which holds Spike's cache report as `<name>: <value>` lines.

    Returns
    -------
    CacheCounters
        The four counters the traffic needs.

    Raises
    ------
    ProfileError
        If a counter is missing, repeated or not a number.
    """

    values: dict[str, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        name, separator, value = line.partition(":")
        if not separator or name not in COUNTERS:
            continue
        if name in values:
            raise ProfileError(f"{path}: '{name}' appears twice")
        try:
            values[name] = int(value.strip())
        except ValueError:
            raise ProfileError(f"{path}: '{name}' is not a number: {value!r}") from None
    missing = [name for name in COUNTERS if name not in values]
    if missing:
        raise ProfileError(
            f"{path}: no cache counter {missing}; was Spike run with the cache flags?"
        )
    return CacheCounters(
        values[DATA_READ_MISSES],
        values[DATA_WRITE_MISSES],
        values[DATA_WRITEBACKS],
        values[INSTRUCTION_READ_MISSES],
    )


def window_traffic(longer: CacheCounters, shorter: CacheCounters) -> Traffic:
    """Subtract a shorter run's counters from a longer one's and turn them into bytes.

    Parameters
    ----------
    longer : CacheCounters
        The run `--windows M-1:M+1`.
    shorter : CacheCounters
        The run `--windows M-1:M`.

    Returns
    -------
    Traffic
        The measured window's misses, writebacks and bytes.

    Raises
    ------
    ProfileError
        If a counter went down from the shorter run to the longer one.
    """

    counters = CacheCounters(
        longer.data_read_misses - shorter.data_read_misses,
        longer.data_write_misses - shorter.data_write_misses,
        longer.data_writebacks - shorter.data_writebacks,
        longer.instruction_misses - shorter.instruction_misses,
    )
    if min(vars(counters).values()) < 0:
        raise ProfileError(
            f"a cache counter went down between the two runs: {counters}"
        )
    # Each miss pulls one line in and each writeback pushes one out
    # Store misses count as well, because Spike's cache fetches the whole line on a store miss (write-allocate)
    # https://github.com/riscv-software-src/riscv-isa-sim/blob/master/riscv/cachesim.cc (`cache_sim_t::access`)
    data_lines = (
        counters.data_read_misses
        + counters.data_write_misses
        + counters.data_writebacks
    )
    data_bytes = data_lines * LINE_BYTES
    instruction_bytes = counters.instruction_misses * LINE_BYTES
    return Traffic(
        counters, data_bytes, instruction_bytes, data_bytes + instruction_bytes
    )


def traffic_floor(tensors: TensorSizes) -> int:
    """Bytes of weights one window must stream through the data cache.

    The weights are far larger than the cache, so each pass reads the tensors it uses from memory.
    floor = bytes(plain set) + HIDDEN_COPIES * bytes(hidden set)

    Parameters
    ----------
    tensors : TensorSizes
        Each tensor's bytes, from the generated table.

    Returns
    -------
    int
        The floor in bytes.
    """

    plain = sum(
        size for name, size in tensors.items() if not name.startswith(PLAIN_EXCLUDED)
    )
    hidden = sum(
        size for name, size in tensors.items() if not name.startswith(HIDDEN_EXCLUDED)
    )
    return plain + HIDDEN_COPIES * hidden
