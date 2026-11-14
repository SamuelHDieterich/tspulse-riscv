"""Cycles per window, as a range, from the instruction mix and the OpenC906 weight table."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
from dataclasses import dataclass
from importlib import resources

# Internal
from score.profile.errors import ProfileError


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The weight table, shipped inside the package. Its delays come from the XuanTie OpenC906 user manual, §4.1 (Tables 4.1, 4.2, 4.4 and 4.5, pp. 27–38): https://cs107e.github.io/readings/XuanTie-OpenC906-UserManual.pdf
WEIGHT_TABLE = "c906.csv"

# The table's columns
MNEMONIC_COLUMN = "mnemonic"
LOW_COLUMN = "low"
HIGH_COLUMN = "high"
SOURCE_COLUMN = "source"

# Each compressed form runs as its base instruction: same unit, and the manual's RVC table (Table 4.6, pp.39-40) lists the same delays as the base tables
COMPRESSED_BASE = {
    "c.add": "add",
    "c.addw": "addw",
    "c.addi": "addi",
    "c.addiw": "addiw",
    "c.sub": "sub",
    "c.subw": "subw",
    "c.addi16sp": "addi",
    "c.addi4spn": "addi",
    "c.and": "and",
    "c.andi": "andi",
    "c.or": "or",
    "c.xor": "xor",
    "c.slli": "slli",
    "c.srli": "srli",
    "c.srai": "srai",
    "c.mv": "add",
    "c.li": "addi",
    "c.lui": "lui",
    "c.beqz": "beq",
    "c.bnez": "bne",
    "c.j": "jal",
    "c.jr": "jalr",
    "c.jalr": "jalr",
    "c.lw": "lw",
    "c.sw": "sw",
    "c.lwsp": "lw",
    "c.swsp": "sw",
    "c.ld": "ld",
    "c.sd": "sd",
    "c.ldsp": "ld",
    "c.sdsp": "sd",
    "c.fld": "fld",
    "c.fsd": "fsd",
    "c.fldsp": "fld",
    "c.fsdsp": "fsd",
    "c.nop": "addi",
    "c.ebreak": "ebreak",
}


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Weight:
    """One instruction's cycles.

    Attributes
    ----------
    low : int
        The issue cost on the single-issue core, or the minimum latency for divide, remainder and square root.
    high : int
        The maximum latency, as if the instruction waited for the one before.
    source : str
        The manual's table and page the row comes from.
    """

    low: int
    high: int
    source: str


@dataclass(frozen=True)
class CycleRange:
    """The cycles a window costs, between the two ends.

    Attributes
    ----------
    low : int
        Every instruction at its low weight.
    high : int
        Every instruction at its high weight.
    """

    low: int
    high: int


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def load_weights() -> dict[str, Weight]:
    """Read the weight table that ships with the package.

    Returns
    -------
    dict[str, Weight]
        The weight of each base mnemonic.
    """

    text = resources.files("score.profile").joinpath(WEIGHT_TABLE).read_text()
    rows = csv.DictReader(text.splitlines())
    return {
        row[MNEMONIC_COLUMN]: Weight(
            int(row[LOW_COLUMN]), int(row[HIGH_COLUMN]), row[SOURCE_COLUMN]
        )
        for row in rows
    }


def cycle_range(mix: dict[str, int], weights: dict[str, Weight]) -> CycleRange:
    """Turn a window's instruction mix into a cycle range.

    Parameters
    ----------
    mix : dict[str, int]
        Instructions per mnemonic, from `mix_counts`.
    weights : dict[str, Weight]
        The weight of each base mnemonic, from `load_weights`.

    Returns
    -------
    CycleRange
        The cycles at the low and the high end.

    Raises
    ------
    ProfileError
        If a mnemonic, once mapped to its base instruction, has no row in the table. The message lists them all.
    """

    # Step 1: refuse a build that runs an instruction the table doesn't cover, instead of counting it as free
    missing = sorted(
        mnemonic
        for mnemonic in mix
        if COMPRESSED_BASE.get(mnemonic, mnemonic) not in weights
    )
    if missing:
        raise ProfileError(f"no weight in {WEIGHT_TABLE} for: {', '.join(missing)}")

    # Step 2: add up each mnemonic's count times its weight
    # cycles = sum(count * cycles per instruction), the CPI equation without the memory term
    # https://en.wikipedia.org/wiki/Cycles_per_instruction
    low = 0
    high = 0
    for mnemonic, count in mix.items():
        weight = weights[COMPRESSED_BASE.get(mnemonic, mnemonic)]
        low += count * weight.low
        high += count * weight.high
    return CycleRange(low, high)
