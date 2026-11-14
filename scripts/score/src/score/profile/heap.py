"""The program's heap allocations, as formulas of the channel count."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class HeapAllocation:
    """One `malloc` whose size grows with the channel count.

    Attributes
    ----------
    file : str
        The source file, relative to the `tspulse` folder.
    call : str
        The text of the `malloc` call, which finds its line.
    formula : str
        Its bytes as a formula of `channels`.
    per_channel : int
        Its bytes for each channel.
    what : str
        What the buffer holds.
    """

    file: str
    call: str
    formula: str
    per_channel: int
    what: str


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The sizes the heap allocations are made of: bytes per element and the row count of one window
DOUBLE_BYTES = 8
WINDOW_LENGTH = 513

# Every allocation the program makes
HEAP_ALLOCATIONS = (
    HeapAllocation(
        "model/history.c",
        "malloc((size_t)channels * WINDOW_LENGTH * sizeof(double))",
        f"channels * {WINDOW_LENGTH} * {DOUBLE_BYTES}",
        WINDOW_LENGTH * DOUBLE_BYTES,
        "the history's ring of raw values per channel",
    ),
    HeapAllocation(
        "io/rows.c",
        "malloc(2 * channels * sizeof(double))",
        f"2 * channels * {DOUBLE_BYTES}",
        2 * DOUBLE_BYTES,
        "the reader's mean and std",
    ),
    HeapAllocation(
        "tspulse.c",
        "malloc(reader->channels * sizeof(double))",
        f"channels * {DOUBLE_BYTES}",
        DOUBLE_BYTES,
        "one row as read",
    ),
)
