"""Spike's `-g` PC histogram, and the per-address difference of two runs.

Spike counts every instruction of the whole run, pk's boot and the rows' reading included, and cannot reset the counts.
A window's own counts are therefore a longer run minus a shorter one that stops one window earlier.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path

# Internal
from score.profile.errors import ProfileError
from score.spike_check import HISTOGRAM_HEADER, HISTOGRAM_LINE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Instructions executed per address
Histogram = dict[int, int]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_histogram(path: Path) -> Histogram:
    """Read the `-g` histogram out of a Spike run's stderr.

    Parameters
    ----------
    path : Path
        The run's stderr, which holds whatever the program printed there before the histogram.

    Returns
    -------
    Histogram
        Instructions executed at each program counter.

    Raises
    ------
    ProfileError
        If the header is missing, a line after it is not `<hex pc> <count>`, an address repeats, or the header's size is not the number of lines.
    """

    # Step 1: find the header, which ends the program's own output
    lines = path.read_text(errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(HISTOGRAM_HEADER)]
    if len(starts) != 1:
        raise ProfileError(
            f"{path}: expected one '{HISTOGRAM_HEADER}' line, found {len(starts)}"
        )
    declared = int(lines[starts[0]].removeprefix(HISTOGRAM_HEADER))

    # Step 2: read one line per address, none skipped
    histogram: Histogram = {}
    for line in lines[starts[0] + 1 :]:
        match = HISTOGRAM_LINE.match(line)
        if match is None:
            raise ProfileError(f"{path}: not a histogram line: {line!r}")
        address = int(match.group(1), 16)
        if address in histogram:
            raise ProfileError(f"{path}: address {address:#x} appears twice")
        histogram[address] = int(match.group(2))

    # Step 3: the header says how many lines to expect, so a cut-off file shows
    if len(histogram) != declared:
        raise ProfileError(
            f"{path}: header says {declared} addresses, file has {len(histogram)}"
        )
    return histogram


def subtract(longer: Histogram, shorter: Histogram) -> Histogram:
    """Subtract a shorter run's counts from a longer run's, address by address.

    Differences may be negative at single addresses: the allocator's free path shifts a few instructions between neighbours.
    Callers check totals per function, never per address.

    Parameters
    ----------
    longer : Histogram
        The run that stops one window later.
    shorter : Histogram
        The run it extends.

    Returns
    -------
    Histogram
        The difference at every address either run touched, zeros dropped.
    """

    difference: Histogram = {}
    for address in longer.keys() | shorter.keys():
        delta = longer.get(address, 0) - shorter.get(address, 0)
        if delta != 0:
            difference[address] = delta
    return difference


def largest_first(counts: dict[str, int]) -> dict[str, int]:
    """Order counts from the largest, ties by name, so a rerun writes the same table.

    Parameters
    ----------
    counts : dict[str, int]
        Counts by name.

    Returns
    -------
    dict[str, int]
        The same counts, largest first.
    """

    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
