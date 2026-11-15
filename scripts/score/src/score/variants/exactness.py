"""Compare a variant's per-window errors with the baseline's, text for text."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass, field
from pathlib import Path

from score.benchmark.results import ERRORS_FILE, find_results, read_identity
from score.benchmark.scope import Scope, Series, scope_range
from score.benchmark.units import (
    HOST_CHUNK_CHANNEL_WINDOWS,
    AccuracyUnit,
    accuracy_units,
)

# Internal
from score.build_record import BASELINE_NAME
from score.variants.state import SCOPES, label_folder, read_selection


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass
class Exactness:
    """How a variant's errors compare with the baseline's on the windows it ran.

    Attributes
    ----------
    windows_compared : int
        The windows whose row both labels hold.
    windows_differing : int
        The compared windows whose row text differs.
    left_out : list[str]
        The selected series with screen units missing, which were not compared.
    problems : list[str]
        What kept a comparison from being made.
    """

    windows_compared: int = 0
    windows_differing: int = 0
    left_out: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _read_rows(
    folder: Path, found: list[AccuracyUnit], start: int, stop: int
) -> dict[int, str]:
    """Read the error rows of windows `[start, stop)` from the units that hold them.

    Parameters
    ----------
    folder : Path
        The label's folder.
    found : list[AccuracyUnit]
        The label's units of one series that have a result.
    start : int
        The first window.
    stop : int
        One past the last window.

    Returns
    -------
    dict[int, str]
        Each window's row as the program printed it.
    """

    rows: dict[int, str] = {}
    for unit in found:
        if unit.stop <= start or unit.start >= stop:
            continue
        path = folder / unit.folder / ERRORS_FILE
        with path.open() as file:
            # The header names the heads, and both labels run the same program
            if next(file, None) is None:
                raise ValueError(f"{path}: no header line")
            for line in file:
                text = line.rstrip("\n")
                window = int(text.split(",", 1)[0])
                if start <= window < stop:
                    rows[window] = text
    return rows


def measure_exactness(data: Path, label: str, working: list[Series]) -> Exactness:
    """Compare a label's error rows with the baseline's on the windows the label ran.

    Parameters
    ----------
    data : Path
        The folder with the data.
    label : str
        The variant's label.
    working : list[Series]
        Every working series, which the label's selection picks from.

    Returns
    -------
    Exactness
        The windows compared and differing, the series left out, and the problems.

    Raises
    ------
    ValueError
        When `selection.json` or a `label.json` can't be used, or an errors file isn't a table of rows.
    OSError
        When a file can't be read.
    """

    # Step 1: the label's selection and identity, and the baseline's identity, which says how its units were cut
    folder = label_folder(data, label)
    baseline_folder = label_folder(data, BASELINE_NAME)
    selection = read_selection(folder)
    identity = read_identity(folder)
    baseline = read_identity(baseline_folder)
    result = Exactness()
    if selection is None or identity is None:
        raise ValueError(f"{label} has no selection.json or label.json")
    if baseline is None:
        result.problems.append("the baseline has no label.json")
        return result
    if selection.window_cap != identity.window_cap:
        result.problems.append(
            f"selection.json has window cap {selection.window_cap} and label.json {identity.window_cap}"
        )
        return result

    # Step 2: each selected univariate series, compared when all its screen units have a result
    baseline_scopes = [Scope(name) for name in baseline.scopes]
    for one in working:
        if one.stem not in selection.series:
            continue
        windows = scope_range(one, Scope.SCREEN, selection.window_cap)
        if windows is None:
            continue
        units = accuracy_units(
            [one], SCOPES, selection.window_cap, HOST_CHUNK_CHANNEL_WINDOWS
        )
        found, _ = find_results(folder, units, identity.sha256)
        if len(found) != len(units):
            result.left_out.append(one.stem)
            continue
        baseline_units = accuracy_units(
            [one],
            baseline_scopes,
            baseline.window_cap,
            baseline.host_chunk_channel_windows,
        )
        baseline_found, _ = find_results(
            baseline_folder, baseline_units, baseline.sha256
        )

        # Step 3: the rows of both labels for the same windows, matched by window number since the unit boundaries differ
        ours = _read_rows(folder, list(found), *windows)
        theirs = _read_rows(baseline_folder, list(baseline_found), *windows)
        missing = [number for number in range(*windows) if number not in theirs]
        if missing:
            result.problems.append(
                f"{one.stem}: the baseline lacks {len(missing)} of windows {windows[0]} to {windows[1] - 1}"
            )
        absent = [number for number in range(*windows) if number not in ours]
        if absent:
            result.problems.append(
                f"{one.stem}: the label lacks {len(absent)} of windows {windows[0]} to {windows[1] - 1}"
            )
        for number, text in ours.items():
            if number in theirs:
                result.windows_compared += 1
                if text != theirs[number]:
                    result.windows_differing += 1
    return result
