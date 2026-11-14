"""What a `time-host` unit saves, when a saved unit counts as done, and the two tables built from them.

Each unit's JSON sits in the machine's folder and marks the unit done; `runs.csv` has one row per unit and `summary.csv` the median over repeats.
In `runs.csv`, a one-core C row's `cpu_seconds` covers the whole run on every window, start-up included, while its `timed_seconds` is the difference T_N − T_1 of the two runs.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import io
import json
import shutil
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Internal
from score.spike_results import _replace
from score.timing.units import B1_T1, B128_WHOLE_PC, TORCH, Unit


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The tables in the machine's folder
RUNS_FILE = "runs.csv"
SUMMARY_FILE = "summary.csv"

# The provenance fields that decide whether a saved unit still counts. The commit, tree and dirty flag are recorded but not compared, so uncommitted work doesn't rerun every unit
SKIP_FIELDS = ("pin_cpu", "parallel", "sha256", "torch", "granite_tsfm")

# The result fields copied into `runs.csv`, in column order, followed by the 1-minute load average.
# On one C core, `cpu_seconds` is the whole `0:N` run, start-up included, while `timed_seconds` is the difference T_N − T_1, so CPU reads above wall time there
RUNS_COLUMNS = (
    "unit",
    "repeat",
    "series",
    "runner",
    "setup",
    "channels",
    "windows",
    "seconds_per_window",
    "timed_seconds",
    "cpu_seconds",
    "max_rss_kib",
    "unit_seconds",
)
LOAD_COLUMN = "load_average_1min"

# `summary.csv`'s columns. A ratio is this row's seconds per channel-window over the PyTorch setup's, so above 1 is slower
SUMMARY_COLUMNS = (
    "series",
    "runner",
    "setup",
    "repeats",
    "channels",
    "windows",
    "seconds_per_window",
    "seconds_per_channel_window",
    "hz_per_channel",
    "windows_per_second",
    "ratio_to_torch_b1_t1",
    "ratio_to_torch_b128_whole_pc",
)

# The PyTorch setups the ratios divide by, with the column each fills
RATIO_REFERENCES = {
    "ratio_to_torch_b1_t1": B1_T1,
    "ratio_to_torch_b128_whole_pc": B128_WHOLE_PC,
}


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Measurement:
    """What a unit measured, in the shape every runner's result shares.

    Attributes
    ----------
    windows : int
        Windows run, as the runner counted them.
    seconds_per_window : float
        Wall seconds per window.
    timed_seconds : float
        The wall seconds `seconds_per_window` comes from.
    cpu_seconds : float
        CPU seconds of the timed work: on one C core the whole `0:N` run, start-up included, though the wall time is the difference T_N − T_1; on the whole PC the sum over the C's processes; for PyTorch the process during its timed call.
    max_rss_kib : int
        The largest peak resident memory of any one process, in KiB: each C process's from GNU time's `%M`, the PyTorch process's from its `VmHWM`.
    details : dict[str, Any]
        The runner's raw numbers and commands.
    """

    windows: int
    seconds_per_window: float
    timed_seconds: float
    cpu_seconds: float
    max_rss_kib: int
    details: dict[str, Any]


@dataclass(frozen=True)
class Provenance:
    """Where, from what and how a unit ran.

    Attributes
    ----------
    machine : str
        The label of the computer.
    commit : str
        `git rev-parse HEAD` of the worktree the command ran in.
    tree : str
        `git rev-parse HEAD^{tree}`, which a rebase with an empty diff leaves unchanged.
    dirty : bool
        Whether the worktree had uncommitted changes.
    pin_cpu : int
        The CPU the pinned setups ran on.
    parallel : int
        P, the processes or threads of the whole-PC setups.
    command : str
        The `time-host` command line, shell-quoted.
    sha256 : str or None
        The C binary's SHA-256, None for a PyTorch unit.
    torch : str or None
        PyTorch's version, None for a C unit.
    granite_tsfm : str or None
        granite-tsfm's version, None for a C unit.
    """

    machine: str
    commit: str
    tree: str
    dirty: bool
    pin_cpu: int
    parallel: int
    command: str
    sha256: str | None
    torch: str | None
    granite_tsfm: str | None


@dataclass(frozen=True)
class UnitResult:
    """A finished unit, in the shape every runner shares.

    Attributes
    ----------
    unit : str
        The unit's name.
    repeat : int
        Which repeat, from 1.
    series : str
        The series' short name.
    runner : str
        Who ran the model.
    setup : str
        How it ran.
    channels : int
        How many channels the series has.
    windows : int
        Windows run, as the runner counted them.
    seconds_per_window : float
        Wall seconds per window: on one C core with start-up taken out, on the whole PC the group's wall time over the windows, for PyTorch the time inside `_forward` over the windows.
    timed_seconds : float
        The wall seconds `seconds_per_window` comes from.
    cpu_seconds : float
        CPU seconds of the timed work, as `Measurement.cpu_seconds` defines it: on one C core this is the whole `0:N` run, not the difference the wall time uses.
    max_rss_kib : int
        The largest peak resident memory of any one process, in KiB: each C process's from GNU time's `%M`, the PyTorch process's from its `VmHWM`.
    load_average : list[float]
        The 1, 5 and 15 minute load averages just before the unit started.
    unit_seconds : float
        The unit's whole wall time, which the ETA uses.
    details : dict[str, Any]
        The runner's raw numbers and commands.
    provenance : Provenance
        Where, from what and how it ran.
    """

    unit: str
    repeat: int
    series: str
    runner: str
    setup: str
    channels: int
    windows: int
    seconds_per_window: float
    timed_seconds: float
    cpu_seconds: float
    max_rss_kib: int
    load_average: list[float]
    unit_seconds: float
    details: dict[str, Any]
    provenance: Provenance


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def skip_key(provenance: dict[str, Any]) -> dict[str, Any]:
    """Pick the provenance fields that decide whether a saved unit still counts.

    Parameters
    ----------
    provenance : dict[str, Any]
        A provenance, as saved.

    Returns
    -------
    dict[str, Any]
        Its `SKIP_FIELDS`, missing ones as None.
    """

    return {name: provenance.get(name) for name in SKIP_FIELDS}


def write_result(folder: Path, unit: Unit, result: UnitResult) -> None:
    """Save a unit's JSON, whose presence marks the unit as done.

    Parameters
    ----------
    folder : Path
        The machine's folder.
    unit : Unit
        The unit, which names the file.
    result : UnitResult
        What it measured.
    """

    folder.mkdir(parents=True, exist_ok=True)
    _replace(folder / unit.file_name, json.dumps(asdict(result), indent=2) + "\n")


def stored_result(
    folder: Path, unit: Unit, provenance: Provenance
) -> dict[str, Any] | None:
    """Read a unit's saved JSON if it was made with the same skip key.

    Parameters
    ----------
    folder : Path
        The machine's folder.
    unit : Unit
        The unit.
    provenance : Provenance
        This call's provenance for the unit.

    Returns
    -------
    dict[str, Any] or None
        The saved result, or None when there is none or its skip key differs, so the unit reruns.

    Raises
    ------
    ValueError
        When the file isn't a JSON object with a provenance.
    OSError
        When the file exists but can't be read.
    """

    path = folder / unit.file_name
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(stored, dict) or not isinstance(stored.get("provenance"), dict):
        raise ValueError(f"{path}: not a unit result")
    if skip_key(stored["provenance"]) != skip_key(asdict(provenance)):
        return None
    return stored


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    """Write a table whole, so a reader never sees half of it.

    Parameters
    ----------
    path : Path
        The CSV.
    columns : tuple[str, ...]
        Its columns, in order.
    rows : list[dict[str, Any]]
        One dict per row, keyed by column.
    """

    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _replace(path, text.getvalue())


def _summary_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Take the median over repeats of each series, runner and setup, and derive the rates and ratios from it.

    Parameters
    ----------
    results : list[dict[str, Any]]
        The saved unit results, in run order.

    Returns
    -------
    list[dict[str, Any]]
        One row per series, runner and setup, in the order they first appear.
    """

    # Step 1: the repeats of each measurement, in run order
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        key = (result["series"], result["runner"], result["setup"])
        groups.setdefault(key, []).append(result)

    # Step 2: the median seconds per window, and the rates from it
    rows = []
    for (series, runner, setup), repeats in groups.items():
        # Median of the repeats: https://docs.python.org/3/library/statistics.html#statistics.median
        seconds = statistics.median(one["seconds_per_window"] for one in repeats)
        channels = repeats[0]["channels"]

        # Hz per channel = channel-windows / seconds, since each window moves every channel on by one reading
        per_channel_window = seconds / channels
        rows.append(
            {
                "series": series,
                "runner": runner,
                "setup": setup,
                "repeats": len(repeats),
                "channels": channels,
                "windows": repeats[0]["windows"],
                "seconds_per_window": seconds,
                "seconds_per_channel_window": per_channel_window,
                "hz_per_channel": 1 / per_channel_window,
                "windows_per_second": 1 / seconds,
            }
        )

    # Step 3: each row against the same series' PyTorch setups, left empty when that setup has no result yet
    reference = {
        (row["series"], row["setup"]): row["seconds_per_channel_window"]
        for row in rows
        if row["runner"] == TORCH
    }
    for row in rows:
        for column, setup in RATIO_REFERENCES.items():
            divisor = reference.get((row["series"], setup))
            row[column] = (
                "" if divisor is None else row["seconds_per_channel_window"] / divisor
            )
    return rows


def write_tables(folder: Path, results: list[dict[str, Any]]) -> Path:
    """Rewrite the machine's `runs.csv` and `summary.csv` from its saved results.

    Parameters
    ----------
    folder : Path
        The machine's folder.
    results : list[dict[str, Any]]
        The saved unit results that still count, in run order.

    Returns
    -------
    Path
        The summary.
    """

    folder.mkdir(parents=True, exist_ok=True)
    runs = [
        {column: result[column] for column in RUNS_COLUMNS}
        | {LOAD_COLUMN: result["load_average"][0]}
        for result in results
    ]
    _write_csv(folder / RUNS_FILE, (*RUNS_COLUMNS, LOAD_COLUMN), runs)
    summary = folder / SUMMARY_FILE
    _write_csv(summary, SUMMARY_COLUMNS, _summary_rows(results))
    return summary


def publish_summary(summary: Path, target: Path) -> None:
    """Copy the summary to the tracked results folder.

    Parameters
    ----------
    summary : Path
        The machine's `summary.csv`.
    target : Path
        Where the tracked copy goes.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(summary, target)
