"""What a benchmark label and each of its units save, and the tidy instruction table.

A label's folder holds `label.json`, which fixes the label's binaries and settings at its first run, then one folder per unit.
A unit's `result.json` marks it done; a unit without one reruns.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import io
import json
import shutil
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

# External
import numpy as np

# Internal
from score.benchmark.units import (
    ACCURACY_FOLDER,
    INSTRUCTIONS_FOLDER,
    SCORES_FOLDER,
    AccuracyUnit,
    InstructionUnit,
)
from score.kind import Kind
from score.spike_results import RESULT_FILE, _replace


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The label's identity, in the label's folder
LABEL_FILE = "label.json"

# The first column of an errors CSV, which the head columns follow
WINDOW_COLUMN = "window"

# What an accuracy unit keeps
ERRORS_FILE = "errors.csv"
STDERR_FILE = "stderr"

# What an instruction unit keeps, beside the host's and Spike's errors under `check-spike`'s names
COUNTS_FILE = "counts.csv"
SPIKE_STDOUT_FILE = "spike.stdout"

# An instruction unit's `identical`: the byte comparison's verdict, or `not checked` when the host check is off
NotChecked = Literal["not checked"]
NOT_CHECKED: NotChecked = "not checked"
Identical = bool | NotChecked

# The `result.json` fields that go into a machine's instruction table, in column order. The commands and hashes stay in `result.json` only
INSTRUCTION_COLUMNS = (
    "series",
    "kind",
    "channels",
    "window",
    "program_instructions",
    "identical",
    "spike_seconds",
    "host_seconds",
    "machine",
    "commit",
    "tree",
    "dirty",
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Platform(StrEnum):
    """Where a program runs.

    Attributes
    ----------
    HOST : str
        Natively, on the host build.
    SPIKE : str
        Under Spike, on the rv64 build.
    """

    HOST = "host"
    SPIKE = "spike"


class HostCheck(StrEnum):
    """Whether an instruction unit also runs its window on the host and compares the errors.

    Attributes
    ----------
    IDENTICAL : str
        Run it and require the two errors CSVs to be byte-identical.
    OFF : str
        Don't, for a build no host build matches.
    """

    IDENTICAL = "identical"
    OFF = "off"


@dataclass(frozen=True)
class LabelIdentity:
    """What fixes a label: its binaries and the settings that decide its units and their results.

    `--series` and `--shard` only filter units, so they are not part of it.

    Attributes
    ----------
    sha256 : dict[str, str]
        The hashes of the host build's `tspulse`, the rv64 build's `tspulse` and pk, by role.
    scopes : list[str]
        The accuracy scopes.
    window_cap : int or None
        The most windows kept of each scope range, None for all.
    host_chunk_channel_windows : int
        Channel-windows per accuracy unit on the host.
    spike_chunk_channel_windows : int
        Channel-windows per accuracy unit under Spike.
    count_windows : str
        Which windows the instruction stage counts.
    isa : str
        Spike's ISA string.
    accuracy_on : str
        Where the accuracy units run.
    host_check : str
        Whether instruction units are checked against the host.
    spike_options : list[str]
        The extra options Spike gets on the counted windows.
    """

    sha256: dict[str, str]
    scopes: list[str]
    window_cap: int | None
    host_chunk_channel_windows: int
    spike_chunk_channel_windows: int
    count_windows: str
    isa: str
    accuracy_on: str
    host_check: str
    spike_options: list[str]

    def differences(self, stored: "LabelIdentity") -> list[str]:
        """Name the fields that differ from a stored identity.

        Parameters
        ----------
        stored : LabelIdentity
            The identity in `label.json`.

        Returns
        -------
        list[str]
            One `field: stored → now` per differing field, a hash per role, empty when they match.
        """

        found = []
        for role in sorted(self.sha256.keys() | stored.sha256.keys()):
            if self.sha256.get(role) != stored.sha256.get(role):
                found.append(
                    f"sha256 of {role}: {stored.sha256.get(role)} → {self.sha256.get(role)}"
                )
        for field in fields(self):
            if field.name == "sha256":
                continue
            ours, theirs = getattr(self, field.name), getattr(stored, field.name)
            if ours != theirs:
                found.append(f"{field.name}: {theirs} → {ours}")
        return found


@dataclass(frozen=True)
class Provenance:
    """Where and from what a unit ran, the same for every unit of a call.

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
    sha256 : dict[str, str]
        The binaries' hashes, as in `label.json`.
    """

    machine: str
    commit: str
    tree: str
    dirty: bool
    sha256: dict[str, str]


@dataclass(frozen=True)
class AccuracyResult:
    """An accuracy unit's outcome; its errors sit beside it in `errors.csv`.

    Attributes
    ----------
    label : str
        The benchmark label.
    series : str
        The series' CSV name without `.csv`.
    kind : Kind
        The archive it comes from.
    channels : int
        How many channels it has.
    start : int
        The first window.
    stop : int
        One past the last window.
    ran_on : Platform
        Where the program ran.
    seconds : float
        The run's wall time.
    command : str
        The exact command line, shell-quoted.
    provenance : Provenance
        The machine, the worktree's state and the binaries' hashes.
    """

    label: str
    series: str
    kind: Kind
    channels: int
    start: int
    stop: int
    ran_on: Platform
    seconds: float
    command: str
    provenance: Provenance


@dataclass(frozen=True)
class InstructionResult:
    """An instruction unit's outcome; its errors, counts and Spike's output sit beside it.

    Attributes
    ----------
    label : str
        The benchmark label.
    series : str
        The series' CSV name without `.csv`.
    kind : Kind
        The archive it comes from.
    channels : int
        How many channels it has.
    window : int
        The window counted.
    program_instructions : int
        The instructions of the window's one model run, from `rdinstret` around it.
    identical : bool or "not checked"
        Whether the host's and Spike's errors CSVs are byte-identical, or `not checked` with the host check off.
    spike_seconds : float
        The Spike run's wall time.
    host_seconds : float or None
        The host run's wall time, None with the host check off.
    spike_command : str
        The Spike run's exact command line, shell-quoted.
    host_command : str or None
        The host run's exact command line, None with the host check off.
    provenance : Provenance
        The machine, the worktree's state and the binaries' hashes.
    """

    label: str
    series: str
    kind: Kind
    channels: int
    window: int
    program_instructions: int
    identical: Identical
    spike_seconds: float
    host_seconds: float | None
    spike_command: str
    host_command: str | None
    provenance: Provenance


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_identity(label_folder: Path) -> LabelIdentity | None:
    """Read a label's `label.json`.

    Parameters
    ----------
    label_folder : Path
        The label's folder.

    Returns
    -------
    LabelIdentity or None
        The stored identity, or None when the label has none yet.

    Raises
    ------
    ValueError
        When the file isn't JSON or its fields don't match `LabelIdentity`'s.
    OSError
        When the file exists but can't be read.
    """

    path = label_folder / LABEL_FILE
    if not path.is_file():
        return None
    try:
        return LabelIdentity(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from error


def create_identity(label_folder: Path, identity: LabelIdentity) -> bool:
    """Write a label's `label.json` unless one exists.

    Parameters
    ----------
    label_folder : Path
        The label's folder.
    identity : LabelIdentity
        The identity to fix.

    Returns
    -------
    bool
        True when this call wrote it, False when another call already had, which the caller then compares against.
    """

    # Created exclusively, so two machines starting the same label at once can't both fix it
    label_folder.mkdir(parents=True, exist_ok=True)
    try:
        with (label_folder / LABEL_FILE).open("x") as file:
            file.write(json.dumps(asdict(identity), indent=2) + "\n")
    except FileExistsError:
        return False
    return True


def write_result(folder: Path, result: AccuracyResult | InstructionResult) -> None:
    """Save a unit's `result.json`, whose presence marks the unit as done.

    Parameters
    ----------
    folder : Path
        The unit's result folder.
    result : AccuracyResult or InstructionResult
        What the unit found.
    """

    _replace(folder / RESULT_FILE, json.dumps(asdict(result), indent=2) + "\n")


def stored_result(folder: Path, sha256: dict[str, str]) -> dict[str, Any] | None:
    """Read a unit's `result.json` if it was made by the label's binaries.

    Parameters
    ----------
    folder : Path
        The unit's result folder.
    sha256 : dict[str, str]
        The label's binary hashes.

    Returns
    -------
    dict[str, Any] or None
        The stored result's fields, or None when there is none or its hashes differ, so the unit reruns.

    Raises
    ------
    ValueError
        When the file isn't a JSON object.
    OSError
        When the file exists but can't be read.
    """

    path = folder / RESULT_FILE
    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(stored, dict):
        raise ValueError(f"{path}: not a JSON object")
    provenance = stored.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("sha256") != sha256:
        return None
    return stored


def read_errors(path: Path, start: int, stop: int) -> dict[str, np.ndarray]:
    """Read an accuracy unit's errors CSV, taking each head's column by its header name.

    Parameters
    ----------
    path : Path
        The unit's `errors.csv`.
    start : int
        The unit's first window.
    stop : int
        One past the unit's last window.

    Returns
    -------
    dict[str, np.ndarray]
        Per head in the file's header, one float32 error per window `[start, stop)`.

    Raises
    ------
    ValueError
        When the header doesn't start with `window`, or the windows are not exactly `[start, stop)` in order.
    OSError
        When the file can't be read.
    """

    # The program prints each float32 error with 9 significant digits, which read back as float32 exactly
    with path.open() as file:
        header = file.readline().strip().split(",")
    table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if header[0] != WINDOW_COLUMN or table.shape != (stop - start, len(header)):
        raise ValueError(f"{path} does not hold windows {start} to {stop - 1}")
    if not np.array_equal(table[:, 0], np.arange(start, stop)):
        raise ValueError(f"{path} does not hold windows {start} to {stop - 1} in order")
    return {
        head: table[:, column].astype(np.float32)
        for column, head in enumerate(header)
        if column
    }


def find_results[UnitT: (AccuracyUnit, InstructionUnit)](
    label_folder: Path, units: list[UnitT], sha256: dict[str, str]
) -> tuple[dict[UnitT, dict[str, Any]], list[str]]:
    """Read the `result.json` of every unit that has one made by the label's binaries.

    Parameters
    ----------
    label_folder : Path
        The label's folder.
    units : list[AccuracyUnit] or list[InstructionUnit]
        The units to look for.
    sha256 : dict[str, str]
        The label's binary hashes, from `label.json`.

    Returns
    -------
    tuple[dict, list[str]]
        The stored results of the units that are done, and one message per `result.json` that was rejected because its hashes differ from the label's or it can't be read. A rejected unit counts as not done.
    """

    found = {}
    problems = []
    for unit in units:
        folder = label_folder / unit.folder
        try:
            stored = stored_result(folder, sha256)
        except (OSError, ValueError) as error:
            problems.append(f"{unit.name}: unreadable result: {error}")
            continue
        if stored is not None:
            found[unit] = stored
        elif (folder / RESULT_FILE).is_file():
            problems.append(
                f"{unit.name}: result.json rejected, its binaries differ from label.json's"
            )
    return found, problems


def write_instruction_table(label_folder: Path, machine: str) -> tuple[Path, list[str]]:
    """Rebuild `instructions/<machine>.csv` from this machine's instruction results.

    Each machine writes only its own table, so machines sharing the folder never write the same file.

    Parameters
    ----------
    label_folder : Path
        The label's folder.
    machine : str
        The machine whose units go in.

    Returns
    -------
    tuple[Path, list[str]]
        The table, and one message per `result.json` left out because it can't be read or lacks a column, from any machine.

    Raises
    ------
    OSError
        When the table can't be written.
    """

    # Step 1: this machine's results. Units are `instructions/<stem>/w<window>/result.json`. A malformed one, perhaps another machine's half-synced file, is left out rather than stopping the run
    folder = label_folder / INSTRUCTIONS_FOLDER
    rows = []
    problems = []
    for path in folder.glob(f"*/*/{RESULT_FILE}"):
        try:
            stored = json.loads(path.read_text())
            provenance = stored["provenance"]
            if provenance["machine"] != machine:
                continue
            merged = {**stored, **provenance}
            rows.append({column: merged[column] for column in INSTRUCTION_COLUMNS})
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"{path}: {type(error).__name__}: {error}")
    rows.sort(key=lambda row: (row["series"], row["window"]))

    # Step 2: the table, replaced whole so a reader never sees half of it
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=INSTRUCTION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    folder.mkdir(parents=True, exist_ok=True)
    table = folder / f"{machine}.csv"
    _replace(table, text.getvalue())
    return table, problems


def delete_label_results(label_folder: Path) -> None:
    """Delete a label's identity, every machine's unit results and the scores made from them, keeping the logs.

    Parameters
    ----------
    label_folder : Path
        The label's folder.
    """

    (label_folder / LABEL_FILE).unlink(missing_ok=True)
    for name in (ACCURACY_FOLDER, INSTRUCTIONS_FOLDER, SCORES_FOLDER):
        shutil.rmtree(label_folder / name, ignore_errors=True)
