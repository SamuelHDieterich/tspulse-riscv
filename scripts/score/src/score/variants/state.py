"""What a variant's label holds on disk: the windows it was asked to run, its reference check, and whether it is done."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Internal
from score.benchmark.results import LabelIdentity, find_results, read_identity
from score.benchmark.scope import Scope, Series
from score.benchmark.tables import ACCURACY_TABLE, INSTRUCTIONS_TABLE, TABLES_FOLDER
from score.benchmark.units import (
    HOST_CHUNK_CHANNEL_WINDOWS,
    CountWindows,
    accuracy_units,
    instruction_units,
)
from score.build_record import BUILD_FILE
from score.layout import BENCHMARK_FOLDER
from score.profile.cli import WORK_SUBFOLDER
from score.spike_check import CLEAN_BINARY, HOST_ROLE, RV64_ROLE
from score.variants.stage_names import VariantStage


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The files the command keeps in a label's folder, beside the ones `run` writes
REFERENCE_FILE = "reference.json"
SELECTION_FILE = "selection.json"
COLLECT_FILE = "collect.json"

# The variant's one-row summary, in its tracked results folder
SUMMARY_FILE = "summary.csv"

# Every variant runs the same accuracy windows and counts the same windows, so their results line up with each other and with the baseline's
SCOPES = [Scope.SCREEN]
COUNT_WINDOWS = CountWindows.MIDDLE

# The field of `ReferenceCheck` that holds nested records, which its reader builds before the check itself
REFERENCE_WINDOWS_FIELD = "windows"

# The key of the profile's `build.json` holding the build's hashes
BUILD_SHA256_KEY = "sha256"

# What `missing_stages` says when a label's files can't be read
MISSING_UNREADABLE = "unreadable files"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Selection:
    """The windows a label was asked to run, kept in `selection.json` since `run` records neither.

    Attributes
    ----------
    series : list[str]
        The chosen series' stems, in `benchmark/series.csv` order.
    window_cap : int or None
        The most windows kept of each scope range, None for all.
    """

    series: list[str]
    window_cap: int | None


@dataclass(frozen=True)
class ReferenceWindow:
    """How one reference window came out, as `reference.json` keeps it.

    Attributes
    ----------
    window : str
        The window's name, `<series>/w<index>`.
    ok : bool
        Whether the window passed.
    first_failure : str or None
        The name of the first record that failed, None when none did.
    error : str
        Why the window couldn't be checked, empty when it could. A window can fail without a failing record, when the program or its dump can't be used.
    """

    window: str
    ok: bool
    first_failure: str | None
    error: str


@dataclass(frozen=True)
class ReferenceCheck:
    """A label's reference check, as `reference.json` keeps it.

    Attributes
    ----------
    windows : list[ReferenceWindow]
        One record per reference window.
    sha256 : dict[str, str]
        The host build's hashes, which tell whether the check still holds.
    """

    windows: list[ReferenceWindow]
    sha256: dict[str, str]


@dataclass(frozen=True)
class CollectRecord:
    """How a label's last collect went, as `collect.json` keeps it.

    Attributes
    ----------
    problems : int
        How many problems the collect found.
    sha256 : dict[str, str]
        The `label.json` hashes the collect read the units with.
    """

    problems: int
    sha256: dict[str, str]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def label_folder(data: Path, label: str) -> Path:
    """The label's folder in the data folder.

    Parameters
    ----------
    data : Path
        The folder with the data.
    label : str
        The label.

    Returns
    -------
    Path
        `results/benchmark/<label>` under the data folder.
    """

    return data / BENCHMARK_FOLDER / label


def tables_folder(top: Path, label: str) -> Path:
    """The label's tracked results folder.

    Parameters
    ----------
    top : Path
        The git top level.
    label : str
        The label.

    Returns
    -------
    Path
        `benchmark/results/<label>` under the git top level.
    """

    return top / TABLES_FOLDER / label


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object from a file.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    dict[str, Any] or None
        Its contents, or None when there is no file.

    Raises
    ------
    ValueError
        When the file isn't a JSON object.
    OSError
        When the file exists but can't be read.
    """

    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(stored, dict):
        raise ValueError(f"{path}: not a JSON object")
    return stored


def write_json(path: Path, contents: dict[str, Any]) -> None:
    """Write a JSON object to a file, making its folder.

    Parameters
    ----------
    path : Path
        The file.
    contents : dict[str, Any]
        The object.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2) + "\n")


def read_selection(folder: Path) -> Selection | None:
    """Read a label's `selection.json`.

    Parameters
    ----------
    folder : Path
        The label's folder.

    Returns
    -------
    Selection or None
        The stored selection, or None when the label has none yet.

    Raises
    ------
    ValueError
        When the file isn't JSON or its fields don't match `Selection`'s.
    OSError
        When the file exists but can't be read.
    """

    path = folder / SELECTION_FILE
    stored = _read_json(path)
    if stored is None:
        return None
    try:
        return Selection(**stored)
    except TypeError as error:
        raise ValueError(f"{path}: {error}") from error


def write_selection(folder: Path, selection: Selection) -> None:
    """Write a label's `selection.json`.

    Parameters
    ----------
    folder : Path
        The label's folder.
    selection : Selection
        The selection to keep.
    """

    write_json(folder / SELECTION_FILE, asdict(selection))


def read_reference(folder: Path) -> ReferenceCheck | None:
    """Read a label's `reference.json`.

    Parameters
    ----------
    folder : Path
        The label's folder.

    Returns
    -------
    ReferenceCheck or None
        The stored check, or None when the label has none yet.

    Raises
    ------
    ValueError
        When the file isn't JSON or its fields don't match `ReferenceCheck`'s.
    OSError
        When the file exists but can't be read.
    """

    path = folder / REFERENCE_FILE
    stored = _read_json(path)
    if stored is None:
        return None
    stored_windows = stored.get(REFERENCE_WINDOWS_FIELD)
    if not isinstance(stored_windows, list):
        raise ValueError(f"{path}: `{REFERENCE_WINDOWS_FIELD}` isn't a list")
    try:
        windows = [ReferenceWindow(**window) for window in stored_windows]
        return ReferenceCheck(**{**stored, REFERENCE_WINDOWS_FIELD: windows})
    except TypeError as error:
        raise ValueError(f"{path}: {error}") from error


def read_reference_hashes(folder: Path) -> dict[str, str] | None:
    """Read the host build hashes a label's reference check was made with.

    Parameters
    ----------
    folder : Path
        The label's folder.

    Returns
    -------
    dict[str, str] or None
        The host `build.json`'s `sha256` map stored in `reference.json`, or None when there is no reference check.

    Raises
    ------
    ValueError
        When the file isn't JSON or its fields don't match `ReferenceCheck`'s.
    OSError
        When the file exists but can't be read.
    """

    check = read_reference(folder)
    return None if check is None else check.sha256


def read_reference_windows(folder: Path) -> list[ReferenceWindow] | None:
    """Read the outcome of each reference window a label's check was made on.

    Parameters
    ----------
    folder : Path
        The label's folder.

    Returns
    -------
    list[ReferenceWindow] or None
        The records of `reference.json`, or None when there is no reference check.

    Raises
    ------
    ValueError
        When the file isn't JSON or its fields don't match `ReferenceCheck`'s.
    OSError
        When the file exists but can't be read.
    """

    check = read_reference(folder)
    return None if check is None else check.windows


def read_profile_hash(tables: Path) -> str | None:
    """Read the hash of the rv64 `tspulse` a label's profile tables were made with.

    Parameters
    ----------
    tables : Path
        The label's tracked results folder.

    Returns
    -------
    str or None
        The profile's `build.json` `sha256` of `tspulse`, or None when the profile has no tables yet. `profile` writes that record after its six tables, so its presence means they are complete.

    Raises
    ------
    ValueError
        When the file isn't a JSON object.
    OSError
        When the file exists but can't be read.
    """

    stored = _read_json(tables / WORK_SUBFOLDER / BUILD_FILE)
    if stored is None:
        return None
    hashes = stored.get(BUILD_SHA256_KEY)
    return hashes.get(CLEAN_BINARY) if isinstance(hashes, dict) else None


def write_collect_record(folder: Path, problems: int, sha256: dict[str, str]) -> None:
    """Keep how a label's last collect went, since its tables are written even when it finds problems.

    Parameters
    ----------
    folder : Path
        The label's folder.
    problems : int
        How many problems the collect found.
    sha256 : dict[str, str]
        The `label.json` hashes the collect read the units with.
    """

    write_json(folder / COLLECT_FILE, asdict(CollectRecord(problems, sha256)))


def read_collect(folder: Path) -> CollectRecord | None:
    """Read a label's `collect.json`.

    Parameters
    ----------
    folder : Path
        The label's folder.

    Returns
    -------
    CollectRecord or None
        The stored record, or None when the label has none yet.

    Raises
    ------
    ValueError
        When the file isn't JSON or its fields don't match `CollectRecord`'s.
    OSError
        When the file exists but can't be read.
    """

    path = folder / COLLECT_FILE
    stored = _read_json(path)
    if stored is None:
        return None
    try:
        return CollectRecord(**stored)
    except TypeError as error:
        raise ValueError(f"{path}: {error}") from error


def units_complete(
    folder: Path, identity: LabelIdentity, series: list[Series], window_cap: int | None
) -> bool:
    """Check that every accuracy and instruction unit of the chosen series has a result made by the label's binaries.

    Parameters
    ----------
    folder : Path
        The label's folder.
    identity : LabelIdentity
        The label's identity, whose hashes the results must carry.
    series : list[Series]
        The chosen series.
    window_cap : int or None
        The most windows kept of each scope range.

    Returns
    -------
    bool
        True when `find_results` finds every unit.
    """

    # The same units `run` makes with these options, so a unit counts here exactly when `run` would skip it
    accuracy = accuracy_units(series, SCOPES, window_cap, HOST_CHUNK_CHANNEL_WINDOWS)
    instructions = instruction_units(series, COUNT_WINDOWS)
    found_accuracy, _ = find_results(folder, accuracy, identity.sha256)
    found_instructions, _ = find_results(folder, instructions, identity.sha256)
    return len(found_accuracy) == len(accuracy) and len(found_instructions) == len(
        instructions
    )


def run_missing(
    folder: Path, identity: LabelIdentity, selection: Selection, working: list[Series]
) -> bool:
    """Check whether `run` still has units to make for a label.

    Parameters
    ----------
    folder : Path
        The label's folder.
    identity : LabelIdentity
        The label's identity, whose hashes the results must carry.
    selection : Selection
        The windows the label was asked to run.
    working : list[Series]
        Every working series, which the selection picks from.

    Returns
    -------
    bool
        True when the selection names a series that isn't working, or some unit of it has no result.
    """

    chosen = [one for one in working if one.stem in selection.series]
    return len(chosen) != len(selection.series) or not units_complete(
        folder, identity, chosen, selection.window_cap
    )


def collect_missing(
    tables: Path, identity: LabelIdentity, collect: CollectRecord | None
) -> bool:
    """Check whether `collect` still has to make the label's tables.

    Parameters
    ----------
    tables : Path
        The label's tracked results folder.
    identity : LabelIdentity
        The label's identity, whose hashes the collect must have read the units with.
    collect : CollectRecord or None
        The label's `collect.json`, None when it has none.

    Returns
    -------
    bool
        True when a table is missing or the last collect didn't read the label's units. A collect that found problems isn't missing: its tables stand, and the problems are its record's `problems`.
    """

    have_tables = all(
        (tables / name).is_file() for name in (ACCURACY_TABLE, INSTRUCTIONS_TABLE)
    )
    return not have_tables or collect is None or collect.sha256 != identity.sha256


def binaries_problem(folder: Path, host_sha256: str, rv64_sha256: str) -> str | None:
    """Compare a label's fixed binaries with freshly built ones.

    Parameters
    ----------
    folder : Path
        The label's folder.
    host_sha256 : str
        The hash of the host build's `tspulse`.
    rv64_sha256 : str
        The hash of the rv64 build's `tspulse`.

    Returns
    -------
    str or None
        Why the label's results can't be added to, or None when it has no identity yet or its binaries match.
    """

    try:
        identity = read_identity(folder)
    except (OSError, ValueError) as error:
        return f"cannot read the label's identity: {error}"
    if identity is None:
        return None
    differing = [
        role
        for role, built in ((HOST_ROLE, host_sha256), (RV64_ROLE, rv64_sha256))
        if identity.sha256.get(role) != built
    ]
    if differing:
        return f"the {' and '.join(differing)} binaries differ from the ones label.json fixed; pass --redo to start the label again"
    return None


def missing_stages(
    top: Path, data: Path, label: str, working: list[Series]
) -> list[str]:
    """Name what a variant's label still lacks of everything the command makes, except `summary.csv`.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The variant's label.
    working : list[Series]
        Every working series, which the label's selection picks from.

    Returns
    -------
    list[str]
        One short text per missing part: the reference check, the run, the collect, which says how many problems the last one found when it did read the label's units, and the profile. Empty when none is missing. Anything unreadable gives `MISSING_UNREADABLE` alone.
    """

    folder = label_folder(data, label)
    tables = tables_folder(top, label)
    try:
        identity = read_identity(folder)
        selection = read_selection(folder)
        reference = read_reference_hashes(folder)
        profile = read_profile_hash(tables)
        collect = read_collect(folder)
    except (OSError, ValueError):
        return [MISSING_UNREADABLE]
    missing: list[str] = []

    # Step 1: the reference check was made with the binaries the label fixed
    if reference is None or (
        identity is not None
        and identity.sha256.get(HOST_ROLE) != reference.get(CLEAN_BINARY)
    ):
        missing.append(VariantStage.REFERENCE)

    # Step 2: without an identity or a selection nothing was run, so nothing was collected, and the profile has no binary to match
    if identity is None or selection is None:
        missing += [VariantStage.RUN, VariantStage.COLLECT]
        if profile is None:
            missing.append(VariantStage.PROFILE)
        return missing

    # Step 3: every unit of the selection has a result
    if run_missing(folder, identity, selection, working):
        missing.append(VariantStage.RUN)

    # Step 4: both collected tables exist, from a collect that found no problem
    if collect_missing(tables, identity, collect):
        missing.append(VariantStage.COLLECT)
    elif collect is not None and collect.problems:
        missing.append(f"collect found {collect.problems} problems")

    # Step 5: the profile was made with the rv64 binary the label fixed
    if identity.sha256.get(RV64_ROLE) != profile:
        missing.append(VariantStage.PROFILE)
    return missing


def is_done(top: Path, data: Path, label: str, working: list[Series]) -> bool:
    """Check on disk whether a variant's label holds every result the command makes.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The variant's label.
    working : list[Series]
        Every working series, which the label's selection picks from.

    Returns
    -------
    bool
        True when nothing is missing, as `missing_stages` says, and the summary exists. Anything unreadable counts as not done.
    """

    summary = tables_folder(top, label) / SUMMARY_FILE
    return not missing_stages(top, data, label, working) and summary.is_file()


def delete_for_redo(top: Path, data: Path, label: str) -> list[Path]:
    """Delete what the command made for a label, beside the units `run --redo` deletes.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The variant's label.

    Returns
    -------
    list[Path]
        The files and folders that existed and were deleted.
    """

    folder = label_folder(data, label)
    deleted = []
    for path in (
        folder / REFERENCE_FILE,
        folder / SELECTION_FILE,
        folder / COLLECT_FILE,
    ):
        if path.is_file():
            path.unlink()
            deleted.append(path)
    for path in (folder / WORK_SUBFOLDER, tables_folder(top, label)):
        if path.is_dir():
            shutil.rmtree(path)
            deleted.append(path)
    return deleted
