"""The Spike runs `benchmark profile` makes for each window, and how a rerun skips the ones already done.

A window M is measured by differences, so it takes three runs:
the clean build over `M-1:M+1` and over `M-1:M`, both with the histogram and the cache counters, and the profile build over `M-1:M+1`.
Each run has its own folder in the work folder, and its `done.json` marks it finished.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import shlex
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# Internal
from score.benchmark.execute import (
    COUNTS_HEADER,
    COUNTS_OPTION,
    WINDOWS_OPTION,
    _check_windows,
)
from score.benchmark.progress import SEPARATOR, Log, StageProgress, duration, eta
from score.processes import Processes
from score.profile.errors import ProfileError
from score.profile.modules import read_profile
from score.profile.traffic import DATA_CACHE, INSTRUCTION_CACHE
from score.spike_check import _program_message
from score.spike_results import _replace


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# What a unit keeps in its folder
STDOUT_FILE = "stdout"
STDERR_FILE = "stderr"
ERRORS_FILE = "errors.csv"
COUNTS_FILE = "counts.csv"
PROFILE_FILE = "profile.csv"
COMMAND_FILE = "command"
DONE_FILE = "done.json"

# Spike's option for the PC histogram
HISTOGRAM_OPTION = "-g"

# The profile build's option for its CSV
PROFILE_OPTION = "--profile"

# Spike's options for the runs of the clean build; the profile build, which counts in the program, runs without them
CLEAN_SPIKE_OPTIONS = [HISTOGRAM_OPTION, INSTRUCTION_CACHE, DATA_CACHE]

# The prefix of the ISA argument, which ends Spike's own options in a command line
ISA_ARGUMENT_PREFIX = "--isa="

# Wall seconds assumed per run of one channel before any has finished, from a timed 428:430 sample
ASSUMED_SHORT_SECONDS = 90.0
ASSUMED_LONG_SECONDS = 180.0
ASSUMED_PROFILE_SECONDS = 65.0


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Kind(StrEnum):
    """Which run of a window a unit is.

    Attributes
    ----------
    SHORT : str
        The clean build over `M-1:M`.
    LONG : str
        The clean build over `M-1:M+1`.
    PROFILE : str
        The profile build over `M-1:M+1`.
    """

    SHORT = "short"
    LONG = "long"
    PROFILE = "profile"


@dataclass(frozen=True)
class ProfileUnit:
    """One Spike run.

    Attributes
    ----------
    window : int
        The measured window M, at least 1.
    kind : Kind
        Which of its three runs this is.
    """

    window: int
    kind: Kind

    @property
    def start(self) -> int:
        """The first window the run covers, the one before the measured one."""

        return self.window - 1

    @property
    def stop(self) -> int:
        """Just past the last window the run covers."""

        return self.window if self.kind == Kind.SHORT else self.window + 1

    @property
    def name(self) -> str:
        """The unit's name in a log line and its folder."""

        return f"w{self.window}-{self.kind}"

    @property
    def assumed_seconds(self) -> float:
        """Wall seconds of the run on one channel, for the first ETA."""

        return {
            Kind.SHORT: ASSUMED_SHORT_SECONDS,
            Kind.LONG: ASSUMED_LONG_SECONDS,
            Kind.PROFILE: ASSUMED_PROFILE_SECONDS,
        }[self.kind]

    @property
    def spike_options(self) -> list[str]:
        """Spike's own options for the run, before the ISA."""

        return [] if self.kind == Kind.PROFILE else CLEAN_SPIKE_OPTIONS

    @property
    def windows(self) -> int:
        """How many windows the run covers."""

        return self.stop - self.start


@dataclass(frozen=True)
class ProfileSettings:
    """What every unit of one call shares.

    Attributes
    ----------
    clean : Path
        The clean rv64 `tspulse`.
    profile : Path
        The profile build's `tspulse-profile`.
    pk : Path
        The proxy kernel Spike runs the program under.
    isa : str
        Spike's ISA string.
    rows : Path
        The series' row file.
    work : Path
        The folder with one folder per unit.
    channels : int
        The series' channels, which scale a unit's cost.
    sha256 : dict[str, str]
        The hashes of the binaries and the row file, saved with each unit.
    processes : Processes
        The programs started so far.
    """

    clean: Path
    profile: Path
    pk: Path
    isa: str
    rows: Path
    work: Path
    channels: int
    sha256: dict[str, str]
    processes: Processes = field(default_factory=Processes)

    def folder(self, unit: ProfileUnit) -> Path:
        """The folder a unit keeps its files in.

        Parameters
        ----------
        unit : ProfileUnit
            The unit.

        Returns
        -------
        Path
            `<work>/w<window>-<kind>`.
        """

        return self.work / unit.name


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def units_of(windows: list[int]) -> list[ProfileUnit]:
    """List the runs of the measured windows, window by window.

    Parameters
    ----------
    windows : list[int]
        The measured windows.

    Returns
    -------
    list[ProfileUnit]
        Each window's short, long and profile run.
    """

    return [ProfileUnit(window, kind) for window in windows for kind in Kind]


def _command(unit: ProfileUnit, settings: ProfileSettings) -> list[str]:
    """Write a unit's Spike command line.

    Parameters
    ----------
    unit : ProfileUnit
        The unit.
    settings : ProfileSettings
        What every unit shares.

    Returns
    -------
    list[str]
        Spike, then the program with its row file, errors file and window range.
    """

    folder = settings.folder(unit)
    binary = settings.profile if unit.kind == Kind.PROFILE else settings.clean
    extra = (
        [PROFILE_OPTION, str(folder / PROFILE_FILE)]
        if unit.kind == Kind.PROFILE
        else [COUNTS_OPTION, str(folder / COUNTS_FILE)]
    )
    return [
        "spike",
        *unit.spike_options,
        f"--isa={settings.isa}",
        str(settings.pk),
        str(binary),
        str(settings.rows),
        str(folder / ERRORS_FILE),
        WINDOWS_OPTION,
        f"{unit.start}:{unit.stop}",
        *extra,
    ]


def read_counts(path: Path) -> dict[int, int]:
    """Read a `--counts` CSV.

    Parameters
    ----------
    path : Path
        The counts CSV.

    Returns
    -------
    dict[int, int]
        Instructions of each window it holds.

    Raises
    ------
    ProfileError
        If the header or a line is wrong.
    """

    lines = path.read_text().splitlines()
    if lines[:1] != [COUNTS_HEADER]:
        raise ProfileError(f"{path}: header is not {COUNTS_HEADER}")
    counts: dict[int, int] = {}
    for line in lines[1:]:
        window, _, count = line.partition(",")
        if not window.isdigit() or not count.isdigit():
            raise ProfileError(f"{path}: not a counts line: {line!r}")
        counts[int(window)] = int(count)
    return counts


def _check_outputs(unit: ProfileUnit, folder: Path) -> str | None:
    """Check that a finished run wrote the files the tables read.

    Parameters
    ----------
    unit : ProfileUnit
        The unit.
    folder : Path
        Its folder.

    Returns
    -------
    str or None
        What is wrong, or None when the run is whole.
    """

    # Step 1: the errors must hold exactly the windows asked for
    problem = _check_windows(folder / ERRORS_FILE, range(unit.start, unit.stop))
    if problem is not None:
        return problem

    # Step 2: the profile build's CSV, or the clean build's counts, must cover them too
    if unit.kind == Kind.PROFILE:
        try:
            read_profile(folder / PROFILE_FILE, unit.window)
        except (OSError, ProfileError) as error:
            return str(error)
        return None
    try:
        counts = read_counts(folder / COUNTS_FILE)
    except (OSError, ProfileError) as error:
        return str(error)
    if sorted(counts) != list(range(unit.start, unit.stop)):
        return f"{COUNTS_FILE} holds windows {sorted(counts)}"
    return None


def _identity(settings: ProfileSettings) -> dict[str, object]:
    """What a unit's `done.json` must match for the unit to count as done.

    Parameters
    ----------
    settings : ProfileSettings
        What every unit shares.

    Returns
    -------
    dict[str, object]
        The hashes and the ISA.
    """

    return {"sha256": settings.sha256, "isa": settings.isa}


def is_done(unit: ProfileUnit, settings: ProfileSettings) -> bool:
    """Say whether a unit already ran with these binaries and this row file.

    Parameters
    ----------
    unit : ProfileUnit
        The unit.
    settings : ProfileSettings
        What every unit shares.

    Returns
    -------
    bool
        True when `done.json` is readable and matches; an unreadable one counts as not done, so the unit reruns.
    """

    try:
        stored = json.loads((settings.folder(unit) / DONE_FILE).read_text())
    except (OSError, ValueError):
        return False
    wanted = _identity(settings)
    if not all(stored.get(key) == value for key, value in wanted.items()):
        return False

    # The recorded command has Spike's options between `spike` and `--isa=`; they must be the ones this unit runs with now
    try:
        recorded = shlex.split(stored["command"])[1:]
    except (KeyError, ValueError, TypeError):
        return False
    options = []
    for token in recorded:
        if token.startswith(ISA_ARGUMENT_PREFIX):
            return options == unit.spike_options
        options.append(token)
    return False


def run_unit(unit: ProfileUnit, settings: ProfileSettings) -> float | str | None:
    """Run one unit and save its files and its `done.json`.

    Parameters
    ----------
    unit : ProfileUnit
        The unit.
    settings : ProfileSettings
        What every unit shares.

    Returns
    -------
    float, str or None
        Its wall seconds; a message when the run fails or writes something unexpected, which leaves no `done.json` so it reruns; or None when the call is stopping.
    """

    # Step 1: an empty folder, so no file of an earlier run sits beside this run's
    folder = settings.folder(unit)
    folder.mkdir(parents=True, exist_ok=True)
    for name in (
        DONE_FILE,
        STDOUT_FILE,
        STDERR_FILE,
        ERRORS_FILE,
        COUNTS_FILE,
        PROFILE_FILE,
        COMMAND_FILE,
    ):
        (folder / name).unlink(missing_ok=True)

    # Step 2: the exact command is kept, since it is what a reader reruns
    command = _command(unit, settings)
    (folder / COMMAND_FILE).write_text(shlex.join(command) + "\n")

    # Step 3: the run, with stdout (Spike's cache report) and stderr (the histogram) kept
    try:
        run = settings.processes.run(
            command, folder / STDERR_FILE, folder / STDOUT_FILE
        )
    except OSError as error:
        return f"cannot run spike: {error}"
    if run is None:
        return None
    code, seconds = run
    if code:
        return f"exit {code}: {_program_message(folder / STDERR_FILE)}"

    # Step 4: the files, then the marker
    problem = _check_outputs(unit, folder)
    if problem is not None:
        return problem
    done = {
        **_identity(settings),
        "command": shlex.join(command),
        "seconds": round(seconds, 1),
    }
    _replace(folder / DONE_FILE, json.dumps(done, indent=2) + "\n")
    return seconds


def run_units(
    units: list[ProfileUnit],
    skipped: int,
    jobs: int,
    settings: ProfileSettings,
    log: Log,
) -> int:
    """Run the pending units up to `jobs` at once, logging each as it finishes.

    A unit that fails is logged and counted, and the others go on.

    Parameters
    ----------
    units : list[ProfileUnit]
        The units without a `done.json`.
    skipped : int
        How many units already had one.
    jobs : int
        How many units run at once.
    settings : ProfileSettings
        What every unit shares.
    log : Log
        Where the lines go.

    Returns
    -------
    int
        How many units failed.

    Raises
    ------
    KeyboardInterrupt
        On Ctrl-C or SIGTERM, after every program started has been killed.
    """

    # Step 1: the start line, with an ETA from the assumed seconds
    cost = {unit: unit.windows * settings.channels for unit in units}
    progress = StageProgress("profile", len(units), sum(cost.values()))
    assumed = sum(unit.assumed_seconds * settings.channels for unit in units)
    parallel = max(1, min(jobs, len(units)))
    log.line(
        SEPARATOR.join(
            (
                f"profile: {len(units)} runs to do",
                f"{skipped} skipped",
                f"{jobs} jobs",
                f"ETA {eta(assumed / parallel)} (assumed rate)",
            )
        )
    )

    # Step 2: the runs. Only this thread logs
    failed = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures: dict[Future[float | str | None], ProfileUnit] = {}
        try:
            for unit in units:
                futures[pool.submit(run_unit, unit, settings)] = unit
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = f"{type(error).__name__}: {error}"
                if outcome is None:
                    continue
                if isinstance(outcome, str):
                    failed += 1
                    detail = f"FAILED {outcome}"
                else:
                    detail = f"{unit.start}:{unit.stop} in {duration(outcome)}"
                log.line(progress.finished(unit.name, cost[unit], detail))

        # On Ctrl-C, no program is left running. A run cut short has no `done.json`, so it reruns
        except BaseException:
            for future in futures:
                future.cancel()
            settings.processes.stop()
            raise
    log.line(
        f"profile runs done: {progress.done_units} in {duration(progress.elapsed())}, {failed} failed"
    )
    return failed
