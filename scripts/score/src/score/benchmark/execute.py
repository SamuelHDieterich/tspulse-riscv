"""Running one benchmark unit: its programs, the checks on what they wrote, and its saved result.

An accuracy unit runs a chunk of windows on the host or under Spike; an instruction unit runs one window under Spike with `--counts`, and on the host to compare.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import filecmp
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# Internal
from score.benchmark.results import (
    COUNTS_FILE,
    ERRORS_FILE,
    NOT_CHECKED,
    SPIKE_STDOUT_FILE,
    STDERR_FILE,
    AccuracyResult,
    HostCheck,
    Identical,
    InstructionResult,
    Platform,
    Provenance,
    write_result,
)
from score.benchmark.units import AccuracyUnit, InstructionUnit
from score.processes import Processes
from score.spike_check import (
    HOST_ERRORS,
    HOST_STDERR,
    SPIKE_ERRORS,
    SPIKE_STDERR,
    _program_message,
)
from score.spike_results import RESULT_FILE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The program's options for a window range `START:STOP` and for the counts CSV
WINDOWS_OPTION = "--windows"
COUNTS_OPTION = "--counts"

# The first column of the program's errors CSV, and the counts CSV's header
WINDOW_COLUMN = "window"
COUNTS_HEADER = "window,instructions"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Settings:
    """What every unit of one call shares.

    Attributes
    ----------
    label : str
        The benchmark label.
    label_folder : Path
        Where the label's results go, shared by every machine.
    host_binary : Path
        The host build's `tspulse`.
    rv64_binary : Path
        The rv64 build's `tspulse`.
    pk : Path
        The proxy kernel Spike runs the program under.
    work : Path
        The folder for the row files and the units' scratch files.
    isa : str
        Spike's ISA string.
    spike_options : list[str]
        Extra options Spike gets before pk on the counted windows.
    accuracy_on : Platform
        Where the accuracy units run.
    host_check : HostCheck
        Whether instruction units are checked against the host.
    provenance : Provenance
        The machine, the worktree's state and the binaries' hashes, saved with every result.
    processes : Processes
        The programs started so far.
    """

    label: str
    label_folder: Path
    host_binary: Path
    rv64_binary: Path
    pk: Path
    work: Path
    isa: str
    spike_options: list[str]
    accuracy_on: Platform
    host_check: HostCheck
    provenance: Provenance
    processes: Processes = field(default_factory=Processes)

    def rows_path(self, series: str) -> Path:
        """The row file a series' units read.

        Parameters
        ----------
        series : str
            The series' CSV name without `.csv`.

        Returns
        -------
        Path
            `<work>/<stem>.rows`.
        """

        return self.work / f"{series}.rows"

    def spike_prefix(self, options: list[str]) -> list[str]:
        """Spike's command line up to the program.

        Parameters
        ----------
        options : list[str]
            Extra Spike options, which go before pk.

        Returns
        -------
        list[str]
            `spike [options] --isa=ISA PK RV64_TSPULSE`.
        """

        return [
            "spike",
            *options,
            f"--isa={self.isa}",
            str(self.pk),
            str(self.rv64_binary),
        ]


# A unit's outcome: its result, a message when it failed, or None when the call is stopping
Outcome = AccuracyResult | InstructionResult | str | None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _check_windows(path: Path, windows: range) -> str | None:
    """Check that a CSV the program wrote holds exactly the asked windows, in order.

    Parameters
    ----------
    path : Path
        The errors or counts CSV, whose first column is the window.
    windows : range
        The windows asked for.

    Returns
    -------
    str or None
        What is wrong with it, or None when it is right.
    """

    # Step 1: the header, whose first column names the window. The other columns are read by `collect`
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        return f"cannot read {path.name}: {error}"
    if not lines or lines[0].split(",")[0] != WINDOW_COLUMN:
        return f"{path.name} has no `{WINDOW_COLUMN},…` header"

    # Step 2: one line per window, in order
    found = [line.split(",")[0] for line in lines[1:]]
    expected = [str(window) for window in windows]
    if found != expected:
        return f"{path.name} holds {len(found)} windows, not {windows.start}..{windows.stop - 1} in order"
    return None


def _run_program(
    settings: Settings,
    side: str,
    command: list[str],
    stderr_path: Path,
    stdout_path: Path | None = None,
) -> float | str | None:
    """Run one program and say how it went.

    Parameters
    ----------
    settings : Settings
        What every unit shares.
    side : str
        `host` or `spike`, which names it in a failure.
    command : list[str]
        The program and its arguments.
    stderr_path : Path
        Where its stderr goes.
    stdout_path : Path or None
        Where its stdout goes, or None to discard it.

    Returns
    -------
    float, str or None
        Its wall seconds; a message when it can't start or exits non-zero; or None when the call is stopping.
    """

    try:
        run = settings.processes.run(command, stderr_path, stdout_path)
    except OSError as error:
        return f"cannot run {command[0]}: {error}"
    if run is None:
        return None
    code, seconds = run
    if code:
        return f"{side} exit {code}: {_program_message(stderr_path)}"
    return seconds


def _fresh_folders(
    settings: Settings, folder: Path, names: tuple[str, ...]
) -> tuple[Path, Path]:
    """Make a unit's result and scratch folders, emptied of an earlier run's files.

    Parameters
    ----------
    settings : Settings
        What every unit shares.
    folder : Path
        The unit's folder, relative to the label's folder and the work folder.
    names : tuple[str, ...]
        The files the unit keeps, which an earlier run may have left.

    Returns
    -------
    tuple[Path, Path]
        The result folder, then the scratch folder.
    """

    result_folder = settings.label_folder / folder
    scratch = settings.work / settings.label / folder
    result_folder.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    # The result goes first, so a run cut short leaves the unit to rerun; the other files would otherwise sit beside this run's
    for name in (RESULT_FILE, *names):
        (result_folder / name).unlink(missing_ok=True)

    # A killed Spike can exit 0 without writing, and a cut-short run's scratch file must not then pass for this run's output
    for name in names:
        (scratch / name).unlink(missing_ok=True)
    return result_folder, scratch


def run_accuracy(unit: AccuracyUnit, settings: Settings) -> Outcome:
    """Run an accuracy unit's windows, check the errors CSV and save it with the unit's result.

    Parameters
    ----------
    unit : AccuracyUnit
        The chunk of windows.
    settings : Settings
        What every unit shares.

    Returns
    -------
    AccuracyResult, str or None
        The result, saved as `result.json`; a message when the program fails or writes the wrong windows, which saves no result so the unit reruns; or None when the call is stopping.
    """

    # Step 1: the folders, and the command on the host or under Spike. Spike's extra options are for the counted windows only
    result_folder, scratch = _fresh_folders(
        settings, unit.folder, (ERRORS_FILE, STDERR_FILE)
    )
    errors = scratch / ERRORS_FILE
    arguments = [
        str(settings.rows_path(unit.series)),
        str(errors),
        WINDOWS_OPTION,
        f"{unit.start}:{unit.stop}",
    ]
    if settings.accuracy_on == Platform.HOST:
        command = [str(settings.host_binary), *arguments]
    else:
        command = [*settings.spike_prefix([]), *arguments]

    # Step 2: the run, with its stderr kept beside the result
    run = _run_program(
        settings, settings.accuracy_on, command, result_folder / STDERR_FILE
    )
    if not isinstance(run, float):
        return run

    # Step 3: the errors must hold exactly the chunk's windows
    problem = _check_windows(errors, range(unit.start, unit.stop))
    if problem is not None:
        return problem
    shutil.move(errors, result_folder / ERRORS_FILE)

    # Step 4: the result, whose file marks the unit done
    result = AccuracyResult(
        label=settings.label,
        series=unit.series,
        kind=unit.kind,
        channels=unit.channels,
        start=unit.start,
        stop=unit.stop,
        ran_on=settings.accuracy_on,
        seconds=run,
        command=shlex.join(command),
        provenance=settings.provenance,
    )
    write_result(result_folder, result)
    return result


def _read_count(path: Path, window: int) -> int | str:
    """Read the one window's instructions from the counts CSV.

    Parameters
    ----------
    path : Path
        The counts CSV, `window,instructions` then one line.
    window : int
        The window asked for.

    Returns
    -------
    int or str
        The instructions, or what is wrong with the file.
    """

    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        return f"cannot read {path.name}: {error}"
    if lines[:1] != [COUNTS_HEADER] or len(lines) != 2:
        return f"{path.name} is not `{COUNTS_HEADER}` and one line"
    found, _, count = lines[1].partition(",")
    if found != str(window) or not count.isdigit():
        return f"{path.name} has `{lines[1]}`, not window {window} and a count"
    return int(count)


def run_instruction(unit: InstructionUnit, settings: Settings) -> Outcome:
    """Run one window under Spike with `--counts`, compare it with the host's run and save the unit's result.

    Parameters
    ----------
    unit : InstructionUnit
        The window.
    settings : Settings
        What every unit shares.

    Returns
    -------
    InstructionResult, str or None
        The result, saved as `result.json`; a message when a program fails or writes something unexpected, which saves no result so the unit reruns; or None when the call is stopping.
    """

    # Step 1: the folders and both command lines, with the same program arguments
    kept = (HOST_ERRORS, SPIKE_ERRORS, COUNTS_FILE, SPIKE_STDOUT_FILE, SPIKE_STDERR)
    result_folder, scratch = _fresh_folders(settings, unit.folder, kept)
    host_errors, spike_errors, counts = (
        scratch / HOST_ERRORS,
        scratch / SPIKE_ERRORS,
        scratch / COUNTS_FILE,
    )
    rows_path = str(settings.rows_path(unit.series))
    windows = [WINDOWS_OPTION, f"{unit.window}:{unit.window + 1}"]
    host_command = [str(settings.host_binary), rows_path, str(host_errors), *windows]
    spike_command = [
        *settings.spike_prefix(settings.spike_options),
        rows_path,
        str(spike_errors),
        *windows,
        COUNTS_OPTION,
        str(counts),
    ]

    # Step 2: the host first, since it takes a fraction of a second and a failure there saves the Spike run
    host_seconds = None
    if settings.host_check == HostCheck.IDENTICAL:
        run = _run_program(settings, Platform.HOST, host_command, scratch / HOST_STDERR)
        if not isinstance(run, float):
            return run
        host_seconds = run

    # Step 3: Spike, with its stdout and stderr kept for options such as `-g` that print there
    run = _run_program(
        settings,
        Platform.SPIKE,
        spike_command,
        result_folder / SPIKE_STDERR,
        result_folder / SPIKE_STDOUT_FILE,
    )
    if not isinstance(run, float):
        return run
    spike_seconds = run

    # Step 4: the files must hold the one window; the host's errors must match Spike's byte for byte
    window = range(unit.window, unit.window + 1)
    problem = _check_windows(spike_errors, window)
    if problem is not None:
        return problem
    count = _read_count(counts, unit.window)
    if isinstance(count, str):
        return count
    identical: Identical = NOT_CHECKED
    if host_seconds is not None:
        # The host can exit 0 without writing, and the comparison needs its CSV whole
        problem = _check_windows(host_errors, window)
        if problem is not None:
            return f"host: {problem}"
        identical = filecmp.cmp(host_errors, spike_errors, shallow=False)
        shutil.move(host_errors, result_folder / HOST_ERRORS)
    shutil.move(spike_errors, result_folder / SPIKE_ERRORS)
    shutil.move(counts, result_folder / COUNTS_FILE)

    # Step 5: the result, whose file marks the unit done, even when the two sides differ
    result = InstructionResult(
        label=settings.label,
        series=unit.series,
        kind=unit.kind,
        channels=unit.channels,
        window=unit.window,
        program_instructions=count,
        identical=identical,
        spike_seconds=spike_seconds,
        host_seconds=host_seconds,
        spike_command=shlex.join(spike_command),
        host_command=shlex.join(host_command) if host_seconds is not None else None,
        provenance=settings.provenance,
    )
    write_result(result_folder, result)
    return result
