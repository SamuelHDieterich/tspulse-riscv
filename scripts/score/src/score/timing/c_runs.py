"""Timing the C program: its row files, its processes with their CPU time and peak memory, and the checks on what they wrote.

Each process is started through GNU time with `subprocess.Popen` and reaped with `os.wait4`: `wait4` gives the CPU time, GNU time the peak memory.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Internal
from score import rows
from score.capture import load
from score.timing.results import Measurement


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The program's option for a window range `START:STOP`
WINDOWS_OPTION = "--windows"

# The first column of an errors CSV, which the head columns follow
WINDOW_COLUMN = "window"

# The one-core pair's short run: the first window alone, whose time is the program's start-up plus one window
FIRST_WINDOWS = 1

# Wait for this one child, with no options, so the call blocks until it ends
WAIT_BLOCKING = 0

# GNU time's options: print only the peak resident memory in KiB (`%M`) to a file, and end its own options before the program
TIME_FORMAT_OPTION = "-f"
PEAK_KIB_FORMAT = "%M"
TIME_OUTPUT_OPTION = "-o"
END_OF_OPTIONS = "--"

# The line GNU time writes to its output file when a signal ended the program, before the `%M` line
SIGNAL_LINE = re.compile(r"Command terminated by signal (\d+)")

# A process group of its own, numbered after the launcher, so one kill reaches GNU time and the program it started
OWN_PROCESS_GROUP = 0

# The files each run leaves in the unit's work folder
ERRORS_SUFFIX = ".errors.csv"
STDERR_SUFFIX = ".stderr"
PEAK_SUFFIX = ".time"
ALL_WINDOWS_STEM = "all"
FIRST_WINDOW_STEM = "first"
PART_STEM = "part{index:02d}"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class PreparedSeries:
    """A series' row file, and how many windows the units run of it.

    Attributes
    ----------
    rows_path : Path
        The `.rows` file the program reads.
    channels : int
        How many channels the series has.
    windows : int
        The series' window count, or its cap when smaller.
    """

    rows_path: Path
    channels: int
    windows: int


@dataclass(frozen=True)
class ProgramRun:
    """One run of the C program, as the operating system accounted for it.

    Attributes
    ----------
    argv : list[str]
        The exact command, GNU time included.
    windows : str
        The `START:STOP` range it ran.
    exit_code : int
        GNU time's exit code: the program's own, or 128 plus the signal number when a signal ended the program. Negative only when a signal ended GNU time itself.
    stderr : str
        The file its stderr went to.
    errors : str
        The errors CSV it wrote.
    cpu_seconds : float
        User plus system CPU time of GNU time and the program it waited for, from `wait4`.
    max_rss_kib : int or None
        Peak resident memory of the C process in KiB, from GNU time's `%M`, or None when GNU time's file is missing or holds no number.
    peak_file : str
        The file GNU time wrote its report to.
    seconds : float or None
        Its wall time, or None when it ran beside others, where only the group's wall time is measured.
    """

    argv: list[str]
    windows: str
    exit_code: int
    stderr: str
    errors: str
    cpu_seconds: float
    max_rss_kib: int | None
    peak_file: str
    seconds: float | None


@dataclass
class _Started:
    """A program started and not yet reaped.

    Attributes
    ----------
    process : subprocess.Popen[bytes]
        GNU time, which starts the program and waits for it.
    argv : list[str]
        Its command.
    windows : range
        The windows it was asked for.
    stderr : Path
        Its stderr file.
    errors : Path
        Its errors CSV.
    peak : Path
        The file GNU time writes the peak memory to.
    """

    process: subprocess.Popen[bytes]
    argv: list[str]
    windows: range
    stderr: Path
    errors: Path
    peak: Path


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def prepare_series(
    data: Path, work: Path, name: str, csv: Path, cap: int | None
) -> PreparedSeries:
    """Write a series' row file under the work folder, as `prepare-rows` does, and count its windows.

    Parameters
    ----------
    data : Path
        The folder with the data.
    work : Path
        The folder for the row files and the programs' output.
    name : str
        The series' short name, which names the row file.
    csv : Path
        Its CSV, relative to `data`.
    cap : int or None
        The most windows run, or None for every window.

    Returns
    -------
    PreparedSeries
        The row file and the windows the units run.
    """

    series = load(data / csv)
    rows_path = work / f"{name}.rows"
    rows.write(rows_path, series.raw, series.mean, series.std)
    windows = series.windows if cap is None else min(cap, series.windows)
    return PreparedSeries(rows_path, series.raw.shape[1], windows)


def _check_windows(path: Path, windows: range) -> str | None:
    """Check that an errors CSV the program wrote holds exactly the asked windows, in order.

    Parameters
    ----------
    path : Path
        The errors CSV, whose first column is the window.
    windows : range
        The windows asked for.

    Returns
    -------
    str or None
        What is wrong with it, or None when it is right.
    """

    # Step 1: the header, whose first column names the window
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


def _start(
    launcher: Path,
    binary: Path,
    prepared: PreparedSeries,
    folder: Path,
    stem: str,
    windows: range,
    pin_cpu: int | None,
) -> _Started:
    """Start the program through GNU time on a window range, its stderr in a file.

    Parameters
    ----------
    launcher : Path
        GNU time.
    binary : Path
        The `tspulse` to run.
    prepared : PreparedSeries
        The series' row file.
    folder : Path
        The unit's work folder.
    stem : str
        Names the run's errors and stderr files.
    windows : range
        The windows to run.
    pin_cpu : int or None
        The CPU to pin it to, or None to leave it free.

    Returns
    -------
    _Started
        The child and what it was asked for.

    Raises
    ------
    OSError
        When the program can't be started.
    """

    errors = folder / f"{stem}{ERRORS_SUFFIX}"
    stderr_path = folder / f"{stem}{STDERR_SUFFIX}"
    peak = folder / f"{stem}{PEAK_SUFFIX}"
    argv = [
        str(launcher),
        TIME_FORMAT_OPTION,
        PEAK_KIB_FORMAT,
        TIME_OUTPUT_OPTION,
        str(peak),
        END_OF_OPTIONS,
        str(binary),
        str(prepared.rows_path),
        str(errors),
        WINDOWS_OPTION,
        f"{windows.start}:{windows.stop}",
    ]

    # Pinned by starting from a pinned thread, so GNU time and the program it forks are both born with the mask. `preexec_fn` is unsafe in a parent with threads, and pinning GNU time's pid after the start could miss the program it has already forked
    previous = os.sched_getaffinity(0)
    if pin_cpu is not None:
        os.sched_setaffinity(0, {pin_cpu})
    try:
        with stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                process_group=OWN_PROCESS_GROUP,
            )
    finally:
        os.sched_setaffinity(0, previous)
    return _Started(process, argv, windows, stderr_path, errors, peak)


def _time_report(path: Path) -> list[str]:
    """Read GNU time's report.

    Parameters
    ----------
    path : Path
        GNU time's output file.

    Returns
    -------
    list[str]
        Its lines, empty when the file is missing or can't be read, so the exit code and stderr still tell what went wrong.
    """

    try:
        return path.read_text().splitlines()
    except OSError:
        return []


def _read_peak(path: Path) -> int | None:
    """Read the peak memory GNU time wrote.

    Parameters
    ----------
    path : Path
        GNU time's output file, whose last line is `%M`. Lines before it say when the program exited non-zero or was ended by a signal.

    Returns
    -------
    int or None
        The program's peak resident memory in KiB, or None when the file is missing or its last line holds no number.
    """

    lines = _time_report(path)
    if not lines or not lines[-1].strip().isdigit():
        return None
    return int(lines[-1])


def _signal_name(path: Path) -> str | None:
    """Name the signal that ended the program, from GNU time's report.

    Parameters
    ----------
    path : Path
        GNU time's output file.

    Returns
    -------
    str or None
        Such as `SIGSEGV`, or None when no signal ended the program.
    """

    for line in _time_report(path):
        match = SIGNAL_LINE.search(line)
        if match is not None:
            number = int(match.group(1))
            try:
                return signal.Signals(number).name
            except ValueError:
                return f"signal {number}"
    return None


def _reap(started: _Started) -> ProgramRun:
    """Wait for a started program to end and read its resource usage.

    Parameters
    ----------
    started : _Started
        The child.

    Returns
    -------
    ProgramRun
        Its exit code, CPU time and peak memory, with no wall time, which only the caller knows.
    """

    _, status, usage = os.wait4(started.process.pid, WAIT_BLOCKING)
    exit_code = os.waitstatus_to_exitcode(status)

    # `wait4` reaped the child behind Popen's back; telling Popen keeps it from waiting on a pid that is gone
    started.process.returncode = exit_code

    # The peak memory comes from GNU time, not from `ru_maxrss`: Linux carries the forking parent's resident size into a forked-and-exec'd child's `ru_maxrss`, so a 6.5 MB run started from this Python process read about 510 MB
    return ProgramRun(
        argv=started.argv,
        windows=f"{started.windows.start}:{started.windows.stop}",
        exit_code=exit_code,
        stderr=str(started.stderr),
        errors=str(started.errors),
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        max_rss_kib=_read_peak(started.peak),
        peak_file=str(started.peak),
        seconds=None,
    )


def _kill(started: list[_Started]) -> None:
    """Kill and reap every started program that has not been reaped, GNU time and the program it started alike.

    Parameters
    ----------
    started : list[_Started]
        Every program the unit started.
    """

    for one in started:
        if one.process.returncode is None:
            # The whole group, since killing GNU time alone would leave the program running
            os.killpg(one.process.pid, signal.SIGKILL)
            one.process.wait()


def _problem(run: ProgramRun, windows: range) -> str | None:
    """Say what went wrong with a finished run: a non-zero exit or a signal, no peak memory from GNU time, or the wrong windows in its errors CSV.

    Parameters
    ----------
    run : ProgramRun
        The finished run.
    windows : range
        The windows it was asked for.

    Returns
    -------
    str or None
        What is wrong, or None when the run is good.
    """

    # Step 1: a failed run, named by its signal when one ended it, since GNU time then exits 128 plus the signal number
    if run.exit_code:
        message = "; ".join(
            Path(run.stderr).read_text(errors="replace").strip().splitlines()
        )
        name = _signal_name(Path(run.peak_file))
        reason = f"killed by {name}" if name else f"exit {run.exit_code}"
        return f"{reason} on {run.windows}: {message}"

    # Step 2: a clean run must have its peak memory and exactly its windows
    if run.max_rss_kib is None:
        return f"exit 0 on {run.windows}, but GNU time's {Path(run.peak_file).name} is missing or empty, so the run has no peak memory"
    return _check_windows(Path(run.errors), windows)


def _peaks(runs: list[ProgramRun]) -> list[int]:
    """Collect the runs' peak memories, once `_problem` has passed every run.

    Parameters
    ----------
    runs : list[ProgramRun]
        Runs that `_problem` found good, so each has its peak.

    Returns
    -------
    list[int]
        Each run's peak resident memory in KiB, in order.
    """

    return [run.max_rss_kib for run in runs if run.max_rss_kib is not None]


def _fresh_folder(folder: Path) -> None:
    """Empty a unit's work folder, so no file from an earlier run sits beside this run's.

    Parameters
    ----------
    folder : Path
        The unit's work folder.
    """

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)


def run_one_core(
    launcher: Path, binary: Path, prepared: PreparedSeries, folder: Path, pin_cpu: int
) -> Measurement | str:
    """Time the program on one pinned core, with start-up taken out.

    The wall time per window comes from the difference T_N − T_1 of the `0:N` and `0:1` runs, while `cpu_seconds` is the whole `0:N` run's CPU time, start-up included.

    Parameters
    ----------
    launcher : Path
        GNU time, which measures each run's peak memory.
    binary : Path
        The `tspulse` to run.
    prepared : PreparedSeries
        The series' row file and window count N.
    folder : Path
        The unit's work folder.
    pin_cpu : int
        The CPU both runs are pinned to.

    Returns
    -------
    Measurement or str
        The measurement, or what went wrong.
    """

    _fresh_folder(folder)
    count = prepared.windows
    ranges = {
        ALL_WINDOWS_STEM: range(0, count),
        FIRST_WINDOW_STEM: range(0, FIRST_WINDOWS),
    }

    # Step 1: the long run, then the one-window run, one at a time
    started: list[_Started] = []
    runs: dict[str, ProgramRun] = {}
    wall: dict[str, float] = {}
    try:
        for stem, windows in ranges.items():
            began = time.perf_counter()
            started.append(
                _start(launcher, binary, prepared, folder, stem, windows, pin_cpu)
            )
            run = _reap(started[-1])
            wall[stem] = time.perf_counter() - began
            runs[stem] = replace(run, seconds=wall[stem])
    except OSError as error:
        _kill(started)
        return f"cannot run {binary}: {error}"
    except BaseException:
        _kill(started)
        raise

    # Step 2: both must have exited cleanly and written exactly their windows
    for stem, windows in ranges.items():
        problem = _problem(runs[stem], windows)
        if problem is not None:
            return problem

    # Step 3: start-up costs the same in both runs, so it cancels in the difference
    # seconds per window = (T_N − T_1) / (N − 1), a difference quotient: https://en.wikipedia.org/wiki/Difference_quotient
    timed = wall[ALL_WINDOWS_STEM] - wall[FIRST_WINDOW_STEM]
    long, short = runs[ALL_WINDOWS_STEM], runs[FIRST_WINDOW_STEM]

    # The CPU time is the long run's whole, start-up included, not a difference like the wall time; both runs' raw values stay in the details
    return Measurement(
        windows=count,
        seconds_per_window=timed / (count - FIRST_WINDOWS),
        timed_seconds=timed,
        cpu_seconds=long.cpu_seconds,
        max_rss_kib=max(_peaks([long, short])),
        details={stem: asdict(run) for stem, run in runs.items()},
    )


def _split(count: int, parts: int) -> list[range]:
    """Split windows `0:count` into contiguous ranges whose sizes differ by at most one.

    Parameters
    ----------
    count : int
        How many windows.
    parts : int
        How many ranges.

    Returns
    -------
    list[range]
        Range `i` is `count·i // parts` to `count·(i + 1) // parts`.
    """

    return [range(count * i // parts, count * (i + 1) // parts) for i in range(parts)]


def run_whole_pc(
    launcher: Path, binary: Path, prepared: PreparedSeries, folder: Path, parallel: int
) -> Measurement | str:
    """Time P unpinned copies of the program, each over its own share of the windows.

    Parameters
    ----------
    launcher : Path
        GNU time, which measures each run's peak memory.
    binary : Path
        The `tspulse` to run.
    prepared : PreparedSeries
        The series' row file and window count N.
    folder : Path
        The unit's work folder.
    parallel : int
        How many processes, P.

    Returns
    -------
    Measurement or str
        The measurement, or what went wrong.
    """

    _fresh_folder(folder)
    ranges = _split(prepared.windows, parallel)

    # Step 1: start every process, then reap them in turn. The clock stops once the last has ended, start-up included
    started: list[_Started] = []
    try:
        began = time.perf_counter()
        for index, windows in enumerate(ranges):
            stem = PART_STEM.format(index=index)
            started.append(
                _start(launcher, binary, prepared, folder, stem, windows, None)
            )
        runs = [_reap(one) for one in started]
        seconds = time.perf_counter() - began
    except OSError as error:
        _kill(started)
        return f"cannot run {binary}: {error}"
    except BaseException:
        _kill(started)
        raise

    # Step 2: every process must have exited cleanly and written exactly its windows
    for run, windows in zip(runs, ranges, strict=True):
        problem = _problem(run, windows)
        if problem is not None:
            return problem

    # Step 3: CPU time adds up over processes; memory is kept both as the largest process and as the sum, an upper bound on what they held at once
    windows = sum(len(one) for one in ranges)
    peaks = _peaks(runs)
    largest = max(peaks)
    return Measurement(
        windows=windows,
        seconds_per_window=seconds / windows,
        timed_seconds=seconds,
        cpu_seconds=sum(run.cpu_seconds for run in runs),
        max_rss_kib=largest,
        details={
            "seconds": seconds,
            "max_rss_kib": largest,
            "max_rss_sum_kib": sum(peaks),
            "processes": [asdict(run) for run in runs],
        },
    )
