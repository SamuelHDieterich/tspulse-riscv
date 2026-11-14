"""The `time-host` command: time any number of the C port's builds against IBM's PyTorch pipeline on this computer, under a label.

Every unit saves its JSON as soon as it finishes, so a second call skips what is done.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import inspect
import os
import shlex
import shutil
import signal
import socket
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any

# External
import typer

# Internal
from score.benchmark.progress import Log, duration
from score.benchmark.run import _folder_name
from score.build_record import (
    BuildRecord,
    BuildRecordError,
    check_binaries,
    read_build,
)
from score.layout import HOST_TIMING_FOLDER
from score.spike_check import CLEAN_BINARY, _git, _sha256, _top_level
from score.timing.c_runs import prepare_series
from score.timing.execute import Settings, run_unit
from score.timing.progress import Progress
from score.timing.results import (
    Provenance,
    publish_summary,
    stored_result,
    write_result,
    write_tables,
)
from score.timing.units import (
    C_WHOLE_PC,
    RESULT_SUFFIX,
    SERIES_CAP,
    SERIES_CSV,
    TORCH,
    Unit,
    all_units,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The default work folder, under the worktree's top level
WORK_FOLDER = Path("tspulse/build/host-timing")

# The tracked copy of each label's machine summaries, under the worktree's top level
TRACKED_FOLDER = Path("benchmark/results/host-timing")

# The defaults: on tesla, CPU 2 is a performance core and 10 the physical core count; other machines pass `--pin-cpu` and `--parallel`
DEFAULT_REPEATS = 3
DEFAULT_PIN_CPU = 2
DEFAULT_PARALLEL = 10

# The distributions whose versions decide whether a PyTorch unit still counts
TORCH_DISTRIBUTION = "torch"
GRANITE_DISTRIBUTION = "granite-tsfm"

# GNU time, which starts every C process and reports its peak memory
TIME_PROGRAM = "time"

# The only build target that runs on this machine
HOST_TARGET = "host"

# The exit code after Ctrl-C or SIGTERM, the shell's own for SIGINT
EXIT_INTERRUPTED = 130


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _read_builds(folders: list[Path]) -> dict[str, tuple[BuildRecord, Path]]:
    """Read each build folder's record and check its `tspulse` against it.

    Parameters
    ----------
    folders : list[Path]
        The build folders, in the order given.

    Returns
    -------
    dict[str, tuple[BuildRecord, Path]]
        Each build's record and `tspulse`, by the build's name, in the order given.

    Raises
    ------
    typer.Exit
        With code 1 when a record is missing or malformed, a binary doesn't match it, a build is not for the host, a build is named `torch`, or two builds share a name.
    """

    builds: dict[str, tuple[BuildRecord, Path]] = {}
    for folder in folders:
        folder = folder.resolve()
        try:
            record = read_build(folder)
            check_binaries(record, folder, (CLEAN_BINARY,))
        except BuildRecordError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(1) from error
        if record.target != HOST_TARGET:
            typer.echo(
                f"{folder} is a {record.target} build: time-host times {HOST_TARGET} builds.",
                err=True,
            )
            raise typer.Exit(1)
        if record.name == TORCH:
            typer.echo(
                f"{folder} is named {TORCH!r}, which PyTorch's rows use: no build can take it.",
                err=True,
            )
            raise typer.Exit(1)
        if record.name in builds:
            typer.echo(
                f"{folder} has the build name {record.name!r}, which {builds[record.name][1].parent} has too: each build is timed once.",
                err=True,
            )
            raise typer.Exit(1)
        builds[record.name] = record, folder / CLEAN_BINARY
    return builds


def _provenances(
    top: Path,
    machine: str,
    builds: dict[str, tuple[BuildRecord, Path]],
    pin_cpu: int,
    parallel: int,
) -> dict[str, Provenance]:
    """Describe where, from what and how each build's units run, and PyTorch's.

    Parameters
    ----------
    top : Path
        The worktree's top level.
    machine : str
        The label of the computer.
    builds : dict[str, tuple[BuildRecord, Path]]
        Each C build's record and `tspulse`, by the build's name.
    pin_cpu : int
        The CPU the pinned setups run on.
    parallel : int
        How many C processes, and PyTorch threads, the whole-PC setups use.

    Returns
    -------
    dict[str, Provenance]
        One per C build given, by its name, and one for PyTorch under `torch`.
    """

    shared: dict[str, Any] = {
        "machine": machine,
        "commit": _git(top, "rev-parse", "HEAD"),
        "tree": _git(top, "rev-parse", "HEAD^{tree}"),
        # The tracked summary this command copies is its output, not the program's state, so it must not turn the run dirty
        "dirty": bool(
            _git(top, "status", "--porcelain", "--", ".", f":(exclude){TRACKED_FOLDER}")
        ),
        "pin_cpu": pin_cpu,
        "parallel": parallel,
        "command": shlex.join(sys.argv),
    }
    provenances = {
        name: Provenance(
            **shared,
            build=name,
            build_record=asdict(record),
            sha256=_sha256(path),
            torch=None,
            granite_tsfm=None,
        )
        for name, (record, path) in builds.items()
    }
    provenances[TORCH] = Provenance(
        **shared,
        build=None,
        build_record=None,
        sha256=None,
        torch=version(TORCH_DISTRIBUTION),
        granite_tsfm=version(GRANITE_DISTRIBUTION),
    )
    return provenances


def _saved_results(
    units: list[Unit],
    folder: Path,
    provenances: dict[str, Provenance],
    log: Log | None,
) -> dict[Unit, dict[str, Any]]:
    """Read every unit's saved result that still counts.

    Parameters
    ----------
    units : list[Unit]
        Every unit of the call, in order.
    folder : Path
        The label's machine folder.
    provenances : dict[str, Provenance]
        Each build's provenance, `torch` included.
    log : Log or None
        Where a saved result that can't be read is reported, or None to leave it out silently.

    Returns
    -------
    dict[Unit, dict[str, Any]]
        The saved results, in unit order. A unit without one, with another skip key or with an unreadable one is missing, so it runs again.
    """

    saved = {}
    for unit in units:
        try:
            stored = stored_result(folder, unit, provenances[unit.build])
        except (OSError, ValueError) as error:
            if log is not None:
                log.line(f"{unit.name}: unreadable result, run again: {error}")
            continue
        if stored is not None:
            saved[unit] = stored
    return saved


def _time_host(
    builds: Annotated[
        list[Path],
        typer.Argument(
            help="The host build folders to time, each holding `tspulse` and its `build.json`."
        ),
    ],
    label: Annotated[
        str,
        typer.Option(
            help="Names this timing run: results go under `host-timing/<label>/<machine>/`. A machine's own folder, such as `tesla`, is refused."
        ),
    ],
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
    repeats: Annotated[
        int, typer.Option(min=1, help="How many times each measurement runs.")
    ] = DEFAULT_REPEATS,
    pin_cpu: Annotated[
        int,
        typer.Option(
            min=0,
            help="The CPU the one-core C runs and the one-thread PyTorch runs are pinned to.",
        ),
    ] = DEFAULT_PIN_CPU,
    parallel: Annotated[
        int,
        typer.Option(
            min=1,
            help="How many C processes, and PyTorch threads, the whole-PC setups use.",
        ),
    ] = DEFAULT_PARALLEL,
    machine: Annotated[
        str | None,
        typer.Option(
            help="A label for this computer, naming its folder and log under the label. Defaults to the hostname."
        ),
    ] = None,
    work: Annotated[
        Path | None,
        typer.Option(
            help="The folder for the row files and the C program's output. Defaults to tspulse/build/host-timing in this worktree."
        ),
    ] = None,
    redo: Annotated[
        bool,
        typer.Option(
            "--redo",
            help="Delete this label's saved units for this machine and run every unit again.",
        ),
    ] = False,
) -> None:
    """Time the C port's builds and IBM's PyTorch pipeline on two series, on one core and on the whole computer.

    Parameters
    ----------
    builds : list[Path]
        The host build folders.
    label : str
        The name of this timing run.
    data : Path or None
        The folder with the data.
    repeats : int
        How many times each measurement runs.
    pin_cpu : int
        The CPU the pinned setups run on.
    parallel : int
        How many C processes, and PyTorch threads, the whole-PC setups use.
    machine : str or None
        The label of this computer.
    work : Path or None
        The work folder.
    redo : bool
        Whether to delete this label's saved unit results for this machine first, so every unit runs again and the tables hold only this call's.

    Raises
    ------
    typer.Exit
        With code 1 when something needed is missing, a build's record or binary is refused, `--label` is a machine's own folder, `--pin-cpu` is not a CPU this process may use, `--parallel` is more than a series' windows, or any unit failed, and with `EXIT_INTERRUPTED` after Ctrl-C or SIGTERM.
    """

    # SIGTERM otherwise ends the parent without unwinding, and its children keep running. As Ctrl-C it kills them first
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    # Step 1: the label, the data folder, the builds checked against their records, GNU time and the CPU to pin to
    _folder_name(label, "--label")
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    # A machine's own results have a log of this name, and a label must not write into them
    machine_log = data / HOST_TIMING_FOLDER / f"{label}.log"
    if machine_log.exists():
        typer.echo(
            f"{label!r} is a machine's own results folder ({machine_log} exists): pick another --label.",
            err=True,
        )
        raise typer.Exit(1)
    checked = _read_builds(builds)
    binaries = {name: path for name, (_, path) in checked.items()}
    launcher = shutil.which(TIME_PROGRAM)
    if launcher is None:
        typer.echo(
            "GNU time is missing: run this inside the dev shell, from `nix develop`.",
            err=True,
        )
        raise typer.Exit(1)
    if pin_cpu not in os.sched_getaffinity(0):
        typer.echo(f"--pin-cpu {pin_cpu} is not a CPU this process may use.", err=True)
        raise typer.Exit(1)

    # Step 2: where results, the log and scratch files go, and each build's provenance
    top = _top_level()
    machine = machine or socket.gethostname()
    results_folder = data / HOST_TIMING_FOLDER / label
    folder = results_folder / machine
    log = Log(results_folder / f"{machine}.log")
    work = (work or top / WORK_FOLDER).resolve()
    work.mkdir(parents=True, exist_ok=True)
    provenances = _provenances(top, machine, checked, pin_cpu, parallel)

    # Step 3: the units still to run, and the saved times that seed the ETA. A redo runs every unit
    units = all_units(repeats, list(binaries))
    done = {} if redo else _saved_results(units, folder, provenances, log)
    pending = [unit for unit in units if unit not in done]

    # Step 4: the row file of each series with a C unit to run, written once per call
    c_series = {unit.series for unit in pending if unit.build != TORCH}
    prepared = {
        name: prepare_series(data, work, name, SERIES_CSV[name], SERIES_CAP[name])
        for name in sorted(c_series)
    }

    # Each whole-PC process needs at least one window
    for unit in pending:
        if unit.setup == C_WHOLE_PC and parallel > prepared[unit.series].windows:
            typer.echo(
                f"--parallel {parallel} is more than the {prepared[unit.series].windows} windows of {unit.series}.",
                err=True,
            )
            raise typer.Exit(1)

    # A redo deletes the saved units only once every check has passed, so no old result reaches the tables and a refused call loses nothing
    if redo:
        for path in folder.glob(f"*{RESULT_SUFFIX}"):
            path.unlink()
    began = time.perf_counter()
    log.line(
        f"time-host start: {len(pending)} units to run, {len(done)} already saved, machine {machine}, pin CPU {pin_cpu}, P {parallel}"
    )
    settings = Settings(
        data=data,
        work=work,
        machine=machine,
        binaries=binaries,
        launcher=Path(launcher),
        pin_cpu=pin_cpu,
        parallel=parallel,
        prepared=prepared,
        provenance=provenances,
    )

    # Step 5: the units, each saved and tabled as soon as it finishes
    progress = Progress(
        pending, {unit: stored["unit_seconds"] for unit, stored in done.items()}
    )
    failed = 0
    try:
        for unit in pending:
            unit_began = time.perf_counter()
            result = run_unit(unit, settings)
            seconds = time.perf_counter() - unit_began
            if isinstance(result, str):
                failed += 1
                log.line(f"problem: {unit.name}: {result}")
                log.line(progress.finished(unit, seconds, "failed", failed=True))
                continue
            write_result(folder, unit, result)
            write_tables(
                folder, list(_saved_results(units, folder, provenances, None).values())
            )
            per_window = f"{result.seconds_per_window * 1000:.1f} ms per window"
            log.line(progress.finished(unit, seconds, per_window, failed=False))
    except KeyboardInterrupt:
        log.line(
            "Interrupted: every program started was stopped; run the same command again to resume."
        )
        raise typer.Exit(EXIT_INTERRUPTED) from None

    # Step 6: the tables once more, since a call with nothing to run wrote none, and the tracked copy of the summary
    summary = write_tables(
        folder, list(_saved_results(units, folder, provenances, None).values())
    )
    tracked = top / TRACKED_FOLDER / label / f"{machine}.csv"
    publish_summary(summary, tracked)
    wall = time.perf_counter() - began
    log.line(
        f"time-host end: {len(pending) - failed} ran, {failed} failed, {len(done)} skipped, in {duration(wall)} ({wall:.0f} s); summary in {tracked}"
    )
    if failed:
        raise typer.Exit(1)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█
# ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█
# ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀

# The command's help is its docstring's summary, since typer would otherwise print the numpydoc sections too
app = typer.Typer(no_args_is_help=True, add_completion=False)
app.command(help=inspect.cleandoc(_time_host.__doc__ or "").partition("\n")[0])(
    _time_host
)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `time-host`."""

    app()
