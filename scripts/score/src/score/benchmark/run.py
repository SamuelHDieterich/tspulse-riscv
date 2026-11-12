"""`benchmark run`: run a labelled build's accuracy windows and count its instructions under Spike.

The instruction stage runs a few windows per series under Spike with `--counts`, and checks each against the host build byte for byte.
The accuracy stage runs every window of the chosen scopes on the host build, in chunks, and keeps their errors for `collect` to score.
Every unit saves its own folder under the label's, so a run resumes, and several computers can share the folder with `--shard`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import shutil
import signal
import socket
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from score.benchmark.execute import Outcome, Settings, run_accuracy, run_instruction
from score.benchmark.progress import ASSUMED_SECONDS, Log
from score.benchmark.results import (
    HostCheck,
    LabelIdentity,
    Platform,
    Provenance,
    create_identity,
    delete_label_results,
    read_identity,
)
from score.benchmark.scope import Scope, Series, read_series
from score.benchmark.stage import StageTally, run_stage, split_done
from score.benchmark.units import (
    HOST_CHUNK_CHANNEL_WINDOWS,
    SPIKE_CHUNK_CHANNEL_WINDOWS,
    AccuracyUnit,
    CountWindows,
    InstructionUnit,
    Shard,
    accuracy_units,
    instruction_units,
    order_and_shard,
    parse_shard,
)
from score.layout import BENCHMARK_FOLDER
from score.series_check import _find_series, write_rows
from score.spike_check import (
    CLEAN_BINARY,
    HOST_ROLE,
    ISA,
    PK_VARIABLE,
    RV64_ROLE,
    _git,
    _sha256,
    _top_level,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Where the row files and each unit's scratch files go, under the git top level
WORK_FOLDER = Path("tspulse/build/benchmark")

# The role pk's hash is recorded under, beside the two builds'
PK_ROLE = "pk"

# The ISA extension `rdinstret` needs; without it the counter read traps
COUNTER_EXTENSION = "zicntr"

# Extensions follow the base ISA after underscores, as in `rv64gc_zicntr`
ISA_SEPARATOR = "_"

# Names that a label or a machine can't take, since each names a folder or a file
RESERVED_NAMES = ("", ".", "..")

# The accuracy scopes a call runs when none is named
DEFAULT_SCOPES = [Scope.SCREEN, Scope.WHOLE_PYTHON]


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Stage(StrEnum):
    """Which stages a call runs.

    Attributes
    ----------
    INSTRUCTIONS : str
        The counted windows under Spike.
    ACCURACY : str
        The scopes' windows, for the accuracy tables.
    ALL : str
        Both, instructions first, so its shorter results land first.
    """

    INSTRUCTIONS = "instructions"
    ACCURACY = "accuracy"
    ALL = "all"


@dataclass(frozen=True)
class StagePlan:
    """How one stage runs: its units and the pool they run in.

    Attributes
    ----------
    units : list[AccuracyUnit] or list[InstructionUnit]
        The stage's selected units, largest first.
    runner : Callable
        Runs one unit.
    jobs : int
        How many units run at once.
    seconds_per_channel_window : float
        The assumed rate on one thread, for the first ETA.
    rebuild_table : bool
        Whether the stage keeps this machine's instruction table.
    """

    units: list[AccuracyUnit] | list[InstructionUnit]
    runner: Callable[..., Outcome]
    jobs: int
    seconds_per_channel_window: float
    rebuild_table: bool


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _write_rows(
    units: list[AccuracyUnit] | list[InstructionUnit], data: Path, settings: Settings
) -> None:
    """Write the row files the units need and the work folder lacks, before any unit starts, so no two units write the same file.

    Parameters
    ----------
    units : list[AccuracyUnit] or list[InstructionUnit]
        The units about to run.
    data : Path
        The folder with the data.
    settings : Settings
        What every unit shares.

    Raises
    ------
    typer.Exit
        With code 1 when a series' CSV is missing or can't be turned into a row file.
    """

    for series in sorted({unit.series for unit in units}):
        rows_path = settings.rows_path(series)
        # A row file is only replaced whole, so one that exists is complete and is kept for reuse
        if rows_path.is_file():
            continue
        csv_path = _find_series(data, series)
        if csv_path is None:
            typer.echo(f"No series {series} under {data}.", err=True)
            raise typer.Exit(1)
        try:
            write_rows(csv_path, rows_path)
        except (OSError, ValueError, KeyError) as error:
            typer.echo(f"Cannot read {csv_path}: {error}", err=True)
            raise typer.Exit(1) from error


def _check_label(
    label_folder: Path, identity: LabelIdentity, redo: bool, log: Log
) -> None:
    """Fix the label's identity on its first run, or check this run against it.

    Parameters
    ----------
    label_folder : Path
        The label's folder.
    identity : LabelIdentity
        This run's binaries and settings.
    redo : bool
        Whether to delete the label's results and start it again.
    log : Log
        Where the lines go.

    Raises
    ------
    typer.Exit
        With code 1 when the stored identity can't be read or differs from this run's.
    """

    if redo:
        delete_label_results(label_folder)
        log.line(
            f"--redo: deleted every machine's results of label {label_folder.name}"
        )
    if create_identity(label_folder, identity):
        log.line(f"label {label_folder.name}: identity written to label.json")
        return
    try:
        stored = read_identity(label_folder)
    except (OSError, ValueError) as error:
        typer.echo(f"Cannot read the label's identity: {error}", err=True)
        raise typer.Exit(1) from error
    differences = identity.differences(stored) if stored is not None else []
    if differences:
        typer.echo(
            f"Label {label_folder.name} was started with other binaries or settings:\n  "
            + "\n  ".join(differences)
            + "\nUse another --label, or --redo to delete its results and start it again.",
            err=True,
        )
        raise typer.Exit(1)


def _folder_name(value: str, option: str) -> str:
    """Check that a label or machine can name a folder or file on its own.

    Parameters
    ----------
    value : str
        The option's value.
    option : str
        The option, such as `--label`, which names it in the error.

    Returns
    -------
    str
        The value unchanged.

    Raises
    ------
    typer.BadParameter
        When the value is empty, `.` or `..`, or holds a path separator.
    """

    if value in RESERVED_NAMES or Path(value).name != value:
        raise typer.BadParameter(f"{value!r} is not a folder name.", param_hint=option)
    return value


def _stage_plan(
    stage: Stage,
    series: list[Series],
    shard: Shard | None,
    scopes: list[Scope],
    window_cap: int | None,
    count_windows: CountWindows,
    accuracy_on: Platform,
    host_check: HostCheck,
    jobs: int,
    spike_jobs: int,
) -> StagePlan:
    """Choose a stage's units, its runner, its pool size and its assumed rate.

    Parameters
    ----------
    stage : Stage
        `instructions` or `accuracy`.
    series : list[Series]
        The series asked for.
    shard : Shard or None
        The part of each stage's units to keep, or None for all.
    scopes : list[Scope]
        The accuracy scopes.
    window_cap : int or None
        The most windows kept of each scope range.
    count_windows : CountWindows
        Which windows to count.
    accuracy_on : Platform
        Where the accuracy units run.
    host_check : HostCheck
        Whether instruction units also run on the host.
    jobs : int
        Host accuracy units at once.
    spike_jobs : int
        Spike units at once.

    Returns
    -------
    StagePlan
        The stage's plan.
    """

    # Every instruction unit runs under Spike, plus a fraction of a second on the host when checked
    if stage == Stage.INSTRUCTIONS:
        rate = ASSUMED_SECONDS[Platform.SPIKE]
        if host_check == HostCheck.IDENTICAL:
            rate += ASSUMED_SECONDS[Platform.HOST]
        units = order_and_shard(instruction_units(series, count_windows), shard)
        return StagePlan(units, run_instruction, spike_jobs, rate, rebuild_table=True)

    # Accuracy units are cut smaller under Spike, which then also sets the pool size
    on_spike = accuracy_on == Platform.SPIKE
    chunk = SPIKE_CHUNK_CHANNEL_WINDOWS if on_spike else HOST_CHUNK_CHANNEL_WINDOWS
    units = order_and_shard(accuracy_units(series, scopes, window_cap, chunk), shard)
    return StagePlan(
        units,
        run_accuracy,
        spike_jobs if on_spike else jobs,
        ASSUMED_SECONDS[accuracy_on],
        rebuild_table=False,
    )


def run(
    host_build: Annotated[
        Path, typer.Argument(help="The host build folder, holding `tspulse`.")
    ],
    rv64_build: Annotated[
        Path,
        typer.Argument(
            help="The rv64 build folder, holding `tspulse` built with `--counts`."
        ),
    ],
    label: Annotated[
        str,
        typer.Option(
            help="The name the results go under. Its first run fixes its binaries and settings."
        ),
    ],
    stage: Annotated[
        Stage,
        typer.Option(
            help="The counted windows under Spike, the accuracy windows, or both, instructions first."
        ),
    ] = Stage.ALL,
    jobs: Annotated[
        int, typer.Option(min=1, help="Accuracy units run at once on the host.")
    ] = 1,
    spike_jobs: Annotated[
        int,
        typer.Option(
            min=1,
            help="Units run at once under Spike: every instruction unit, and accuracy units with `--accuracy-on spike`.",
        ),
    ] = 1,
    series: Annotated[
        list[str] | None,
        typer.Option(
            help="Only this working series, by CSV name without `.csv`. Repeatable."
        ),
    ] = None,
    shard: Annotated[
        str | None,
        typer.Option(
            help="K/N: keep every N-th unit of each stage's largest-first list, from the K-th (0-based), so N computers sharing the data folder split a run."
        ),
    ] = None,
    scope: Annotated[
        list[Scope] | None,
        typer.Option(
            help="The accuracy windows: each univariate screen span, every window of the Python baseline's series, or both. Repeatable; defaults to both."
        ),
    ] = None,
    count_windows: Annotated[
        CountWindows,
        typer.Option(
            help="The counted windows: a univariate series' first, middle and last screen window, or only the middle; a multivariate series' middle window either way."
        ),
    ] = CountWindows.ENDS_AND_MIDDLE,
    window_cap: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Keep only the first N windows of each series' scope range, for short screens and smoke tests. Defaults to no cap.",
        ),
    ] = None,
    redo: Annotated[
        bool,
        typer.Option(
            "--redo",
            help="Delete the label's results, on every machine sharing the data folder, and start it again. Refused with --shard.",
        ),
    ] = False,
    isa: Annotated[
        str, typer.Option(help="Spike's ISA string. Counting needs `zicntr`.")
    ] = ISA,
    accuracy_on: Annotated[
        Platform,
        typer.Option(help="Run the accuracy units on the host build or under Spike."),
    ] = Platform.HOST,
    host_check: Annotated[
        HostCheck,
        typer.Option(
            help="Also run each counted window on the host and require byte-identical errors, or skip that."
        ),
    ] = HostCheck.IDENTICAL,
    spike_option: Annotated[
        list[str] | None,
        typer.Option(
            help="An extra Spike option for the counted windows, placed before pk, such as `--spike-option=-g`. Repeatable."
        ),
    ] = None,
    work: Annotated[
        Path | None,
        typer.Option(
            help="The folder for the row files and the units' scratch files. Defaults to tspulse/build/benchmark in this worktree."
        ),
    ] = None,
    machine: Annotated[
        str | None,
        typer.Option(
            help="A label for this computer, recorded with each result and naming its log. Defaults to the hostname."
        ),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
) -> None:
    """Run a build's accuracy windows on the host and count its instructions under Spike, skipping units already done.

    Parameters
    ----------
    host_build : Path
        The host build folder.
    rv64_build : Path
        The rv64 build folder.
    label : str
        The benchmark label.
    stage : Stage
        Which stages to run.
    jobs : int
        Host accuracy units at once.
    spike_jobs : int
        Spike units at once.
    series : list[str] or None
        Series stems to keep, or every working series.
    shard : str or None
        `K/N`, or None for every unit.
    scope : list[Scope] or None
        The accuracy scopes, or both.
    count_windows : CountWindows
        Which windows to count.
    window_cap : int or None
        The most windows kept of each scope range.
    redo : bool
        Whether to delete the label's results first.
    isa : str
        Spike's ISA string.
    accuracy_on : Platform
        Where the accuracy units run.
    host_check : HostCheck
        Whether instruction units are checked against the host.
    spike_option : list[str] or None
        Extra Spike options for the counted windows.
    work : Path or None
        The work folder.
    machine : str or None
        The label of this computer.
    data : Path or None
        The folder with the data.

    Raises
    ------
    typer.Exit
        With code 1 when something needed is missing, a program fails, or an instruction unit differs from the host, whether run now or already on disk.
    """

    # SIGTERM otherwise ends the parent without unwinding, and its Spikes keep running. As Ctrl-C it kills them first
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    # Step 1: the options that can't go together, and the stages asked for
    parsed_shard = parse_shard(shard)
    if redo and parsed_shard is not None:
        raise typer.BadParameter(
            "--redo deletes every machine's results, so it can't run as one shard.",
            param_hint="--redo",
        )
    _folder_name(label, "--label")
    if machine is not None:
        _folder_name(machine, "--machine")
    stages = [Stage.INSTRUCTIONS, Stage.ACCURACY] if stage == Stage.ALL else [stage]
    if Stage.INSTRUCTIONS in stages and COUNTER_EXTENSION not in isa.split(
        ISA_SEPARATOR
    ):
        raise typer.BadParameter(
            f"{isa!r} lacks {COUNTER_EXTENSION}, without which `rdinstret` traps.",
            param_hint="--isa",
        )
    scopes = sorted(set(scope or DEFAULT_SCOPES))
    spike_options = spike_option or []

    # Step 2: the data folder, Spike, pk and the binaries, hashed once so the skip decision and every result use the same values
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    pk = os.environ.get(PK_VARIABLE)
    if not pk or shutil.which("spike") is None:
        typer.echo(
            f"Spike or ${PK_VARIABLE} is missing. Run this inside the dev shell, from `nix develop`.",
            err=True,
        )
        raise typer.Exit(1)
    binaries = {
        HOST_ROLE: host_build.resolve() / CLEAN_BINARY,
        RV64_ROLE: rv64_build.resolve() / CLEAN_BINARY,
        PK_ROLE: Path(pk),
    }
    for path in binaries.values():
        if not path.is_file():
            typer.echo(f"Not found: {path}", err=True)
            raise typer.Exit(1)
    sha256 = {role: _sha256(path) for role, path in binaries.items()}

    # Step 3: the series asked for, checked before the label's folder is made
    top = _top_level()
    working = read_series(top)
    known = {one.stem for one in working}
    unknown = sorted(set(series or []) - known)
    if unknown:
        raise typer.BadParameter(
            f"Not a working series: {', '.join(unknown)}", param_hint="--series"
        )
    chosen: list[Series] = [one for one in working if not series or one.stem in series]

    # Step 4: the worktree's state, the label's folder and its log
    machine = machine or socket.gethostname()
    label_folder = data / BENCHMARK_FOLDER / label
    log = Log(label_folder / f"{machine}.log")
    provenance = Provenance(
        machine=machine,
        commit=_git(top, "rev-parse", "HEAD"),
        tree=_git(top, "rev-parse", "HEAD^{tree}"),
        dirty=bool(_git(top, "status", "--porcelain")),
        sha256=sha256,
    )
    settings = Settings(
        label=label,
        label_folder=label_folder,
        host_binary=binaries[HOST_ROLE],
        rv64_binary=binaries[RV64_ROLE],
        pk=binaries[PK_ROLE],
        work=(work or top / WORK_FOLDER).resolve(),
        isa=isa,
        spike_options=spike_options,
        accuracy_on=accuracy_on,
        host_check=host_check,
        provenance=provenance,
    )
    settings.work.mkdir(parents=True, exist_ok=True)

    # Step 5: the label's identity, fixed by its first run
    identity = LabelIdentity(
        sha256=sha256,
        scopes=[str(one) for one in scopes],
        window_cap=window_cap,
        host_chunk_channel_windows=HOST_CHUNK_CHANNEL_WINDOWS,
        spike_chunk_channel_windows=SPIKE_CHUNK_CHANNEL_WINDOWS,
        count_windows=count_windows,
        isa=isa,
        accuracy_on=accuracy_on,
        host_check=host_check,
        spike_options=spike_options,
    )
    _check_label(label_folder, identity, redo, log)

    # Step 6: each stage's units, largest first, without those already done; then the stage
    tally = StageTally()
    try:
        for current in stages:
            plan = _stage_plan(
                current,
                chosen,
                parsed_shard,
                scopes,
                window_cap,
                count_windows,
                accuracy_on,
                host_check,
                jobs,
                spike_jobs,
            )
            pending, skipped, stored_differing = split_done(plan.units, settings, log)
            _write_rows(pending, data, settings)
            stage_tally = run_stage(
                current,
                pending,
                skipped,
                plan.runner,
                plan.jobs,
                plan.seconds_per_channel_window,
                settings,
                log,
                plan.rebuild_table,
            )
            tally.failed += stage_tally.failed
            tally.differing += stage_tally.differing + stored_differing
    except KeyboardInterrupt:
        log.line(
            "Interrupted: every program started was stopped; run the same command again to resume."
        )
        raise typer.Exit(1) from None

    # Step 7: the verdict
    if tally.failed or tally.differing:
        log.line(
            f"Finished with {tally.failed} failed and {tally.differing} differing units."
        )
        raise typer.Exit(1)
