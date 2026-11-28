"""Take one variant through its stages: build, reference check, `run`, `collect`, `profile` and the summary, each skipped when its results are already on disk."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# External
import click
import typer

from score.benchmark.collect import collect_label
from score.benchmark.progress import Log, duration
from score.benchmark.results import read_identity
from score.benchmark.run import EXIT_INTERRUPTED, run
from score.benchmark.scope import Series

# Internal
from score.build_record import BASELINE_NAME, BUILD_FILE, DEFAULT_STRIDE
from score.profile.cli import profile
from score.spike_check import CLEAN_BINARY, HELD_GROUP, Group
from score.variants.building import BuiltVariant, ResolvedVariant, build_variant
from score.variants.held_summary import held_status_text, write_held_variant_summary
from score.variants.listing import VariantError
from score.variants.reference import check_reference
from score.variants.stage_names import VariantStage
from score.variants.state import (
    COUNT_WINDOWS,
    GROUP_SCOPES,
    Selection,
    binaries_problem,
    collect_missing,
    delete_for_redo,
    is_done,
    label_folder,
    label_stride,
    read_collect,
    read_profile_hash,
    read_reference_hashes,
    read_selection,
    run_missing,
    stack_name,
    tables_folder,
    write_collect_record,
    write_selection,
)
from score.variants.summary import status_text, write_variant_summary


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Every variant's tables are compared with the baseline's, which is collected once and never rerun
AGAINST = BASELINE_NAME

# Opens every line a stage writes to the log, so the variant's lines stand out among the ones `run` writes there
LINE_PREFIX = "variants"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class RunOptions:
    """How this machine runs every variant, the same for all of them.

    Attributes
    ----------
    suffix : str
        Appended to each variant's name to make its label.
    selection : Selection
        The series and the window cap.
    jobs : int
        Host accuracy units at once.
    spike_jobs : int
        Spike runs at once, for the counted windows and the profile.
    machine : str or None
        The label of this computer, or None for the hostname.
    log_name : str
        The log file's name in each label's folder, the one `run` appends to.
    group : Group
        The series group of every label: the working series, or the held-back ones.
    shard : str or None
        `K/N`, to run only that share of each label's units and stop after the run stage, or None for every unit.
    stack_label : str or None
        The held-back stack's label, which the picks are paired with. None for the working group.
    """

    suffix: str
    selection: Selection
    jobs: int
    spike_jobs: int
    machine: str | None
    log_name: str
    group: Group
    shard: str | None
    stack_label: str | None


@dataclass
class VariantOutcome:
    """How a variant's turn ended.

    Attributes
    ----------
    label : str
        The variant's label.
    problems : list[str]
        One line per problem, already logged.
    interrupted : bool
        Whether Ctrl-C or SIGTERM stopped it, which stops the whole command.
    done : bool
        Whether the label holds every result once its turn is over.
    """

    label: str
    problems: list[str] = field(default_factory=list)
    interrupted: bool = False
    done: bool = False


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _problem(log: Log, outcome: VariantOutcome, text: str) -> None:
    """Log a problem and keep it for the exit code.

    Parameters
    ----------
    log : Log
        The label's log.
    outcome : VariantOutcome
        The variant's outcome.
    text : str
        What went wrong.
    """

    log.line(f"{LINE_PREFIX}: problem: {text}")
    outcome.problems.append(text)


def _stage[ResultT](
    stage: VariantStage,
    action: Callable[[], ResultT],
    log: Log,
    outcome: VariantOutcome,
) -> tuple[bool, ResultT | None]:
    """Run one stage, logging its start and its end with the wall time, and turn its failure into a problem.

    Parameters
    ----------
    stage : VariantStage
        The stage.
    action : Callable[[], ResultT]
        Runs the stage.
    log : Log
        The label's log.
    outcome : VariantOutcome
        The variant's outcome, which gets the problem or the interruption.

    Returns
    -------
    tuple[bool, ResultT or None]
        Whether the stage succeeded, and what it returned.
    """

    log.line(f"{LINE_PREFIX}: {stage} started")
    started = time.monotonic()
    try:
        result = action()
    except KeyboardInterrupt:
        outcome.interrupted = True
        log.line(
            f"{LINE_PREFIX}: {stage} interrupted after {duration(time.monotonic() - started)}"
        )
        return False, None
    except typer.Exit as stop:
        took = duration(time.monotonic() - started)
        # The stages catch Ctrl-C themselves and leave with this code, so it stops the command just as Ctrl-C does
        if stop.exit_code == EXIT_INTERRUPTED:
            outcome.interrupted = True
            log.line(f"{LINE_PREFIX}: {stage} interrupted after {took}")
            return False, None
        if stop.exit_code:
            _problem(
                log,
                outcome,
                f"{stage} failed after {took} with exit code {stop.exit_code}",
            )
            return False, None
        log.line(f"{LINE_PREFIX}: {stage} finished in {took}")
        return True, None
    except (click.exceptions.ClickException, VariantError) as error:
        took = duration(time.monotonic() - started)
        _problem(log, outcome, f"{stage} failed after {took}: {error}")
        return False, None
    log.line(
        f"{LINE_PREFIX}: {stage} finished in {duration(time.monotonic() - started)}"
    )
    return True, result


def _profile_hash(tables: Path) -> str | None:
    """Read the rv64 binary a label's profile was made with.

    Parameters
    ----------
    tables : Path
        The label's tracked results folder.

    Returns
    -------
    str or None
        The binary's hash, or None when there is no profile or its `build.json` can't be read, so that the profile is made again.
    """

    try:
        return read_profile_hash(tables)
    except (OSError, ValueError):
        return None


def _skip(stage: VariantStage, reason: str, log: Log) -> None:
    """Log that a stage is skipped and why.

    Parameters
    ----------
    stage : VariantStage
        The stage.
    reason : str
        Why its results already stand.
    log : Log
        The label's log.
    """

    log.line(f"{LINE_PREFIX}: {stage} skipped: {reason}")


def _run_skippable(
    label: str, data: Path, group_series: list[Series], group: Group, redo: bool
) -> bool:
    """Check whether `run` has nothing left to do for a label.

    Parameters
    ----------
    label : str
        The variant's label.
    data : Path
        The folder with the data.
    group_series : list[Series]
        Every series of the label's group.
    group : Group
        The label's series group.
    redo : bool
        Whether the label is being started again, which `run` must do even when its units are all there.

    Returns
    -------
    bool
        True when the label has an identity and every unit of its selection has a result.
    """

    if redo:
        return False
    folder = label_folder(data, label)
    try:
        identity = read_identity(folder)
        selection = read_selection(folder)
    except (OSError, ValueError):
        return False
    if identity is None or selection is None:
        return False
    return not run_missing(folder, identity, selection, group_series, group)


def _collect_skippable(folder: Path, tables: Path, run_skipped: bool) -> bool:
    """Check whether the label's tables still stand, so `collect` has nothing to add.

    Parameters
    ----------
    folder : Path
        The label's folder.
    tables : Path
        The label's tracked results folder.
    run_skipped : bool
        Whether `run` was skipped, so no unit is new.

    Returns
    -------
    bool
        True when no unit is new, both tables exist, and the last collect read the label's units and found no problem.
    """

    if not run_skipped:
        return False
    try:
        identity = read_identity(folder)
        collect = read_collect(folder)
    except (OSError, ValueError):
        return False
    if identity is None or collect is None:
        return False
    return not collect_missing(tables, identity, collect) and collect.problems == 0


def _profile_step(
    label: str,
    folder: Path,
    tables: Path,
    built: BuiltVariant,
    options: RunOptions,
    top: Path,
    data: Path,
    log: Log,
    outcome: VariantOutcome,
) -> bool:
    """Make a working label's profile, or derive it from its stack's, unless it stands already.

    The profile stays while the rv64 binary is the one its tables were made with.

    Parameters
    ----------
    label : str
        The variant's label.
    folder : Path
        The label's folder.
    tables : Path
        The label's tracked results folder.
    built : BuiltVariant
        The variant's builds.
    options : RunOptions
        How this machine runs the variants.
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    log : Log
        The label's log.
    outcome : VariantOutcome
        The variant's outcome, which gets the problem or the interruption.

    Returns
    -------
    bool
        False when the stage failed or was interrupted, so the variant's turn ends.
    """

    try:
        identity = read_identity(folder)
    except (OSError, ValueError) as error:
        _problem(log, outcome, f"cannot read the label's identity: {error}")
        return False
    # A stride variant takes its stack's profile, so the stack must have one; only its build record is copied
    if label_stride(identity) != DEFAULT_STRIDE:
        try:
            stack = stack_name(identity)
            if read_profile_hash(tables_folder(top, stack)) is None:
                _problem(
                    log, outcome, f"cannot derive the profile: {stack} has no profile"
                )
                return False
            log.line(f"{LINE_PREFIX}: {VariantStage.PROFILE}: derived from {stack}")
            tables.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(built.rv64_folder / BUILD_FILE, tables / BUILD_FILE)
        except (OSError, ValueError) as error:
            _problem(log, outcome, f"cannot derive the profile: {error}")
            return False
    elif _profile_hash(tables) == built.rv64.sha256[CLEAN_BINARY]:
        _skip(
            VariantStage.PROFILE, "the profile's build.json holds this rv64 binary", log
        )
    else:
        ok, _ = _stage(
            VariantStage.PROFILE,
            lambda: profile(
                label=label, build=built.rv64_folder, jobs=options.spike_jobs, data=data
            ),
            log,
            outcome,
        )
        if not ok:
            return False
    return True


def run_variant(
    variant: ResolvedVariant,
    options: RunOptions,
    redo: bool,
    top: Path,
    data: Path,
    group_series: list[Series],
) -> VariantOutcome:
    """Build a variant and run each stage whose results its label lacks, logging to the label's own log.

    A failed stage is a problem that skips the variant's later stages; Ctrl-C or SIGTERM ends the turn as interrupted.

    Parameters
    ----------
    variant : ResolvedVariant
        The list entry and its build name.
    options : RunOptions
        How this machine runs the variants.
    redo : bool
        Whether to delete the label's results and start it again.
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    group_series : list[Series]
        Every series of the options' group.

    Returns
    -------
    VariantOutcome
        The problems, whether it was interrupted, and whether the label is done.
    """

    label = variant.name + options.suffix
    folder = label_folder(data, label)
    tables = tables_folder(top, label)
    log = Log(folder / options.log_name)
    outcome = VariantOutcome(label=label)
    log.line(f"{LINE_PREFIX}: variant {variant.name} as label {label}")

    # Step 1: with --redo, what the command made for the label goes; `run --redo` then deletes its units and identity
    if redo:
        for path in delete_for_redo(top, data, label):
            log.line(f"{LINE_PREFIX}: --redo: deleted {path}")

    # Step 2: the builds, which find everything up to date when nothing changed
    built_ok, built = _stage(
        VariantStage.BUILD, lambda: build_variant(variant, top), log, outcome
    )
    if not built_ok:
        return outcome
    if built is None:
        _problem(log, outcome, f"{VariantStage.BUILD} returned no builds")
        return outcome

    # Step 3: the binaries guard. The label's results stand for the binaries label.json fixed, so other binaries can't add to them
    if not redo:
        problem = binaries_problem(
            folder, built.host.sha256[CLEAN_BINARY], built.rv64.sha256[CLEAN_BINARY]
        )
        if problem is not None:
            _problem(log, outcome, problem)
            return outcome

    # Step 4: the selection, fixed by the label's first run. A different one was refused before any build
    if read_selection(folder) is None:
        write_selection(folder, options.selection)

    # Step 5: the reference check, kept while the host binaries are the ones it was made with. An unreadable reference.json counts as no check, so it is made again
    try:
        reference_hashes = read_reference_hashes(folder)
    except (OSError, ValueError):
        reference_hashes = None
    if reference_hashes == built.host.sha256:
        _skip(VariantStage.REFERENCE, "reference.json holds these host binaries", log)
    else:
        ok, differing = _stage(
            VariantStage.REFERENCE,
            lambda: check_reference(
                built.host_folder, built.host, folder, top, data, log.line
            ),
            log,
            outcome,
        )
        if not ok:
            return outcome
        log.line(f"{LINE_PREFIX}: {differing} of the reference windows differ")

    # Step 6: the accuracy and counted windows
    held = options.group == HELD_GROUP
    run_skipped = _run_skippable(label, data, group_series, options.group, redo)
    if run_skipped:
        _skip(VariantStage.RUN, "every unit of selection.json has a result", log)
    else:
        ok, _ = _stage(
            VariantStage.RUN,
            lambda: run(
                host_build=built.host_folder,
                riscv_build=built.rv64_folder,
                label=label,
                jobs=options.jobs,
                spike_jobs=options.spike_jobs,
                series=options.selection.series,
                scope=list(GROUP_SCOPES[options.group]),
                count_windows=COUNT_WINDOWS,
                window_cap=options.selection.window_cap,
                redo=redo,
                shard=options.shard,
                machine=options.machine,
                data=data,
            ),
            log,
            outcome,
        )
        if not ok:
            return outcome

    # A shard holds only some of the label's units, so the label is merged and collected where all of them are
    if options.shard is not None:
        log.line(
            f"{LINE_PREFIX}: shard {options.shard} is done: copy this label's folder to the machine that merges the shards and run `benchmark variants` there without --shard to collect it"
        )
        return outcome

    # Step 7: the tables, rescored when `run` ran or the last collect found a problem, since a problem stays until a collect finds none. A held-back label has no baseline run to compare with: its summary pairs it with the stack's tables itself
    if _collect_skippable(folder, tables, run_skipped):
        _skip(
            VariantStage.COLLECT,
            "both tables exist, no unit is new and collect.json holds no problem",
            log,
        )
    else:
        ok, collected = _stage(
            VariantStage.COLLECT,
            lambda: collect_label(label, None if held else AGAINST, data, log.line),
            log,
            outcome,
        )
        if not ok:
            return outcome
        if collected is None:
            _problem(log, outcome, f"{VariantStage.COLLECT} returned no tables")
            return outcome
        try:
            identity = read_identity(folder)
        except (OSError, ValueError) as error:
            _problem(log, outcome, f"cannot read the label's identity: {error}")
            return outcome
        if identity is None:
            _problem(log, outcome, "the label has no label.json after collect")
            return outcome
        write_collect_record(folder, collected.problems, identity.sha256)
        # The tables are written either way, so the later stages still run
        if collected.problems:
            _problem(log, outcome, f"collect found {collected.problems} problems")

    # Step 8: the profile of a working label. A held-back label keeps only the build record, for its summary
    if held:
        try:
            tables.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(built.rv64_folder / BUILD_FILE, tables / BUILD_FILE)
        except OSError as error:
            _problem(log, outcome, f"cannot copy the build record: {error}")
            return outcome
    elif not _profile_step(
        label, folder, tables, built, options, top, data, log, outcome
    ):
        return outcome

    # Step 9: the summary, cheap enough to rewrite on every call, whether the label is done or not. A held-back label's pairs it with the stack's tables; a working label's adds its exactness against the baseline
    if held:
        ok, held_written = _stage(
            VariantStage.SUMMARY,
            lambda: write_held_variant_summary(
                top, data, label, variant, group_series, options.stack_label
            ),
            log,
            outcome,
        )
        if not ok:
            return outcome
        if held_written is not None:
            log.line(f"{LINE_PREFIX}: status: {held_status_text(held_written)}")
    else:
        ok, written = _stage(
            VariantStage.SUMMARY,
            lambda: write_variant_summary(top, data, label, variant, group_series),
            log,
            outcome,
        )
        if not ok:
            return outcome
        if written is not None:
            summary, exactness = written
            log.line(
                f"{LINE_PREFIX}: {summary.verdict}, {exactness.windows_differing} of {exactness.windows_compared} windows differ, status: {status_text(summary)}"
            )
            if exactness.left_out:
                log.line(
                    f"{LINE_PREFIX}: left out of the comparison: {', '.join(exactness.left_out)}"
                )
            for text in exactness.problems:
                _problem(log, outcome, f"exactness: {text}")

    # Step 10: whether the label now holds everything. A problem in this call means it doesn't, whatever is on disk
    outcome.done = not outcome.problems and is_done(
        top, data, label, group_series, options.group
    )
    # A held-back label whose summary still names missing labels isn't final
    if held and held_written is not None and held_written.missing:
        outcome.done = False
    log.line(
        f"{LINE_PREFIX}: label {label} is {'done' if outcome.done else 'not done'}"
    )
    return outcome
