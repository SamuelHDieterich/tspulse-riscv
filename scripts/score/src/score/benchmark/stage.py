"""Running one stage of a benchmark call: which of its units are already done, and the rest in a pool with a log line each."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# Internal
from score.benchmark.execute import Outcome, Settings
from score.benchmark.progress import SEPARATOR, Log, StageProgress, duration, eta
from score.benchmark.results import (
    NOT_CHECKED,
    AccuracyResult,
    InstructionResult,
    stored_result,
    write_instruction_table,
)
from score.benchmark.units import AccuracyUnit, InstructionUnit


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass
class StageTally:
    """What went wrong in a stage, for the call's exit code.

    Attributes
    ----------
    failed : int
        Units whose program failed or wrote something unexpected.
    differing : int
        Instruction units whose host and Spike errors differ, run now or already on disk.
    """

    failed: int = 0
    differing: int = 0


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _detail(outcome: AccuracyResult | InstructionResult | str, cost: int) -> str:
    """Describe a finished unit for its log line.

    Parameters
    ----------
    outcome : AccuracyResult, InstructionResult or str
        The unit's result, or the message it failed with.
    cost : int
        Its channel-windows.

    Returns
    -------
    str
        Such as `5,000 ch-win in 1,104 s`, with the instructions and the host check's verdict for an instruction unit.
    """

    if isinstance(outcome, str):
        return f"FAILED: {outcome}"
    if isinstance(outcome, AccuracyResult):
        return f"{cost:,} ch-win in {outcome.seconds:,.0f} s"
    verdict = {True: "identical", False: "DIFFERENT from the host"}.get(
        outcome.identical, NOT_CHECKED
    )
    return SEPARATOR.join(
        (
            f"{cost:,} ch-win in {outcome.spike_seconds:,.0f} s",
            f"{outcome.program_instructions:,} instructions",
            verdict,
        )
    )


def _rewrite_table(settings: Settings, log: Log, reported: set[str]) -> bool:
    """Rewrite this machine's instruction table, logging what it had to leave out.

    Parameters
    ----------
    settings : Settings
        What every unit shares.
    log : Log
        Where the lines go.
    reported : set[str]
        Problems already logged during this stage, so each is logged once.

    Returns
    -------
    bool
        False when the table couldn't be written.
    """

    try:
        _, problems = write_instruction_table(
            settings.label_folder, settings.provenance.machine
        )
    except OSError as error:
        log.line(f"cannot write the instruction table: {error}")
        return False
    for problem in problems:
        if problem not in reported:
            reported.add(problem)
            log.line(f"left out of the instruction table: {problem}")
    return True


def run_stage[UnitT: (AccuracyUnit, InstructionUnit)](
    stage: str,
    units: list[UnitT],
    skipped: int,
    runner: Callable[[UnitT, Settings], Outcome],
    jobs: int,
    seconds_per_channel_window: float,
    settings: Settings,
    log: Log,
    rebuild_table: bool,
) -> StageTally:
    """Run a stage's pending units up to `jobs` at once, logging each as it finishes.

    A unit that fails, whether its program fails or something raises while it runs, is logged and counted, and the other units go on.

    Parameters
    ----------
    stage : str
        The stage's name, which starts its lines.
    units : list[AccuracyUnit] or list[InstructionUnit]
        The units to run, largest first.
    skipped : int
        How many of the stage's selected units already have a result.
    runner : Callable
        Runs one unit.
    jobs : int
        How many units run at once.
    seconds_per_channel_window : float
        The assumed rate on one thread, for the first ETA.
    settings : Settings
        What every unit shares.
    log : Log
        Where the lines go.
    rebuild_table : bool
        Whether the stage keeps this machine's instruction table, rebuilt as units finish and once more at the end.

    Returns
    -------
    StageTally
        How many units failed or differed.

    Raises
    ------
    KeyboardInterrupt
        On Ctrl-C or SIGTERM, after every program started has been killed.
    """

    # Step 1: the start line, with an ETA from the assumed rate spread over the jobs that can actually run at once
    total = sum(unit.cost for unit in units)
    progress = StageProgress(stage, len(units), total)
    parallel = max(1, min(jobs, len(units)))
    log.line(
        SEPARATOR.join(
            (
                f"{stage}: {len(units)} units, {total:,} ch-win to run",
                f"{skipped} skipped",
                f"{jobs} jobs",
                f"ETA {eta(total * seconds_per_channel_window / parallel)} (assumed rate)",
            )
        )
    )

    # Step 2: the units. Only this thread logs and writes the instruction table, so each has one writer
    tally = StageTally()
    reported: set[str] = set()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures: dict[Future[Outcome], UnitT] = {}
        try:
            for unit in units:
                futures[pool.submit(runner, unit, settings)] = unit
            for future in as_completed(futures):
                unit = futures[future]

                # Anything a unit raises, such as a full disk while saving, fails that unit only. A unit that raised saved no `result.json`, so it reruns
                try:
                    outcome = future.result()
                except Exception as error:
                    outcome = f"{type(error).__name__}: {error}"
                if outcome is None:
                    continue
                if isinstance(outcome, str):
                    tally.failed += 1
                elif isinstance(outcome, InstructionResult):
                    tally.differing += outcome.identical is False
                    # A failed write here is only logged: Step 3 writes the table again and counts a failure only if that write fails too
                    if rebuild_table:
                        _rewrite_table(settings, log, reported)
                log.line(
                    progress.finished(unit.name, unit.cost, _detail(outcome, unit.cost))
                )

        # On Ctrl-C, and on any error of this thread's own, no program is left running. A unit cut short saved no `result.json`, so it reruns on the next call
        except BaseException:
            for future in futures:
                future.cancel()
            settings.processes.stop()
            raise

    # Step 3: rebuild the table from the saved results, even when every unit was skipped, so a failed write earlier is repaired
    if rebuild_table:
        if not _rewrite_table(settings, log, reported):
            tally.failed += 1

    # Step 4: the end line
    log.line(
        SEPARATOR.join(
            (
                f"{stage} done: {progress.done_units} units in {duration(progress.elapsed())}",
                f"{tally.failed} failed",
                f"{tally.differing} differing",
            )
        )
    )
    return tally


def split_done[UnitT: (AccuracyUnit, InstructionUnit)](
    units: list[UnitT], settings: Settings, log: Log
) -> tuple[list[UnitT], int, int]:
    """Set aside the units that already have a result from the label's binaries.

    Parameters
    ----------
    units : list[AccuracyUnit] or list[InstructionUnit]
        The stage's selected units.
    settings : Settings
        What every unit shares.
    log : Log
        Where a result that can't be read is reported.

    Returns
    -------
    tuple[list, int, int]
        The units still to run in the same order, how many were skipped, and how many skipped instruction units differ from the host.
    """

    pending = []
    skipped = differing = 0
    for unit in units:
        folder = settings.label_folder / unit.folder

        # A malformed result, perhaps half-synced from another machine, counts as not done, so the unit reruns and replaces it
        try:
            stored = stored_result(folder, settings.provenance.sha256)
        except (OSError, ValueError) as error:
            log.line(f"{unit.name}: unreadable result, rerunning: {error}")
            stored = None
        if stored is None:
            pending.append(unit)
            continue
        skipped += 1
        differing += stored.get("identical") is False
    return pending, skipped, differing
