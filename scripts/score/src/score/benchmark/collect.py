"""`benchmark collect`: score a label's finished units and write its tracked tables.

It scores every series and scope whose units are all present and lists the rest, so it can run during a run, after a metric fix, or on another computer than the one that ran.
Accuracy is scored from the host build's errors with the pipeline's post-processing; Spike only counts instructions.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

# External
import typer
from tsfm_public.models.tspulse import TSPulseForReconstruction

# Internal
from score.benchmark.progress import duration
from score.benchmark.results import (
    LabelIdentity,
    Platform,
    find_results,
    read_identity,
)
from score.benchmark.scope import Scope, Series, read_series, scope_range
from score.benchmark.scoring import (
    ScopeScore,
    overlapping_units,
    score_scope,
    stitch,
)
from score.benchmark.summary import (
    print_against,
    print_instructions,
    print_mean_vus_pr,
    print_python_check,
)
from score.benchmark.tables import (
    ACCURACY_TABLE,
    INSTRUCTIONS_TABLE,
    TABLES_FOLDER,
    AccuracyRow,
    InstructionRow,
    accuracy_rows,
    compare_accuracy,
    compare_instructions,
    read_against_accuracy,
    read_against_instructions,
    read_baseline,
    write_table,
)
from score.benchmark.units import (
    SCORES_FOLDER,
    CountWindows,
    accuracy_units,
    instruction_units,
)
from score.layout import BENCHMARK_FOLDER, MODEL_FOLDER
from score.pipeline import load_model, load_series
from score.series_check import _find_series
from score.spike_check import _top_level


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Collected:
    """What a collection wrote.

    Attributes
    ----------
    accuracy_table : Path
        The tracked `accuracy.csv`.
    instruction_table : Path
        The tracked `instructions.csv`.
    missing : int
        How many series still lack units, for information only: a run limited to some series leaves the others missing by design.
    problems : int
        How many scopes could not be scored, windows differ, or `result.json` files could not be read or came from other binaries, which fails the collection.
    """

    accuracy_table: Path
    instruction_table: Path
    missing: int
    problems: int


@dataclass
class _Count:
    """How many of some units are done.

    Attributes
    ----------
    done : int
        The units with a result.
    total : int
        The units expected.
    """

    done: int = 0
    total: int = 0

    @property
    def complete(self) -> bool:
        """Whether every expected unit is done.

        Returns
        -------
        bool
            True when `done` has reached `total`.
        """

        return self.done >= self.total

    def __str__(self) -> str:
        """Write the count as `done/total`.

        Returns
        -------
        str
            Such as `3/5`.
        """

        return f"{self.done}/{self.total}"


@dataclass
class _Progress:
    """How much of one series' units are done, for the missing listing.

    Attributes
    ----------
    accuracy : _Count
        The accuracy units.
    instructions : _Count
        The instruction units.
    """

    accuracy: _Count
    instructions: _Count


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _score_series(
    one: Series,
    scopes: list[Scope],
    identity: LabelIdentity,
    label_folder: Path,
    data: Path,
    models: dict[int, TSPulseForReconstruction],
    progress: _Progress,
    problems: list[str],
    emit: Callable[[str], None],
) -> list[ScopeScore]:
    """Score every scope of a series whose units are all done.

    Parameters
    ----------
    one : Series
        The series.
    scopes : list[Scope]
        The label's scopes.
    identity : LabelIdentity
        The label's identity.
    label_folder : Path
        The label's folder.
    data : Path
        The folder with the data.
    models : dict[int, TSPulseForReconstruction]
        Loaded models by channel count, filled as needed.
    progress : _Progress
        This series' unit counts, which this function fills for accuracy.
    problems : list[str]
        Where a unit or scope that can't be used is reported.
    emit : Callable[[str], None]
        Prints one line.

    Returns
    -------
    list[ScopeScore]
        The scopes scored.
    """

    # Step 1: the series' units, and which of them are done
    # The chunk size the label was run with, so a label stays collectable after the defaults change
    chunk = (
        identity.spike_chunk_channel_windows
        if identity.accuracy_on == Platform.SPIKE
        else identity.host_chunk_channel_windows
    )
    units = accuracy_units([one], scopes, identity.window_cap, chunk)
    done, unit_problems = find_results(label_folder, units, identity.sha256)
    problems.extend(unit_problems)
    progress.accuracy = _Count(len(done), len(units))

    # Step 2: each scope with all its units done is stitched and scored
    scored = []
    frame = None
    for scope in scopes:
        # An array from an earlier pass is deleted first, so a scope not scored now leaves none from other binaries or settings
        destination = label_folder / SCORES_FOLDER / f"{one.stem}.{scope}.npz"
        destination.unlink(missing_ok=True)
        window_range = scope_range(one, scope, identity.window_cap)
        if window_range is None:
            continue
        start, stop = window_range
        needed = overlapping_units(units, start, stop)
        if not needed or any(unit not in done for unit in needed):
            continue
        began = time.monotonic()
        try:
            errors = stitch(label_folder, needed, start, stop)
            if frame is None:
                csv_path = _find_series(data, one.stem)
                if csv_path is None:
                    raise ValueError(f"no series {one.stem} under {data}")
                frame = load_series(csv_path)
            if one.channels not in models:
                models[one.channels] = load_model(data / MODEL_FOLDER, one.channels)
            scored.append(
                score_scope(
                    one,
                    scope,
                    frame,
                    models[one.channels],
                    errors,
                    start,
                    stop,
                    destination,
                )
            )
        except (OSError, ValueError, RuntimeError, KeyError, IndexError) as error:
            problems.append(
                f"{one.stem} {scope}: not scored: {type(error).__name__}: {error}"
            )
            continue
        emit(
            f"scored {one.stem} {scope}, windows {start} to {stop - 1}, in {duration(time.monotonic() - began)}"
        )
    return scored


def _instruction_rows(
    one: Series,
    identity: LabelIdentity,
    label_folder: Path,
    progress: _Progress,
    problems: list[str],
) -> list[InstructionRow]:
    """Read the instruction results of a series' counted windows.

    Parameters
    ----------
    one : Series
        The series.
    identity : LabelIdentity
        The label's identity.
    label_folder : Path
        The label's folder.
    progress : _Progress
        This series' unit counts, which this function fills for instructions.
    problems : list[str]
        Where a result that can't be used is reported.

    Returns
    -------
    list[InstructionRow]
        One row per counted window that is done.
    """

    units = instruction_units([one], CountWindows(identity.count_windows))
    done, unit_problems = find_results(label_folder, units, identity.sha256)
    problems.extend(unit_problems)
    progress.instructions = _Count(0, len(units))
    rows = []
    for unit, stored in done.items():
        try:
            rows.append(
                InstructionRow(
                    series=unit.series,
                    kind=str(unit.kind),
                    channels=unit.channels,
                    window=unit.window,
                    program_instructions=stored["program_instructions"],
                    per_channel=stored["program_instructions"] / unit.channels,
                    identical=stored["identical"],
                    spike_seconds=stored["spike_seconds"],
                    machine=stored["provenance"]["machine"],
                )
            )
        except (KeyError, TypeError) as error:
            problems.append(f"{unit.name}: result.json lacks {error}")
            continue
        progress.instructions.done += 1
    return rows


def collect_label(
    label: str,
    against: str | None,
    data: Path,
    emit: Callable[[str], None],
) -> Collected:
    """Score a label's finished units, write its tables and print the summaries.

    Parameters
    ----------
    label : str
        The label to collect.
    against : str or None
        Another label whose tables each series is compared with.
    data : Path
        The folder with the data.
    emit : Callable[[str], None]
        Prints one line.

    Returns
    -------
    Collected
        The tables written, what is still missing and how many problems were found.

    Raises
    ------
    typer.Exit
        With code 1 when the label has no identity, a table can't be read or written, or the model is missing.
    """

    # Step 1: the label's identity, which says which units to expect
    top = _top_level()
    label_folder = data / BENCHMARK_FOLDER / label
    try:
        identity = read_identity(label_folder)
    except (OSError, ValueError) as error:
        emit(f"Cannot read the label's identity: {error}")
        raise typer.Exit(1) from error
    if identity is None:
        emit(f"No label {label} under {label_folder.parent}.")
        raise typer.Exit(1)
    if not (data / MODEL_FOLDER).is_dir():
        emit(f"Not found: {data / MODEL_FOLDER}")
        raise typer.Exit(1)
    tables = top / TABLES_FOLDER / label
    against_tables = top / TABLES_FOLDER / against if against else None
    if against_tables is not None:
        for name in (ACCURACY_TABLE, INSTRUCTIONS_TABLE):
            if not (against_tables / name).is_file():
                emit(
                    f"Not found: {against_tables / name}. Collect label {against} first."
                )
                raise typer.Exit(1)

    # Step 2: every series' units, scored or read
    emit(f"collecting label {label} from {label_folder}")
    scopes = [Scope(name) for name in identity.scopes]
    models: dict[int, TSPulseForReconstruction] = {}
    scored: list[ScopeScore] = []
    instruction_rows: list[InstructionRow] = []
    progress: dict[str, _Progress] = {}
    problems: list[str] = []
    for one in read_series(top):
        progress[one.stem] = _Progress(_Count(), _Count())
        scored += _score_series(
            one,
            scopes,
            identity,
            label_folder,
            data,
            models,
            progress[one.stem],
            problems,
            emit,
        )
        instruction_rows += _instruction_rows(
            one, identity, label_folder, progress[one.stem], problems
        )

    # Step 3: the tables, with the other label's values beside them
    accuracy: list[AccuracyRow] = accuracy_rows(
        scored, read_baseline(top), compare_python=identity.window_cap is None
    )
    if against_tables is not None:
        accuracy = compare_accuracy(
            accuracy, read_against_accuracy(against_tables / ACCURACY_TABLE)
        )
        instruction_rows = compare_instructions(
            instruction_rows,
            read_against_instructions(against_tables / INSTRUCTIONS_TABLE),
        )
    try:
        write_table(tables / ACCURACY_TABLE, AccuracyRow, accuracy, against is not None)
        write_table(
            tables / INSTRUCTIONS_TABLE,
            InstructionRow,
            instruction_rows,
            against is not None,
        )
    except OSError as error:
        emit(f"Cannot write the tables under {tables}: {error}")
        raise typer.Exit(1) from error

    # Step 4: the summaries
    emit(
        "accuracy is scored from the host build's errors; Spike only counts instructions"
    )
    if accuracy:
        print_mean_vus_pr(emit, accuracy)
        print_python_check(emit, accuracy)
    if instruction_rows:
        print_instructions(emit, instruction_rows)
    if against:
        print_against(emit, against, accuracy, instruction_rows)

    # Step 5: what can't be used yet
    differing = [row for row in instruction_rows if row.identical is False]
    for row in differing:
        problems.append(
            f"{row.series} window {row.window}: host and Spike errors differ"
        )
    missing = {
        stem: found
        for stem, found in progress.items()
        if not found.accuracy.complete or not found.instructions.complete
    }
    for text in problems:
        emit(f"problem: {text}")
    if missing:
        emit(f"{len(missing)} of {len(progress)} series still lack units:")
        for stem, found in missing.items():
            emit(
                f"  {stem}: accuracy {found.accuracy} units, "
                f"instructions {found.instructions} units"
            )
    emit(
        f"scored {len(scored)} scopes and {len(instruction_rows)} counted windows into {tables}"
    )
    return Collected(
        accuracy_table=tables / ACCURACY_TABLE,
        instruction_table=tables / INSTRUCTIONS_TABLE,
        missing=len(missing),
        problems=len(problems),
    )


def collect(
    label: Annotated[str, typer.Option(help="The label to collect.")],
    against: Annotated[
        str | None,
        typer.Option(
            help="Another collected label to compare with: adds each series' difference in accuracy and the ratio of instructions."
        ),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
) -> None:
    """Score a label's finished units and write its accuracy and instruction tables, skipping what is missing.

    Parameters
    ----------
    label : str
        The label to collect.
    against : str or None
        The label to compare with.
    data : Path or None
        The folder with the data.

    Raises
    ------
    typer.Exit
        With code 1 when the label doesn't exist, something needed can't be read or written, or a scope can't be scored or a window differs; the tables of what is complete are still written. Units still missing are listed and don't fail it.
    """

    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    collected = collect_label(label, against, data, typer.echo)
    if collected.problems:
        raise typer.Exit(1)
