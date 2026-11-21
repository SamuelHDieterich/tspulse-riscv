"""A label's summary: what the variant is, whether it is exact, and what it costs and scores."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import math
from pathlib import Path
from statistics import fmean
from typing import get_args

# Internal
from score.benchmark.results import read_identity
from score.benchmark.scope import Scope, Series, read_series
from score.benchmark.tables import ACCURACY_TABLE, INSTRUCTIONS_TABLE
from score.benchmark.units import _counted_windows
from score.build_record import (
    BASELINE_NAME,
    BUILD_FILE,
    DEFAULT_STRIDE,
    BuildRecordError,
    read_build,
)
from score.kind import Kind
from score.profile.cli import WORK_SUBFOLDER
from score.profile.memory import TEXT_SECTION, MemoryKind
from score.profile.tables import (
    CORE_CLASS,
    CYCLES,
    CYCLES_TABLE,
    MEAN_WINDOW,
    MEMORY_TABLE,
    TRAFFIC_TABLE,
)
from score.spike_check import WORKING_GROUP
from score.variants.building import ResolvedVariant
from score.variants.exactness import Exactness, measure_exactness
from score.variants.formula import evaluate_formula
from score.variants.listing import VariantError
from score.variants.state import (
    COUNT_WINDOWS,
    label_folder,
    label_stride,
    missing_stages,
    read_reference_windows,
    stack_name,
    tables_folder,
)
from score.variants.summary_record import (
    BuildSource,
    Cycles,
    ExactnessCounts,
    HeadVus,
    Instructions,
    MemoryBytes,
    ReferenceCounts,
    Summary,
    Verdict,
    summary_path,
    write_summary,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The heads whose VUS-PR the summary keeps, the one the short form shows, and the scope and metric it reads them in
ENSEMBLE_HEAD = "ensemble"
VUS_HEADS = ("time", "fft", "forecast", ENSEMBLE_HEAD)
VUS_SCOPE = Scope.SCREEN
VUS_METRIC = "VUS-PR"

# What a variant with no summary yet lacks
MISSING_NOT_RUN = "not run"

# What the terminal and the log say of a label that lacks nothing
STATUS_DONE = "done"

# The channel count the heap formulas are evaluated at
HEAP_CHANNELS = 1

# The `traffic.csv` measure the summary keeps
TRAFFIC_MEASURE = "total_bytes"

# The memory kinds the summary reads
WEIGHTS: MemoryKind = "weights"
CODE: MemoryKind = "code"
STATIC: MemoryKind = "static"
STACK: MemoryKind = "stack"
HEAP: MemoryKind = "heap"

# The columns of the tables the summary reads
KIND_COLUMN = "kind"
SERIES_COLUMN = "series"
WINDOW_COLUMN = "window"
PER_CHANNEL_COLUMN = "per_channel"
RATIO_COLUMN = "ratio"
VALUE_COLUMN = "value"
DIFFERENCE_COLUMN = "difference"
SCOPE_COLUMN = "scope"
METRIC_COLUMN = "metric"
HEAD_COLUMN = "head"
CLASS_COLUMN = "class"
QUANTITY_COLUMN = "quantity"
MEASURE_COLUMN = "measure"
ITEM_COLUMN = "item"
BYTES_COLUMN = "bytes"
FORMULA_COLUMN = "formula"
LOW_COLUMN = "low"
HIGH_COLUMN = "high"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _read_table(path: Path) -> list[dict[str, str]]:
    """Read a CSV table into one dict per row.

    Parameters
    ----------
    path : Path
        The table.

    Returns
    -------
    list[dict[str, str]]
        The rows by header name.

    Raises
    ------
    OSError
        When the table can't be read.
    """

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _numbers(rows: list[dict[str, str]], column: str) -> list[float]:
    """Collect a column's numbers, leaving out empty and NaN cells.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The rows.
    column : str
        The column, which a table written without `--against` may lack.

    Returns
    -------
    list[float]
        The numbers, in row order.
    """

    values = []
    for row in rows:
        text = row.get(column) or ""
        if text and not math.isnan(float(text)):
            values.append(float(text))
    return values


def _mean(values: list[float]) -> float | None:
    """Take the mean of some numbers.

    Parameters
    ----------
    values : list[float]
        The numbers.

    Returns
    -------
    float or None
        Their mean, or None when there are none.
    """

    return fmean(values) if values else None


def _instructions(
    tables: Path, counted: set[tuple[str, str]] | None = None
) -> Instructions:
    """Take the instruction means from a label's instruction table.

    Parameters
    ----------
    tables : Path
        The label's tracked results folder.
    counted : set[tuple[str, str]] or None
        The `(series, window)` pairs, as the table spells them, to average over, or None for every row.

    Returns
    -------
    Instructions
        The means over the univariate and multivariate rows, and the mean ratio, None for a table without a `ratio` column.
    """

    rows = _read_table(tables / INSTRUCTIONS_TABLE)
    if counted is not None:
        rows = [
            row for row in rows if (row[SERIES_COLUMN], row[WINDOW_COLUMN]) in counted
        ]
    of_u = [row for row in rows if row[KIND_COLUMN] == Kind.U]
    of_m = [row for row in rows if row[KIND_COLUMN] == Kind.M]
    return Instructions(
        u_mean=_mean(_numbers(of_u, PER_CHANNEL_COLUMN)),
        m_mean=_mean(_numbers(of_m, PER_CHANNEL_COLUMN)),
        ratio=_mean(_numbers(of_u, RATIO_COLUMN)),
    )


def _vus(tables: Path) -> dict[str, HeadVus]:
    """Take each head's screen VUS-PR from a label's accuracy table.

    Parameters
    ----------
    tables : Path
        The label's tracked results folder.

    Returns
    -------
    dict[str, HeadVus]
        Per head, the mean `value` over series, the mean `difference` from the baseline and the most negative one. The differences are None for a table without `--against` columns, and any field is None when no series has a VUS.
    """

    rows = _read_table(tables / ACCURACY_TABLE)
    vus = {}
    for head in VUS_HEADS:
        of_head = [
            row
            for row in rows
            if row[SCOPE_COLUMN] == VUS_SCOPE
            and row[METRIC_COLUMN] == VUS_METRIC
            and row[HEAD_COLUMN] == head
        ]
        differences = _numbers(of_head, DIFFERENCE_COLUMN)
        vus[head] = HeadVus(
            value=_mean(_numbers(of_head, VALUE_COLUMN)),
            difference=_mean(differences),
            # The most negative difference is the series that got worst
            worst=min(differences) if differences else None,
        )
    return vus


def _profile(profile: Path) -> tuple[Cycles, MemoryBytes, int]:
    """Take the cycles, sizes and traffic from a label's profile tables.

    Parameters
    ----------
    profile : Path
        The label's `profile` folder in its tracked results.

    Returns
    -------
    tuple[Cycles, MemoryBytes, int]
        The cycle range, the memory sizes and the traffic in bytes.

    Raises
    ------
    ValueError
        When a row the summary needs is missing or a heap formula can't be evaluated.
    OSError
        When a table can't be read.
    """

    # Step 1: the cycle range of the mean window, and the mean traffic
    cycles = next(
        (
            row
            for row in _read_table(profile / CYCLES_TABLE)
            if row[WINDOW_COLUMN] == MEAN_WINDOW
            and row[CLASS_COLUMN] == CORE_CLASS
            and row[QUANTITY_COLUMN] == CYCLES
        ),
        None,
    )
    traffic = next(
        (
            row
            for row in _read_table(profile / TRAFFIC_TABLE)
            if row[WINDOW_COLUMN] == MEAN_WINDOW
            and row[MEASURE_COLUMN] == TRAFFIC_MEASURE
        ),
        None,
    )
    if cycles is None or traffic is None:
        raise ValueError(f"{profile} has no mean cycles row or no mean traffic row")

    # Step 2: the sizes by kind. The heap's formulas are in bytes per channel count
    memory = _read_table(profile / MEMORY_TABLE)
    by_kind = {
        kind: [row for row in memory if row[KIND_COLUMN] == kind]
        for kind in get_args(MemoryKind)
    }
    text = next(
        (row for row in by_kind[CODE] if row[ITEM_COLUMN] == TEXT_SECTION), None
    )
    if text is None or not by_kind[WEIGHTS] or not by_kind[STACK]:
        raise ValueError(f"{profile}/{MEMORY_TABLE} lacks weights, .text or stack")
    return (
        Cycles(low=int(cycles[LOW_COLUMN]), high=int(cycles[HIGH_COLUMN])),
        MemoryBytes(
            weights=int(by_kind[WEIGHTS][0][BYTES_COLUMN]),
            text=int(text[BYTES_COLUMN]),
            static=sum(int(row[BYTES_COLUMN]) for row in by_kind[STATIC]),
            stack=int(by_kind[STACK][0][BYTES_COLUMN]),
            heap_per_channel=sum(
                evaluate_formula(row[FORMULA_COLUMN], HEAP_CHANNELS)
                for row in by_kind[HEAP]
            ),
        ),
        int(traffic[VALUE_COLUMN]),
    )


def _build(folder: Path) -> BuildSource | None:
    """Take the build's commit, tree and dirty flag from a `build.json`.

    Parameters
    ----------
    folder : Path
        The folder holding the `build.json`: the label's `profile` folder in its tracked results, or for a variant with a stride the label's results folder itself.

    Returns
    -------
    BuildSource or None
        The build's commit, tree and dirty flag, None when there is no record, as for the baseline.

    Raises
    ------
    ValueError
        When the record can't be used as a build record.
    """

    if not (folder / BUILD_FILE).is_file():
        return None
    try:
        record = read_build(folder)
    except BuildRecordError as error:
        raise ValueError(str(error)) from error
    return BuildSource(commit=record.commit, tree=record.tree, dirty=record.dirty)


def status_text(summary: Summary) -> str:
    """Say in words whether a label is done, or what it lacks.

    Parameters
    ----------
    summary : Summary
        The label's summary.

    Returns
    -------
    str
        `done` when nothing is missing, else the missing parts joined by `; `.
    """

    return "; ".join(summary.missing) or STATUS_DONE


def not_run_summary(variant: ResolvedVariant) -> Summary:
    """Stand in for the summary of a variant that has none yet.

    Parameters
    ----------
    variant : ResolvedVariant
        The list entry and its name.

    Returns
    -------
    Summary
        Only the variant's name and settings, and `not run` as what it lacks.
    """

    return Summary(
        name=variant.name,
        settings=dict(variant.entry.settings),
        missing=[MISSING_NOT_RUN],
    )


def baseline_summary(top: Path) -> Summary:
    """Compute the baseline's summary from its tracked tables and profile, running nothing.

    Parameters
    ----------
    top : Path
        The git top level.

    Returns
    -------
    Summary
        The baseline's summary, its instruction means over the windows the variants count. Its settings, build, missing, exactness, reference, ratio and difference fields are left out, since it is measured against nothing and has no build record.

    Raises
    ------
    ValueError
        When a profile table lacks a row.
    OSError
        When a table can't be read.
    """

    tables = tables_folder(top, BASELINE_NAME)

    # The baseline counted more windows than the variants do, so only the ones they count are averaged
    counted = {
        (one.stem, str(window))
        for one in read_series(top)
        for window in _counted_windows(one, COUNT_WINDOWS)
    }
    cycles, memory, traffic = _profile(tables / WORK_SUBFOLDER)
    return Summary(
        name=BASELINE_NAME,
        verdict=Verdict.REFERENCE,
        instructions=_instructions(tables, counted),
        cycles=cycles,
        memory_bytes=memory,
        traffic_bytes=traffic,
        vus_pr=_vus(tables),
    )


def _missing(
    top: Path, data: Path, label: str, working: list[Series], exactness: Exactness
) -> list[str]:
    """Name what the label lacks.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The variant's label.
    working : list[Series]
        Every working series.
    exactness : Exactness
        The label's exactness, whose left-out series and problems count.

    Returns
    -------
    list[str]
        What the done rule lacks except for the summary itself, and the left-out series and exactness problems. Empty when the label is done.
    """

    lacking = missing_stages(top, data, label, working, WORKING_GROUP)
    if exactness.left_out:
        lacking.append(f"{len(exactness.left_out)} series left out")
    if exactness.problems:
        lacking.append(f"{len(exactness.problems)} exactness problems")
    return lacking


def _verdict(exactness: Exactness, reference_fail: int, reference_pass: int) -> Verdict:
    """Judge a variant from its exactness and its reference windows.

    Parameters
    ----------
    exactness : Exactness
        The comparison with the baseline's errors.
    reference_fail : int
        How many reference windows failed.
    reference_pass : int
        How many passed.

    Returns
    -------
    Verdict
        `IDENTICAL` when every compared window matches and no selected series was left out, else `WITHIN` when every reference window passed, else `BEYOND`.
    """

    exact = (
        exactness.windows_compared > 0
        and not exactness.windows_differing
        and not exactness.left_out
        and not exactness.problems
    )
    if exact:
        return Verdict.IDENTICAL
    if reference_pass and not reference_fail:
        return Verdict.WITHIN
    return Verdict.BEYOND


def variant_summary(
    top: Path, data: Path, label: str, variant: ResolvedVariant, working: list[Series]
) -> tuple[Summary, Exactness]:
    """Compute a variant's summary from its own files, never from a local build.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The variant's label.
    variant : ResolvedVariant
        The list entry and its name.
    working : list[Series]
        Every working series.

    Returns
    -------
    tuple[Summary, Exactness]
        The variant's summary and its exactness.

    Raises
    ------
    ValueError
        When a file the summary needs is missing or can't be used.
    OSError
        When a file can't be read.
    """

    tables = tables_folder(top, label)
    windows = read_reference_windows(label_folder(data, label))
    if windows is None:
        raise ValueError(f"{label} has no reference.json")
    reference_pass = sum(window.ok for window in windows)
    reference_fail = len(windows) - reference_pass
    exactness = measure_exactness(data, label, working)

    # A variant with a stride runs its stack's model code on every N-th window, so its profile is the stack's with the per-window costs divided by N
    identity = read_identity(label_folder(data, label))
    stride = label_stride(identity)
    if stride == DEFAULT_STRIDE:
        profile_from = None
        cycles, memory, traffic = _profile(tables / WORK_SUBFOLDER)
    else:
        profile_from = stack_name(identity)
        cycles, memory, traffic = _profile(
            tables_folder(top, profile_from) / WORK_SUBFOLDER
        )
        cycles = Cycles(
            low=round(cycles.low / stride), high=round(cycles.high / stride)
        )
        traffic = round(traffic / stride)
    summary = Summary(
        name=variant.name,
        settings=dict(variant.entry.settings),
        # Without a profile of its own, the build record sits in the label's results folder
        build=_build(tables if profile_from is not None else tables / WORK_SUBFOLDER),
        missing=_missing(top, data, label, working, exactness),
        verdict=_verdict(exactness, reference_fail, reference_pass),
        exactness=ExactnessCounts(
            windows_compared=exactness.windows_compared,
            windows_differing=exactness.windows_differing,
        ),
        reference=ReferenceCounts(passed=reference_pass, failed=reference_fail),
        instructions=_instructions(tables),
        cycles=cycles,
        memory_bytes=memory,
        traffic_bytes=traffic,
        profile_from=profile_from,
        vus_pr=_vus(tables),
    )
    return summary, exactness


def write_variant_summary(
    top: Path, data: Path, label: str, variant: ResolvedVariant, working: list[Series]
) -> tuple[Summary, Exactness]:
    """Write a variant's `summary.json`.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The variant's label.
    variant : ResolvedVariant
        The list entry and its name.
    working : list[Series]
        Every working series.

    Returns
    -------
    tuple[Summary, Exactness]
        The summary written and the exactness it holds, whose problems and left-out series the caller reports.

    Raises
    ------
    VariantError
        When a file the summary needs is missing or can't be read.
    """

    try:
        summary, exactness = variant_summary(top, data, label, variant, working)
        write_summary(summary_path(top, label), summary)
    except (OSError, ValueError, KeyError) as error:
        raise VariantError(f"cannot write {label}'s summary: {error}") from error
    return summary, exactness
