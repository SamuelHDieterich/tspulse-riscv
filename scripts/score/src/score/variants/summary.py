"""A label's one-row summary: what the variant is, whether it is exact, and what it costs and scores."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import math
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import get_args

# Internal
from score.benchmark.scope import Scope, Series, read_series
from score.benchmark.tables import ACCURACY_TABLE, INSTRUCTIONS_TABLE
from score.benchmark.units import _counted_windows
from score.build_record import (
    BASELINE_NAME,
    BUILD_FILE,
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
from score.variants.building import ResolvedVariant
from score.variants.exactness import Exactness, measure_exactness
from score.variants.formula import evaluate_formula
from score.variants.listing import VariantError
from score.variants.state import (
    COUNT_WINDOWS,
    SUMMARY_FILE,
    label_folder,
    missing_stages,
    read_reference_windows,
    tables_folder,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The heads whose VUS-PR the summary keeps, and the scope and metric it reads them in
VUS_HEADS = ("time", "fft", "forecast", "ensemble")
VUS_SCOPE = Scope.SCREEN
VUS_METRIC = "VUS-PR"

# The status of a variant that has no summary yet, and of one with nothing missing
STATUS_NOT_RUN = "not run"
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

# The summary's columns, in the order of the merged table
COLUMNS = [
    "name",
    "settings",
    "build_commit",
    "build_tree",
    "build_dirty",
    "status",
    "verdict",
    "windows_compared",
    "windows_differing",
    "reference_pass",
    "reference_fail",
    "instructions_u_mean",
    "instructions_m_mean",
    "instructions_ratio",
    "cycles_low",
    "cycles_high",
    "weights_bytes",
    "text_bytes",
    "static_bytes",
    "stack_bytes",
    "heap_bytes_per_channel",
    "traffic_bytes",
    *[
        f"vus_pr_{head}{suffix}"
        for head in VUS_HEADS
        for suffix in ("", "_diff", "_worst")
    ],
]


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Verdict(StrEnum):
    """The verdicts a summary gives.

    Attributes
    ----------
    IDENTICAL : str
        Every compared window matches the baseline's.
    WITHIN : str
        Not identical, but every reference window passed.
    BEYOND : str
        Some reference window failed.
    REFERENCE : str
        The baseline itself, which the others are measured against.
    """

    IDENTICAL = "identical"
    WITHIN = "within-tolerance"
    BEYOND = "beyond-tolerance"
    REFERENCE = "reference"


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


def _mean(values: list[float]) -> str:
    """Format the mean of some numbers.

    Parameters
    ----------
    values : list[float]
        The numbers.

    Returns
    -------
    str
        Their mean in full precision, or an empty cell when there are none.
    """

    return "" if not values else repr(fmean(values))


def _instruction_cells(
    tables: Path, counted: set[tuple[str, str]] | None = None
) -> dict[str, str]:
    """Take the instruction means from a label's instruction table.

    Parameters
    ----------
    tables : Path
        The label's tracked results folder.
    counted : set[tuple[str, str]] or None
        The `(series, window)` pairs, as the table spells them, to average over, or None for every row.

    Returns
    -------
    dict[str, str]
        `instructions_u_mean`, `instructions_m_mean` and `instructions_ratio`, the last empty for a table without a `ratio` column.
    """

    rows = _read_table(tables / INSTRUCTIONS_TABLE)
    if counted is not None:
        rows = [
            row for row in rows if (row[SERIES_COLUMN], row[WINDOW_COLUMN]) in counted
        ]
    of_u = [row for row in rows if row[KIND_COLUMN] == Kind.U]
    of_m = [row for row in rows if row[KIND_COLUMN] == Kind.M]
    return {
        "instructions_u_mean": _mean(_numbers(of_u, PER_CHANNEL_COLUMN)),
        "instructions_m_mean": _mean(_numbers(of_m, PER_CHANNEL_COLUMN)),
        "instructions_ratio": _mean(_numbers(of_u, RATIO_COLUMN)),
    }


def _vus_cells(tables: Path) -> dict[str, str]:
    """Take each head's screen VUS-PR from a label's accuracy table.

    Parameters
    ----------
    tables : Path
        The label's tracked results folder.

    Returns
    -------
    dict[str, str]
        Per head, the mean `value` over series, the mean `difference` from the baseline and the most negative one. The difference cells are empty for a table without `--against` columns, and any cell is empty when no series has a VUS.
    """

    rows = _read_table(tables / ACCURACY_TABLE)
    cells = {}
    for head in VUS_HEADS:
        of_head = [
            row
            for row in rows
            if row[SCOPE_COLUMN] == VUS_SCOPE
            and row[METRIC_COLUMN] == VUS_METRIC
            and row[HEAD_COLUMN] == head
        ]
        differences = _numbers(of_head, DIFFERENCE_COLUMN)
        cells[f"vus_pr_{head}"] = _mean(_numbers(of_head, VALUE_COLUMN))
        cells[f"vus_pr_{head}_diff"] = _mean(differences)
        # The most negative difference is the series that got worst
        cells[f"vus_pr_{head}_worst"] = repr(min(differences)) if differences else ""
    return cells


def _profile_cells(profile: Path) -> dict[str, str]:
    """Take the cycles, sizes and traffic from a label's profile tables.

    Parameters
    ----------
    profile : Path
        The label's `profile` folder in its tracked results.

    Returns
    -------
    dict[str, str]
        `cycles_low`, `cycles_high`, the memory sizes and `traffic_bytes`.

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
    return {
        "cycles_low": cycles[LOW_COLUMN],
        "cycles_high": cycles[HIGH_COLUMN],
        "weights_bytes": by_kind[WEIGHTS][0][BYTES_COLUMN],
        "text_bytes": text[BYTES_COLUMN],
        "static_bytes": str(sum(int(row[BYTES_COLUMN]) for row in by_kind[STATIC])),
        "stack_bytes": by_kind[STACK][0][BYTES_COLUMN],
        "heap_bytes_per_channel": str(
            sum(
                evaluate_formula(row[FORMULA_COLUMN], HEAP_CHANNELS)
                for row in by_kind[HEAP]
            )
        ),
        "traffic_bytes": traffic[VALUE_COLUMN],
    }


def _build_cells(profile: Path) -> dict[str, str]:
    """Take the build's commit, tree and dirty flag from the profile's `build.json`.

    Parameters
    ----------
    profile : Path
        The label's `profile` folder in its tracked results.

    Returns
    -------
    dict[str, str]
        `build_commit`, `build_tree` and `build_dirty`, all empty when there is no record, as for the baseline.

    Raises
    ------
    ValueError
        When the record can't be used as a build record.
    """

    if not (profile / BUILD_FILE).is_file():
        return {"build_commit": "", "build_tree": "", "build_dirty": ""}
    try:
        record = read_build(profile)
    except BuildRecordError as error:
        raise ValueError(str(error)) from error
    return {
        "build_commit": str(record.commit),
        "build_tree": str(record.tree),
        "build_dirty": str(record.dirty),
    }


def _row(cells: dict[str, str]) -> dict[str, str]:
    """Fill the columns a row doesn't set with empty cells.

    Parameters
    ----------
    cells : dict[str, str]
        The cells set.

    Returns
    -------
    dict[str, str]
        One cell per column of `COLUMNS`.
    """

    return {column: cells.get(column, "") for column in COLUMNS}


def write_row(path: Path, cells: dict[str, str]) -> None:
    """Write a one-row summary, a header and the row.

    Parameters
    ----------
    path : Path
        The file, whose folder is made.
    cells : dict[str, str]
        The row's cells by column.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(_row(cells))


def read_row(path: Path) -> dict[str, str] | None:
    """Read a one-row summary.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    dict[str, str] or None
        The row by column, or None when there is no file.

    Raises
    ------
    ValueError
        When the file doesn't hold exactly one row of `COLUMNS`.
    OSError
        When the file exists but can't be read.
    """

    if not path.is_file():
        return None
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        if reader.fieldnames != COLUMNS or len(rows) != 1:
            raise ValueError(f"{path} is not a one-row summary")
    return rows[0]


def summary_path(top: Path, label: str) -> Path:
    """Where a label's summary sits.

    Parameters
    ----------
    top : Path
        The git top level.
    label : str
        The label.

    Returns
    -------
    Path
        `summary.csv` in the label's tracked results folder.
    """

    return tables_folder(top, label) / SUMMARY_FILE


def settings_text(variant: ResolvedVariant) -> str:
    """Spell a variant's settings the way the table does.

    Parameters
    ----------
    variant : ResolvedVariant
        The list entry and its name.

    Returns
    -------
    str
        `A=b B=c`, in the list's order.
    """

    return " ".join(f"{name}={value}" for name, value in variant.entry.settings.items())


def baseline_cells(top: Path) -> dict[str, str]:
    """Compute the baseline's row from its tracked tables and profile, running nothing.

    Parameters
    ----------
    top : Path
        The git top level.

    Returns
    -------
    dict[str, str]
        The baseline's cells, its instruction means over the windows the variants count. Its settings, build, status, exactness, reference and difference cells are empty, since it is measured against nothing and has no build record.

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
    return {
        "name": BASELINE_NAME,
        "verdict": Verdict.REFERENCE,
        **_instruction_cells(tables, counted),
        **_vus_cells(tables),
        **_profile_cells(tables / WORK_SUBFOLDER),
    }


def _status(
    top: Path, data: Path, label: str, working: list[Series], exactness: Exactness
) -> str:
    """Say whether the label is done, or what it lacks.

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
    str
        `done` when the done rule holds except for the summary itself, else the missing parts joined by `; `.
    """

    lacking = missing_stages(top, data, label, working)
    if exactness.left_out:
        lacking.append(f"{len(exactness.left_out)} series left out")
    if exactness.problems:
        lacking.append(f"{len(exactness.problems)} exactness problems")
    return "; ".join(lacking) if lacking else STATUS_DONE


def _verdict(exactness: Exactness, reference_fail: int, reference_pass: int) -> str:
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
    str
        `identical` when every compared window matches and no selected series was left out, else `within-tolerance` when every reference window passed, else `beyond-tolerance`.
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


def variant_cells(
    top: Path, data: Path, label: str, variant: ResolvedVariant, working: list[Series]
) -> tuple[dict[str, str], Exactness]:
    """Compute a variant's row from its own files, never from a local build.

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
    tuple[dict[str, str], Exactness]
        The variant's cells and its exactness.

    Raises
    ------
    ValueError
        When a file the row needs is missing or can't be used.
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
    cells = {
        "name": variant.name,
        "settings": settings_text(variant),
        **_build_cells(tables / WORK_SUBFOLDER),
        "status": _status(top, data, label, working, exactness),
        "verdict": _verdict(exactness, reference_fail, reference_pass),
        "windows_compared": str(exactness.windows_compared),
        "windows_differing": str(exactness.windows_differing),
        "reference_pass": str(reference_pass),
        "reference_fail": str(reference_fail),
        **_instruction_cells(tables),
        **_vus_cells(tables),
        **_profile_cells(tables / WORK_SUBFOLDER),
    }
    return cells, exactness


def write_variant_summary(
    top: Path, data: Path, label: str, variant: ResolvedVariant, working: list[Series]
) -> tuple[dict[str, str], Exactness]:
    """Write a variant's `summary.csv`.

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
    tuple[dict[str, str], Exactness]
        The row written and the exactness it holds, whose problems and left-out series the caller reports.

    Raises
    ------
    VariantError
        When a file the row needs is missing or can't be read.
    """

    try:
        cells, exactness = variant_cells(top, data, label, variant, working)
        write_row(summary_path(top, label), cells)
    except (OSError, ValueError, KeyError) as error:
        raise VariantError(f"cannot write {label}'s summary: {error}") from error
    return cells, exactness
