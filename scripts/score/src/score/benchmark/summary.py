"""The summaries `collect` prints: mean VUS-PR, the check against Python, instructions per window, and the differences from another label."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import statistics
from collections.abc import Callable, Iterable

# Internal
from score.benchmark.scope import Scope
from score.benchmark.tables import AccuracyRow, InstructionRow
from score.kind import Kind
from score.series_check import VUS_TOLERANCE
from score.unit import SCORES


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The metric the summaries are about
SUMMARY_METRIC = "VUS-PR"

# The width of the first column and of each value column in a printed table
LABEL_WIDTH = 10
VALUE_WIDTH = 16

# A fraction as a percentage
PERCENT = 100


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _table(
    emit: Callable[[str], None], header: list[str], rows: list[list[str]]
) -> None:
    """Print a table of text cells with aligned columns.

    Parameters
    ----------
    emit : Callable[[str], None]
        Prints one line.
    header : list[str]
        The first cell is the row label, the rest name the value columns.
    rows : list[list[str]]
        The rows, each as long as the header.
    """

    for cells in [header, *rows]:
        first, *rest = cells
        emit(
            f"  {first:<{LABEL_WIDTH}}"
            + "".join(f"{cell:>{VALUE_WIDTH}}" for cell in rest)
        )


def _vus_pr(rows: Iterable[AccuracyRow], scope: str, head: str) -> list[AccuracyRow]:
    """Pick one scope's and head's VUS-PR rows.

    Parameters
    ----------
    rows : Iterable[AccuracyRow]
        The accuracy rows.
    scope : str
        The scope.
    head : str
        The score.

    Returns
    -------
    list[AccuracyRow]
        The matching rows, one per series.
    """

    return [
        row
        for row in rows
        if row.scope == scope and row.head == head and row.metric == SUMMARY_METRIC
    ]


def print_mean_vus_pr(emit: Callable[[str], None], rows: list[AccuracyRow]) -> None:
    """Print each scope's mean VUS-PR per score, over all series and per source.

    Parameters
    ----------
    emit : Callable[[str], None]
        Prints one line.
    rows : list[AccuracyRow]
        The accuracy rows.
    """

    for scope in sorted({row.scope for row in rows}):
        series = {row.file_name for row in rows if row.scope == scope}
        sources = sorted({row.source for row in rows if row.scope == scope})
        emit(
            f"Mean {SUMMARY_METRIC}, scope {scope}, host-scored, {len(series)} series:"
        )
        header = ["head", "all"] + sources
        table = []
        for head in SCORES:
            chosen = _vus_pr(rows, scope, head)
            cells = [
                f"{statistics.fmean(row.value for row in chosen):.4f} ({len(chosen)})"
            ]
            for source in sources:
                part = [row for row in chosen if row.source == source]
                cells.append(
                    f"{statistics.fmean(row.value for row in part):.4f} ({len(part)})"
                )
            table.append([head, *cells])
        _table(emit, header, table)


def print_python_check(emit: Callable[[str], None], rows: list[AccuracyRow]) -> None:
    """Print the largest |C − Python| VUS-PR over the `whole-python` rows, flagged over the tolerance.

    Parameters
    ----------
    emit : Callable[[str], None]
        Prints one line.
    rows : list[AccuracyRow]
        The accuracy rows.
    """

    pairs = [
        (abs(row.value - row.python_value), row)
        for row in rows
        if row.scope == Scope.WHOLE_PYTHON
        and row.metric == SUMMARY_METRIC
        and row.python_value is not None
    ]
    if not pairs:
        return
    worst, row = max(pairs, key=lambda pair: pair[0])
    over = sum(gap > VUS_TOLERANCE for gap, _ in pairs)
    verdict = (
        f"OVER {VUS_TOLERANCE:g} in {over} rows"
        if over
        else f"within {VUS_TOLERANCE:g}"
    )
    emit(
        f"Largest |C - Python| {SUMMARY_METRIC}, whole-python, {len(pairs)} rows: "
        f"{worst:.6f} ({row.file_name}, {row.head}), {verdict}"
    )


def print_instructions(emit: Callable[[str], None], rows: list[InstructionRow]) -> None:
    """Print instructions per channel-window for each archive, and per window for each multivariate series.

    Parameters
    ----------
    emit : Callable[[str], None]
        Prints one line.
    rows : list[InstructionRow]
        The instruction rows.
    """

    for kind in Kind:
        chosen = [row.per_channel for row in rows if row.kind == kind]
        if not chosen:
            continue
        mean = statistics.fmean(chosen)
        spread = (max(chosen) - min(chosen)) / mean * PERCENT
        emit(
            f"Instructions per channel-window, {kind}, {len(chosen)} windows: "
            f"mean {mean:,.0f}, min {min(chosen):,.0f}, max {max(chosen):,.0f}, spread {spread:.2f} %"
        )
    for row in rows:
        if row.kind == Kind.M:
            emit(
                f"  {row.series} window {row.window}: {row.program_instructions:,} instructions, {row.channels} channels"
            )


def print_against(
    emit: Callable[[str], None],
    against: str,
    accuracy: list[AccuracyRow],
    instructions: list[InstructionRow],
) -> None:
    """Print the mean and worst per-series differences from another label.

    Parameters
    ----------
    emit : Callable[[str], None]
        Prints one line.
    against : str
        The other label.
    accuracy : list[AccuracyRow]
        The accuracy rows, with the `--against` columns.
    instructions : list[InstructionRow]
        The instruction rows, with the `--against` columns.
    """

    for scope in sorted({row.scope for row in accuracy}):
        for head in SCORES:
            chosen = [
                row
                for row in _vus_pr(accuracy, scope, head)
                if row.difference is not None
            ]
            if not chosen:
                continue
            worst = max(chosen, key=lambda row: abs(row.difference or 0.0))
            emit(
                f"{SUMMARY_METRIC} against {against}, {scope}, {head}, {len(chosen)} series: "
                f"mean difference {statistics.fmean(row.difference or 0.0 for row in chosen):+.6f}, "
                f"worst {worst.difference:+.6f} ({worst.file_name})"
            )
    compared = [row for row in instructions if row.ratio is not None]
    if compared:
        worst = max(compared, key=lambda row: abs((row.ratio or 1.0) - 1.0))
        emit(
            f"Instructions against {against}, {len(compared)} windows: "
            f"mean ratio {statistics.fmean(row.ratio or 1.0 for row in compared):.6f}, "
            f"worst {worst.ratio:.6f} ({worst.series} window {worst.window})"
        )
