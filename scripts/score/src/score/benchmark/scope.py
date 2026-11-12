"""Which windows of which series a scope means.

Every range is half-open, like the program's `--windows START:STOP`.
Window `w` covers rows `w..w+512`, so a row span `[a, b)` holds the windows `[a, b − 512)`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# External
import typer

# Internal
from score.capture import CONTEXT
from score.kind import Kind
from score.spike_check import SERIES_TABLE, WORKING_GROUP


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The series the Python baseline ran whole, under the git top level
PYTHON_RUNS_TABLE = Path("benchmark/baseline/python-runs.csv")


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Scope(StrEnum):
    """A set of windows the accuracy stage runs.

    Attributes
    ----------
    SCREEN : str
        Each univariate working series' screen span, the windows `[screen_start, screen_stop − 512)`, which variants are compared on.
    WHOLE_PYTHON : str
        Every window of each series the Python baseline ran whole, so the C sits beside Python and IBM on the same series.
    """

    SCREEN = "screen"
    WHOLE_PYTHON = "whole-python"


@dataclass(frozen=True)
class Series:
    """One working series, with what the scopes and units need from the series tables.

    Attributes
    ----------
    stem : str
        The CSV's name without `.csv`.
    kind : Kind
        The archive it comes from.
    source : str
        The original dataset, from `benchmark/series.csv`.
    channels : int
        How many channels it has.
    windows : int
        How many windows it has.
    screen : tuple[int, int] or None
        Its screen span's windows `[start, stop)`, None for a multivariate series, which has none.
    python_run : bool
        Whether the Python baseline ran it whole.
    """

    stem: str
    kind: Kind
    source: str
    channels: int
    windows: int
    screen: tuple[int, int] | None
    python_run: bool


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _read_table(path: Path) -> list[dict[str, str]]:
    """Read a CSV table into one dictionary per row.

    Parameters
    ----------
    path : Path
        The table.

    Returns
    -------
    list[dict[str, str]]
        Its rows.

    Raises
    ------
    typer.Exit
        With code 1 when the table can't be read.
    """

    try:
        with path.open(newline="") as file:
            return list(csv.DictReader(file))
    except OSError as error:
        typer.echo(f"Cannot read {path}: {error}", err=True)
        raise typer.Exit(1) from error


def read_series(top: Path) -> list[Series]:
    """List the working series in `benchmark/series.csv` order.

    Parameters
    ----------
    top : Path
        The worktree's top level.

    Returns
    -------
    list[Series]
        Every working series. The Python baseline's series are all working series.

    Raises
    ------
    typer.Exit
        With code 1 when a table can't be read.
    """

    python_runs = {row["file_name"] for row in _read_table(top / PYTHON_RUNS_TABLE)}
    series = []
    for row in _read_table(top / SERIES_TABLE):
        if row["group"] != WORKING_GROUP:
            continue

        # A row span `[screen_start, screen_stop)` holds the windows `[screen_start, screen_stop − 512)`. Multivariate rows leave the span empty
        screen = None
        if row["screen_start"]:
            screen = (int(row["screen_start"]), int(row["screen_stop"]) - CONTEXT)
        series.append(
            Series(
                stem=Path(row["file_name"]).stem,
                kind=Kind(row["kind"]),
                source=row["source"],
                channels=int(row["channels"]),
                windows=int(row["windows"]),
                screen=screen,
                python_run=row["file_name"] in python_runs,
            )
        )
    return series


def scope_range(
    series: Series, scope: Scope, window_cap: int | None
) -> tuple[int, int] | None:
    """The windows a scope holds of a series.

    Parameters
    ----------
    series : Series
        The series.
    scope : Scope
        The scope.
    window_cap : int or None
        Keep only this many windows from the range's start, or all when None.

    Returns
    -------
    tuple[int, int] or None
        The windows `[start, stop)`, or None when the scope leaves the series out.
    """

    # Step 1: the scope's full range
    if scope == Scope.SCREEN:
        if series.screen is None:
            return None
        start, stop = series.screen
    elif series.python_run:
        start, stop = 0, series.windows
    else:
        return None

    # Step 2: the cap, counted from the range's own start
    if window_cap is not None:
        stop = min(stop, start + window_cap)
    return start, stop
