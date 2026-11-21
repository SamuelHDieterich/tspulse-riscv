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
from score.spike_check import HELD_GROUP, SERIES_TABLE, WORKING_GROUP, Group


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
    HELD : str
        Every window of each held-back series, kept out of every choice made on the working series.
    """

    SCREEN = "screen"
    WHOLE_PYTHON = "whole-python"
    HELD = "held"


@dataclass(frozen=True)
class Series:
    """One series, with what the scopes and units need from the series tables.

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
        Its screen span's windows `[start, stop)`, None for a multivariate or held-back series, which has none.
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


def scopes_group(scopes: list[Scope]) -> Group:
    """The series group a set of scopes runs on.

    Parameters
    ----------
    scopes : list[Scope]
        The scopes.

    Returns
    -------
    Group
        `HELD_GROUP` when the scopes are `HELD` alone, else `WORKING_GROUP`.

    Raises
    ------
    typer.BadParameter
        When `HELD` is mixed with a working scope, since a label holds one group's series.
    """

    if Scope.HELD not in scopes:
        return WORKING_GROUP
    if any(one != Scope.HELD for one in scopes):
        raise typer.BadParameter(
            f"{Scope.HELD} runs the held-back series and can't be mixed with {Scope.SCREEN} or {Scope.WHOLE_PYTHON}, which run the working ones: use a label for each.",
            param_hint="--scope",
        )
    return HELD_GROUP


def read_series(top: Path, group: Group = WORKING_GROUP) -> list[Series]:
    """List one group's series in `benchmark/series.csv` order.

    Parameters
    ----------
    top : Path
        The worktree's top level.
    group : Group
        The group to list, `WORKING_GROUP` or `HELD_GROUP`.

    Returns
    -------
    list[Series]
        Every series of the group. The Python baseline's series are all working series.

    Raises
    ------
    typer.Exit
        With code 1 when a table can't be read.
    """

    python_runs = {row["file_name"] for row in _read_table(top / PYTHON_RUNS_TABLE)}
    series = []
    for row in _read_table(top / SERIES_TABLE):
        if row["group"] != group:
            continue

        # A row span `[screen_start, screen_stop)` holds the windows `[screen_start, screen_stop − 512)`. Multivariate and held-back rows leave the span empty
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
    if scope == Scope.HELD:
        start, stop = 0, series.windows
    elif scope == Scope.SCREEN:
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
