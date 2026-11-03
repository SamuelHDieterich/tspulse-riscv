"""The `select-series` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import sys
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from series.selection import PASSES_PER_SECOND, Failure, Row, select, write_table


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def summary(label: str, passes: list[int]) -> str:
    """Describe one group's cost on one line.

    Parameters
    ----------
    label : str
        What the group is called.
    passes : list[int]
        The passes of each of its series.

    Returns
    -------
    str
        The line.
    """

    total = sum(passes)
    hours = total / PASSES_PER_SECOND / 3600
    return f"{label}: {len(passes)} series, {total / 1e6:.1f}M passes, {hours:.1f} h"


def costs(rows: list[Row], kind: str, group: str) -> list[int]:
    """Collect the passes of one kind and group.

    Parameters
    ----------
    rows : list[Row]
        The table.
    kind : str
        `U` or `M`.
    group : str
        `working` or `held`.

    Returns
    -------
    list[int]
        The passes of each series.
    """

    return [row.passes for row in rows if row.kind == kind and row.group == group]


def _main(
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data and lists."),
    ] = None,
    output: Annotated[Path, typer.Option(help="The table to write.")] = Path(
        "benchmark/series.csv"
    ),
) -> None:
    """Choose the benchmark series and write them to a table."""

    if data is None:
        raise Failure(
            "No data folder. Pass --data, or run this inside the dev shell, from `nix develop`, which sets $TSPULSE_DATA."
        )
    if not data.is_dir():
        raise Failure(f"{data} is not a directory.")

    rows = select(data)
    write_table(rows, output)

    screen = [row.screen_passes for row in rows if row.screen_passes is not None]
    typer.echo(summary("U working", costs(rows, "U", "working")))
    typer.echo(summary("U screen", screen))
    typer.echo(summary("U held", costs(rows, "U", "held")))
    typer.echo(summary("M working", costs(rows, "M", "working")))
    typer.echo(summary("M held", costs(rows, "M", "held")))
    typer.echo(str(output))


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `select-series`, turning a `Failure` into its message and exit code 1."""

    try:
        typer.run(_main)
    except Failure as error:
        typer.echo(str(error), err=True)
        sys.exit(1)
