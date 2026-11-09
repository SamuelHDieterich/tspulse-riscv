"""The `prepare-rows` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from score import rows
from score.capture import flat, load


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The datasets' folder under the data folder
DATASETS_FOLDER = "datasets"

# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _prepare_rows(
    paths: Annotated[
        list[Path],
        typer.Argument(
            metavar="[CSV] OUT",
            help="A CSV, as a path inside datasets/ or an absolute path, then the `.rows` file to write. With --flat, only the file to write.",
        ),
    ],
    flat_window: Annotated[
        bool,
        typer.Option(
            "--flat", help="Write the made-up all-zeros window instead of a CSV."
        ),
    ] = False,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
) -> None:
    """Convert a series into the `.rows` file the C port reads, with the whole-file mean and spread.

    Parameters
    ----------
    paths : list[Path]
        The CSV and the file to write, or just the file to write with `flat_window`.
    flat_window : bool
        Write 513 zeros with mean 0 and spread 1, the window `capture-reference` calls flat.
    data : Path or None
        The folder with the data, needed unless `flat_window` is set.

    Raises
    ------
    typer.Exit
        With code 1 when the paths don't match the mode, or the data folder or CSV is missing.
    """

    # Step 1: the paths for the chosen mode
    expected = 1 if flat_window else 2
    if len(paths) != expected:
        usage = "OUT" if flat_window else "CSV OUT"
        typer.echo(f"Expected {usage}, got {len(paths)} paths.", err=True)
        raise typer.Exit(1)
    out = paths[-1]

    # Step 2: the made-up window needs no data folder
    if flat_window:
        window = flat()
        rows.write(out, window.raw, window.mean, window.std)
        typer.echo(f"{out}: {window.raw.shape[0]} rows, {window.raw.shape[1]} channel")
        return

    # Step 3: the series, standardized over the whole file like the pipeline does, so the mean and spread are the ones the reference used
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    # An absolute CSV replaces the prefix, so any CSV works
    csv_path = data / DATASETS_FOLDER / paths[0]
    if not csv_path.exists():
        typer.echo(f"Not found: {csv_path}", err=True)
        raise typer.Exit(1)
    series = load(csv_path)
    rows.write(out, series.raw, series.mean, series.std)
    typer.echo(
        f"{out}: {series.raw.shape[0]} rows, {series.raw.shape[1]} channels, {series.windows} windows"
    )


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `prepare-rows`."""

    typer.run(_prepare_rows)
