"""The `score-series` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import time
from pathlib import Path
from typing import Annotated

# External
import numpy as np
import typer

# Internal
from score.pipeline import HEADS, LABEL, load_model, load_series, score


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

DEFAULT_SERIES = Path("TSB-AD-U/545_SMAP_id_15_Sensor_tr_1173_1st_2750.csv")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _score_series(
    series: Annotated[
        Path,
        typer.Option(help="A CSV, as a path inside datasets/ or an absolute path."),
    ] = DEFAULT_SERIES,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
    output: Annotated[
        Path, typer.Option(help="The folder to write the scores into.")
    ] = Path("build/reference"),
) -> None:
    """Score one series with each head and the ensemble, and write them to a CSV."""

    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)

    # An absolute series replaces the prefix, so any CSV works
    series_path = data / "datasets" / series
    model_path = data / "models" / "tspulse-r1"
    for path in (series_path, model_path):
        if not path.exists():
            typer.echo(f"Not found: {path}", err=True)
            raise typer.Exit(1)

    frame = load_series(series_path)
    channels = frame.columns.drop(LABEL)
    model = load_model(model_path, len(channels))

    # The pipeline defaults to `forecast` alone. Each head also runs by itself, to compare their accuracy in the variant study.
    # `time` and `fft` always share one model call, so their seconds don't show what dropping either would save.
    # Every call returns one score column, so each head is a separate call.
    runs = {head: (head,) for head in HEADS} | {"ensemble": HEADS}
    scores = {}
    seconds = {}
    for name, modes in runs.items():
        start = time.perf_counter()
        scores[name] = score(model, frame, modes)
        seconds[name] = time.perf_counter() - start

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{series_path.stem}.csv"
    table = frame.assign(**scores)
    table.insert(0, "index", table.index)
    table.to_csv(csv_path, index=False)

    labelled = frame.index[frame[LABEL] == 1]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    typer.echo(
        f"{series_path.name}: {len(frame)} rows, {len(channels)} channels, labelled span {labelled[0]} to {labelled[-1]}, {parameters} parameters"
    )
    for name, values in scores.items():
        typer.echo(
            f"{name}: {seconds[name]:.1f} s, peak at {np.argmax(values)}, max {values.max():.3f}"
        )
    typer.echo(csv_path)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `score-series`."""

    typer.run(_score_series)
