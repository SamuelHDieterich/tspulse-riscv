"""The `capture-reference` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import time
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from score import noise, tdump
from score.capture import capture, flat, load
from score.layout import DATASETS_FOLDER, MODEL_FOLDER, REFERENCE_FOLDER
from score.pipeline import load_model


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

YAHOO = Path("TSB-AD-U/560_YAHOO_id_10_Synthetic_tr_500_1st_893.csv")
TAO = Path("TSB-AD-M/120_TAO_id_5_Environment_tr_500_1st_3.csv")

# Series and window index. `None` is the made-up flat window.
WINDOWS: list[tuple[Path | None, int]] = [
    # The first window
    (YAHOO, 0),
    # Hides the series' only anomaly, row 893, in patch 6
    (YAHOO, 429),
    # The last window
    (YAHOO, 908),
    # The first window of a 3-channel series
    (TAO, 0),
    # A middle window, away from both edges
    (TAO, 4744),
    # The last window
    (TAO, 9487),
    # All zeros: reaches RevIN's variance floor and the FFT's +0.001 guard
    (None, 0),
]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _capture_reference(
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help="The folder to write into. Defaults to reference/ in the data folder."
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Recapture windows that already exist.")
    ] = False,
) -> None:
    """Capture every inner output of TSPulse on 7 fixed windows, and pool PyTorch's own float32 error per step."""

    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    out = out or data / REFERENCE_FOLDER

    model_path = data / MODEL_FOLDER
    for path in {model_path} | {
        data / DATASETS_FOLDER / series for series, _ in WINDOWS if series
    }:
        if not path.exists():
            typer.echo(f"Not found: {path}", err=True)
            raise typer.Exit(1)

    # One model per channel count, loaded only when a window needs it
    models = {}
    noise_paths = []
    for series, index in WINDOWS:
        folder = out / (series.stem if series else "flat")
        dump_path = folder / f"w{index}.tdump"
        noise_path = folder / f"w{index}.noise.csv"
        noise_paths.append(noise_path)
        label = f"{folder.name}/w{index}"

        if dump_path.exists() and noise_path.exists() and not force:
            typer.echo(f"{label}: exists, skipped")
            continue

        start = time.perf_counter()
        window = (
            load(data / DATASETS_FOLDER / series).window(index) if series else flat()
        )
        channels = window.values.shape[1]
        if channels not in models:
            models[channels] = load_model(model_path, channels)
        records, rows = capture(models[channels], window)

        # The dump goes first, so a crash before the noise file leaves the window to be redone
        folder.mkdir(parents=True, exist_ok=True)
        tdump.write(dump_path, records.items())
        noise.write_window(noise_path, rows)

        worst = max(drift / ref_max for _, drift, ref_max in rows if ref_max)
        typer.echo(
            f"{label}: {len(records)} records, {dump_path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - start:.1f} s, largest noise/ref_max {worst:.3g}"
        )

    missing = [path for path in noise_paths if not path.exists()]
    if missing:
        typer.echo("No pooled noise file, these windows are missing:", err=True)
        for path in missing:
            typer.echo(f"  {path}", err=True)
        raise typer.Exit(1)

    pooled_path = out / "noise.csv"
    ratios = noise.pool(noise.read_window(path) for path in noise_paths)
    noise.write_pooled(pooled_path, ratios)
    floored = sum(noise.FACTOR * ratio < noise.FLOOR for ratio in ratios.values())
    typer.echo(
        f"{pooled_path}: {len(ratios)} steps, {floored} on the {noise.FLOOR:g} floor"
    )


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `capture-reference`."""

    typer.run(_capture_reference)
