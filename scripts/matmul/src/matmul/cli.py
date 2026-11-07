"""The `spike-matmul` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import socket
import subprocess
import sys
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from matmul.models import Failure, Result
from matmul.run import (
    HOST,
    SETUPS,
    group_passes,
    load_results,
    projection,
    run_unit,
    write_projection,
    write_results,
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def toplevel() -> Path:
    """The root of the git checkout the command runs in.

    Returns
    -------
    Path
        The checkout's root.

    Raises
    ------
    Failure
        When the current folder isn't in a git checkout.
    """

    process = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if process.returncode != 0:
        raise Failure("Not inside a git checkout. Pass --source and --series.")
    return Path(process.stdout.strip())


def describe(result: Result) -> str:
    """Summarize one unit's measurement on one line, without its name."""

    if result.instret == 0:
        return f"{result.wall_s:.1f} s"
    return f"{result.wall_s:.1f} s, {result.instret / result.macs:.2f} instructions per MAC"


def _main(
    source: Annotated[
        Path | None,
        typer.Option(
            help="The C folder. Defaults to experiments/matmul in this checkout."
        ),
    ] = None,
    series: Annotated[
        Path | None,
        typer.Option(
            help="The benchmark series table. Defaults to benchmark/series.csv in this checkout."
        ),
    ] = None,
    output: Annotated[Path, typer.Option(help="The folder to write into.")] = Path(
        "build/matmul"
    ),
    machine: Annotated[
        str | None,
        typer.Option(
            help="A label for this computer, so runs from several can share one folder. Defaults to the hostname."
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option(help="Rerun units that already finished.")
    ] = False,
) -> None:
    """Run one TSPulse pass's matrix multiplies under Spike, and project Spike core-hours per series group.

    The projection is a floor: it covers matrix multiplies only, without GELU, softmax, LayerNorm or the FFT.
    """

    if source is None:
        source = toplevel() / "experiments" / "matmul"
    if series is None:
        series = toplevel() / "benchmark" / "series.csv"
    if machine is None:
        machine = socket.gethostname()

    folder = output / machine
    host: Result | None = None
    for setup in (HOST, *SETUPS):
        unit = folder / setup.name
        done = unit / "result.json"
        if done.exists() and not force:
            result = Result.load(done)
            typer.echo(f"{setup.name}: skipped")
        else:
            result = run_unit(setup, source, unit, machine, host)
            typer.echo(f"{setup.name}: {describe(result)}")
        if setup is HOST:
            host = result

    results = load_results(folder)
    rows = projection(results, group_passes(series))
    write_results(folder, machine, results)
    write_projection(folder, machine, rows)

    typer.echo("\nProjected Spike core-hours, a floor covering matrix multiplies only:")
    for setup, group, _, _, hours in rows:
        typer.echo(f"{setup:<12} {group:<10} {hours:>10.1f} h")
    typer.echo(str(folder))


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `spike-matmul`, turning a `Failure` into its message and exit code 1."""

    try:
        typer.run(_main)
    except Failure as error:
        typer.echo(str(error), err=True)
        sys.exit(1)
