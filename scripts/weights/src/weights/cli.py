"""The `export-weights` command line."""

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
from weights.export import Failure, export


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _export(
    safetensors: Annotated[
        Path, typer.Argument(help="The tspulse-r1 model.safetensors file.")
    ],
    outdir: Annotated[
        Path,
        typer.Argument(
            help="The folder for weights.c, weights_table.c and weights.tdump."
        ),
    ],
) -> None:
    """Write the C source and the reference dump for the weights the baseline runs."""

    for name, written in export(safetensors, outdir).items():
        typer.echo(f"{name}: {'written' if written else 'unchanged'}")


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `export-weights`, turning a `Failure` into its message and exit code 1."""

    try:
        typer.run(_export)
    except Failure as error:
        typer.echo(str(error), err=True)
        sys.exit(1)
