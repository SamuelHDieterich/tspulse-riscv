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
            help="The folder for weights.c, weights_table.c, weights.tdump and their int8 counterparts weights_int8.c, weights_int8_table.c and weights_int8.tdump."
        ),
    ],
) -> None:
    """Write the C source and the reference dump of the weights, once as float32 and once with int8 mixer weights."""

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
