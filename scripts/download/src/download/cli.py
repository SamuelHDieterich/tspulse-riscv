"""The `download-datasets` and `download-weights` commands."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from download.fetch import (
    DATASETS,
    WEIGHTS,
    Failure,
    data_root,
    ensure,
    ensure_extracted,
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Only(StrEnum):
    """The dataset `--only` can pick, keyed as in `DATASETS`."""

    u = "u"
    m = "m"


Output = Annotated[
    Path | None,
    typer.Option(envvar="TSPULSE_DATA", help="The folder to check and fetch into."),
]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _datasets(
    only: Annotated[
        Only | None,
        typer.Option(help="Handle one dataset instead of both; the M zip is 540 MB."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace a zip that fails its hash, and repair CSVs that don't match it.",
        ),
    ] = False,
    output: Output = None,
) -> None:
    """Check the TSB-AD datasets, fetching whatever is missing."""

    root = data_root(output)
    for name in [only] if only else DATASETS:
        pin = DATASETS[name]
        downloaded = ensure(pin, root, force)
        extracted = ensure_extracted(root / pin.path, force)
        typer.echo(
            f"{Path(pin.path).stem}: zip {'downloaded' if downloaded else 'OK'}, {extracted} CSVs extracted"
        )


def _weights(
    force: Annotated[
        bool, typer.Option("--force", help="Replace a file that fails its hash.")
    ] = False,
    output: Output = None,
) -> None:
    """Check the tspulse-r1 weights, fetching whatever is missing."""

    root = data_root(output)
    for pin in WEIGHTS:
        downloaded = ensure(pin, root, force)
        typer.echo(f"{pin.path}: {'downloaded' if downloaded else 'OK'}")


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def datasets() -> None:
    """Run `download-datasets`, turning a `Failure` into its message and exit code 1."""

    try:
        typer.run(_datasets)
    except Failure as error:
        typer.echo(str(error), err=True)
        sys.exit(1)


def weights() -> None:
    """Run `download-weights`, turning a `Failure` into its message and exit code 1."""

    try:
        typer.run(_weights)
    except Failure as error:
        typer.echo(str(error), err=True)
        sys.exit(1)
