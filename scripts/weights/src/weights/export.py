"""Choosing the tensors the baseline runs and writing the generated files."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
from pathlib import Path

# Internal
from weights.render import render_table, render_weights
from weights.safetensors import Failure, Tensor, read_safetensors
from weights.tdump import render_tdump


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The baseline never runs this head, so its weights stay out of the program
SKIPPED_PREFIXES = ("decoder_with_head.fft_softmax_mapping.",)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def select(tensors: list[Tensor]) -> list[Tensor]:
    """Drop the tensors the baseline never uses.

    Parameters
    ----------
    tensors : list of Tensor
        Every tensor of the file.

    Returns
    -------
    list of Tensor
        The ones whose names don't start with any of `SKIPPED_PREFIXES`.

    Raises
    ------
    Failure
        When nothing was dropped, which means the file isn't the one this tool was written for.
    """

    kept = [t for t in tensors if not t.name.startswith(SKIPPED_PREFIXES)]
    if len(kept) == len(tensors):
        raise Failure(
            f"No tensor starts with {' or '.join(SKIPPED_PREFIXES)}, so this is not the expected file."
        )
    return kept


def write_if_changed(path: Path, content: bytes) -> bool:
    """Replace a file only when its content differs, so Make sees no change otherwise.

    Parameters
    ----------
    path : Path
        The file.
    content : bytes
        What it should hold.

    Returns
    -------
    bool
        Whether the file was written.

    Raises
    ------
    Failure
        When the file can't be written.
    """

    part = path.with_name(f"{path.name}.part")
    try:
        part.write_bytes(content)
        if path.exists() and path.read_bytes() == content:
            return False
        os.replace(part, path)
        return True
    except OSError as error:
        raise Failure(f"Could not write {path}: {error}") from error
    finally:
        part.unlink(missing_ok=True)


def export(safetensors: Path, outdir: Path) -> dict[str, bool]:
    """Write the three generated files for a safetensors file.

    Parameters
    ----------
    safetensors : Path
        The tspulse-r1 weights.
    outdir : Path
        Where the files go. It is created when missing.

    Returns
    -------
    dict of str to bool
        Each file's name and whether it was written, as opposed to already up to date.

    Raises
    ------
    Failure
        When the input is not the expected file or an output can't be written.
    """

    tensors = select(read_safetensors(safetensors))
    files = {
        "weights.c": render_weights(tensors).encode(),
        "weights_table.c": render_table(tensors).encode(),
        "weights.tdump": render_tdump(tensors),
    }

    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise Failure(f"Could not create {outdir}: {error}") from error
    return {
        name: write_if_changed(outdir / name, content)
        for name, content in files.items()
    }
