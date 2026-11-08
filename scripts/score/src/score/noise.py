"""How far a dump may stray from the reference: PyTorch's own float32 error, pooled per step.

A step is a record name without its copy prefix, such as `backbone.encoder_block.mixers.4.patch_mixer.gating_block`.
Its ratio is the largest `max |float32 − float64| / max |float32|` over every copy of every captured window.
A record passes when `max |dump − reference| ≤ max(FACTOR × ratio, FLOOR) × max |reference|`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import re
from collections.abc import Iterable
from pathlib import Path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Fixed on purpose: loosening them until a port passes would hide real errors.
# A port as accurate as PyTorch float32 is about one noise from float64, as is the reference, so up to two noises from the reference. 3 leaves a margin.
FACTOR = 3.0

# Where PyTorch's own error is tiny or zero, the allowed difference never drops below this share of the step's largest value
FLOOR = 1e-6

COPY_PREFIX = re.compile(r"^(plain|hidden\d\d)/")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def step(name: str) -> str:
    """Strip the copy prefix from a record name, leaving its step.

    Parameters
    ----------
    name : str
        A record name, such as `hidden03/backbone.patching#2`.

    Returns
    -------
    str
        The name without `plain/` or `hiddenNN/`. `input.*` and `error.*` names have none and come back as they are.
    """

    return COPY_PREFIX.sub("", name)


def allowed(ratio: float, ref_max: float) -> float:
    """Compute the largest absolute difference a record may have from its reference.

    Parameters
    ----------
    ratio : float
        The step's pooled `noise / ref_max`, from `pool`.
    ref_max : float
        The record's own largest absolute reference value.

    Returns
    -------
    float
        `max(FACTOR × ratio, FLOOR) × ref_max`.
    """

    return max(FACTOR * ratio, FLOOR) * ref_max


def _write_atomic(path: Path, lines: Iterable[str]) -> None:
    """Write lines to a temporary file, then rename it over `path`, so a crash never leaves a half-written table.

    Parameters
    ----------
    path : Path
        The file to write.
    lines : Iterable[str]
        The lines, without their newlines.
    """

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(f"{line}\n" for line in lines))
    os.replace(temporary, path)


def write_window(path: Path, rows: Iterable[tuple[str, float, float]]) -> None:
    """Write one window's `name,noise,ref_max` table.

    Parameters
    ----------
    path : Path
        The window's `.noise.csv`.
    rows : Iterable[tuple[str, float, float]]
        Per record, its name, `max |float32 − float64|` and `max |float32|`, as `capture.capture` returns them.

    Notes
    -----
    The floats are written with `repr`, which keeps every bit, so two runs give byte-identical files.
    """

    _write_atomic(
        path,
        [
            "name,noise,ref_max",
            *(f"{name},{noise!r},{ref_max!r}" for name, noise, ref_max in rows),
        ],
    )


def read_window(path: Path) -> list[tuple[str, float, float]]:
    """Read one window's `name,noise,ref_max` table.

    Parameters
    ----------
    path : Path
        The window's `.noise.csv`.

    Returns
    -------
    list[tuple[str, float, float]]
        Per record, its name, noise and `ref_max`, in file order.
    """

    rows = []
    for line in path.read_text().splitlines()[1:]:
        name, noise, ref_max = line.split(",")
        rows.append((name, float(noise), float(ref_max)))
    return rows


def pool(windows: Iterable[list[tuple[str, float, float]]]) -> dict[str, float]:
    """Pool every window's noise into one ratio per step.

    Parameters
    ----------
    windows : Iterable[list[tuple[str, float, float]]]
        Each window's rows, as `read_window` returns them.

    Returns
    -------
    dict[str, float]
        The largest `noise / ref_max` per step across windows and copies, in first-seen order. A record whose `ref_max` is 0 counts as 0.
    """

    ratios: dict[str, float] = {}
    for rows in windows:
        for name, noise, ref_max in rows:
            ratio = noise / ref_max if ref_max else 0.0
            key = step(name)
            ratios[key] = max(ratios.get(key, 0.0), ratio)
    return ratios


def write_pooled(path: Path, ratios: dict[str, float]) -> None:
    """Write the pooled `step,ratio` table.

    Parameters
    ----------
    path : Path
        The pooled `noise.csv`.
    ratios : dict[str, float]
        The ratio per step, as `pool` returns it.
    """

    _write_atomic(
        path, ["step,ratio", *(f"{name},{ratio!r}" for name, ratio in ratios.items())]
    )


def read_pooled(path: Path) -> dict[str, float]:
    """Read the pooled `step,ratio` table.

    Parameters
    ----------
    path : Path
        The pooled `noise.csv`.

    Returns
    -------
    dict[str, float]
        The ratio per step, in file order.
    """

    ratios = {}
    for line in path.read_text().splitlines()[1:]:
        name, ratio = line.split(",")
        ratios[name] = float(ratio)
    return ratios
