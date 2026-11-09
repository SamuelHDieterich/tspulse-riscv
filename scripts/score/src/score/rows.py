"""Write `.rows` files, the series format the C port reads.

Format
------
Little-endian, with no padding: each field starts right after the previous one.
The file is the 4 bytes `ROWS`, then three `uint32`: the version (1), the channel count C and the row count N, then:

- `float64` mean, C values;
- `float64` std, C values;
- `float64` raw values in row-major order, N rows of C values.

The values stay float64 because the C port standardizes them itself, in double, as scikit-learn's `StandardScaler` does.
A float32 copy of a series sitting near 1000 with a spread of 1 would already be off by about 6e-5 before the model sees it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import struct
from pathlib import Path

# External
import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

MAGIC = b"ROWS"
VERSION = 1


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def write(path: Path, raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    """Write a series to a `.rows` file, replacing it only once complete.

    Parameters
    ----------
    path : Path
        The file to write.
    raw : np.ndarray
        (N, C), the series rows before standardizing.
    mean : np.ndarray
        (C,), the per-channel average the C port subtracts.
    std : np.ndarray
        (C,), the per-channel spread the C port divides by.

    Raises
    ------
    ValueError
        If `raw` isn't 2-dimensional, or `mean` or `std` doesn't have one value per channel.
    """

    # Step 1: refuse shapes the format can't hold, before any byte is written
    if raw.ndim != 2:
        raise ValueError(f"{path}: raw has {raw.ndim} dimensions, expected 2")
    rows, channels = raw.shape
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError(
            f"{path}: mean {mean.shape} and std {std.shape} must both be ({channels},)"
        )

    # A crash mid-write leaves only the temporary file, so a half-written series never passes for a finished one
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        # Step 2: the header. `struct.pack` turns Python numbers into raw bytes: `<` is little-endian, `I` a uint32
        file.write(MAGIC + struct.pack("<III", VERSION, channels, rows))

        # Step 3: mean, std, then the rows. `ascontiguousarray` lays a sliced array out in row-major order as little-endian float64, so `tobytes` gives exactly the bytes the format expects
        for array in (mean, std, raw):
            file.write(np.ascontiguousarray(array, dtype="<f8").tobytes())

    # Step 4: the rename is atomic, so the final name only ever points at a complete file
    os.replace(temporary, path)
