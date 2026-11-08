"""Read and write `.tdump` files, the dump format shared by the Python reference and the C port.

Format
------
Little-endian, with no padding: each field starts right after the previous one.
The file is the 4 bytes `TDMP`, a `uint32` version (1), then records until the end of the file, each laid out as:

- `uint16` name length in bytes, then the name in UTF-8, with no terminator;
- `uint8` dtype (1 float32, 2 float64), then `uint8` ndim;
- `ndim` dims, each a `uint32`;
- `prod(dims)` values in row-major order. A 0-dim record holds one value.

Names are unique, record order doesn't matter, and a file with no records is valid.

Naming
------
One file holds one window. Module outputs are named `<copy>/<module name>`:

- the copy is `plain` for the pass with nothing hidden, or `hidden00` … `hidden11` for the pass that hides patch `i`, at offsets `416 + 8i` … `423 + 8i`;
- the module name is PyTorch's dotted path, such as `backbone.encoder_block.mixers.4.patch_mixer.gating_block`;
- a module that runs again in the same pass gets `#2`, `#3` … from its second run, never `#1`: `hidden03/backbone.patching#2` is the FFT path;
- tuple parts get `.0`, `.1` …, `ModelOutput` fields get `.<field>`, and the model's own output is `<copy>/output.<field>`;
- the one complex value is stored as its real part: `plain/backbone.fft_base_component.real`;
- every module output is float32, masks as 0.0 and 1.0, with the batch dimension dropped. Outputs with no batch dimension aren't recorded.

Records outside the model have no copy prefix:

- `input.raw`: float64 (513, C), the series rows after dropping incomplete ones;
- `input.mean`, `input.std`: float64 (C,), used to standardize;
- `input.window`: float32 (513, C), the standardized rows as fed, 512 past rows and the forecast target;
- `error.time`, `error.fft`, `error.forecast`: float32 (1,), each head's error.

Notes
-----
The C port writes records as it runs, so each one carries its own header. GGUF and safetensors index every tensor at the top of the file, and `.npz` is a zip archive.
The layout follows the legacy GGML files that preceded GGUF [1]_, cut down to two dtypes and no alignment.

References
----------
.. [1] ggml, "GGUF", section "Historical State of Affairs". https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import struct
from collections.abc import Iterable
from pathlib import Path

# External
import numpy as np


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

MAGIC = b"TDMP"
VERSION = 1

# Explicit little-endian, so the file reads the same on any host
DTYPES = {1: np.dtype("<f4"), 2: np.dtype("<f8")}
CODES = {np.dtype("float32"): 1, np.dtype("float64"): 2}


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def write(path: Path, records: Iterable[tuple[str, np.ndarray]]) -> None:
    """Write records to a `.tdump` file, replacing it only once complete.

    Parameters
    ----------
    path : Path
        The file to write.
    records : Iterable[tuple[str, np.ndarray]]
        Name and array pairs, written in this order.

    Raises
    ------
    ValueError
        If an array isn't float32 or float64, or a name repeats.
    """

    # A crash mid-write leaves only the temporary file, so a half-written dump never passes for a finished one
    temporary = path.with_name(path.name + ".tmp")
    seen = set()
    with temporary.open("wb") as file:
        # Step 1: the file header. `struct.pack` turns Python numbers into raw bytes: `<` is little-endian, `I` a uint32
        file.write(MAGIC + struct.pack("<I", VERSION))

        for name, array in records:
            # Step 2: refuse what the format can't hold, before any byte of the record is written
            if name in seen:
                raise ValueError(f"{path}: record {name!r} written twice")
            if array.dtype not in CODES:
                raise ValueError(
                    f"{path}: record {name!r} is {array.dtype}, not float32 or float64"
                )
            seen.add(name)

            # Step 3: the name, as a uint16 (`H`) byte count followed by its UTF-8 bytes
            encoded = name.encode()
            file.write(struct.pack("<H", len(encoded)) + encoded)

            # Step 4: the dtype code and ndim as one uint8 (`B`) each, then one uint32 per dimension
            file.write(struct.pack("<BB", CODES[array.dtype], array.ndim))
            file.write(struct.pack(f"<{array.ndim}I", *array.shape))

            # Step 5: the values. `ascontiguousarray` lays a sliced or transposed array out in row-major order and converts it to little-endian if needed, so `tobytes` gives exactly the bytes the format expects
            file.write(
                np.ascontiguousarray(
                    array, dtype=array.dtype.newbyteorder("<")
                ).tobytes()
            )

    # Step 6: the rename is atomic, so the final name only ever points at a complete file
    os.replace(temporary, path)


def read(path: Path) -> dict[str, np.ndarray]:
    """Read every record of a `.tdump` file.

    Parameters
    ----------
    path : Path
        The file to read.

    Returns
    -------
    dict[str, np.ndarray]
        The records by name, in file order.

    Raises
    ------
    ValueError
        If the magic or version is wrong, a record is cut short or has an unknown dtype, or a name repeats.
    """

    # Step 1: the whole file in memory, and its header: 4 magic bytes, then the version as a uint32 at byte 4
    data = path.read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"{path}: not a tdump file (magic {data[:4]!r})")
    if len(data) < 8:
        raise ValueError(f"{path}: truncated header")
    (version,) = struct.unpack_from("<I", data, 4)
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, expected {VERSION}")

    records = {}
    offset = 8

    # Step 2: a cursor over the bytes. Every read goes through `take`, so a file cut short fails with the field it stopped in, never with a short slice decoded as garbage
    def take(size: int, what: str) -> bytes:
        nonlocal offset
        if offset + size > len(data):
            raise ValueError(f"{path}: truncated {what} at byte {offset}")
        chunk = data[offset : offset + size]
        offset += size
        return chunk

    # Step 3: one record per loop, reading the fields in the order `write` puts them
    while offset < len(data):
        # Step 3a: the name, a uint16 byte count then the UTF-8 bytes
        (length,) = struct.unpack("<H", take(2, "name length"))
        name = take(length, "name").decode()

        # Step 3b: the dtype code and ndim, then the dims, which give how many value bytes follow
        code, ndim = struct.unpack("<BB", take(2, f"header of {name!r}"))
        if code not in DTYPES:
            raise ValueError(f"{path}: record {name!r} has unknown dtype code {code}")
        shape = struct.unpack(f"<{ndim}I", take(4 * ndim, f"dims of {name!r}"))
        dtype = DTYPES[code]

        # `np.prod(())` is 1, so a 0-dim record reads one value
        count = int(np.prod(shape))
        values = take(count * dtype.itemsize, f"values of {name!r}")
        if name in records:
            raise ValueError(f"{path}: record {name!r} appears twice")

        # Step 3c: `frombuffer` views the bytes as little-endian values without copying. The copy in the host's own byte order gives an array that owns its memory and is writable, then `reshape` restores the dims
        records[name] = (
            np.frombuffer(values, dtype=dtype)
            .astype(dtype.newbyteorder("="), copy=True)
            .reshape(shape)
        )
    return records
