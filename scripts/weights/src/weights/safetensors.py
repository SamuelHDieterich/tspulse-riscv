"""Reading a safetensors file."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


FLOAT_BYTES = 4


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Failure(Exception):
    """A step failed in a way the user can act on, and the message says how."""


@dataclass(frozen=True)
class Tensor:
    """One float32 tensor, as stored in the file.

    Attributes
    ----------
    name : str
        Its PyTorch name.
    shape : tuple of int
        Its dimensions.
    raw : bytes
        Its values as little-endian float32, in row-major order.
    """

    name: str
    shape: tuple[int, ...]
    raw: bytes


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_safetensors(path: Path) -> list[Tensor]:
    """Read every tensor of a safetensors file, in file order.

    Parameters
    ----------
    path : Path
        The file: a little-endian `uint64` header length, a JSON header, then the raw data.

    Returns
    -------
    list of Tensor
        The tensors, ordered by their position in the data.

    Raises
    ------
    Failure
        When the file can't be read, the header doesn't parse, a tensor isn't float32 or has no elements, or the offsets overlap or run past the data.
    """

    # Read the file
    try:
        content = path.read_bytes()
    except OSError as error:
        raise Failure(f"Could not read {path}: {error}") from error

    # Read the header length
    if len(content) < 8:
        raise Failure(
            f"{path} is {len(content)} bytes, too short for a safetensors header."
        )
    (header_length,) = struct.unpack_from("<Q", content)
    data_start = 8 + header_length
    if data_start > len(content):
        raise Failure(
            f"{path} declares a {header_length}-byte header, but only {len(content) - 8} bytes follow the length."
        )

    # Parse the JSON header
    try:
        header = json.loads(content[8:data_start])
    except ValueError as error:
        raise Failure(f"{path} has a header that is not JSON: {error}") from error
    if not isinstance(header, dict):
        raise Failure(f"{path} has a header that is not a JSON object.")
    header.pop("__metadata__", None)

    # Check each entry
    data = memoryview(content)[data_start:]
    spans = []
    for name, entry in header.items():
        spans.append((*_span(path, name, entry, len(data)), name, entry["shape"]))

    # Sort by offset and check that no two overlap
    spans.sort()
    tensors = []
    previous_end = 0
    previous_name = None
    for begin, end, name, shape in spans:
        if begin < previous_end:
            raise Failure(f"{path}: {name} overlaps {previous_name}.")
        tensors.append(Tensor(name, tuple(shape), bytes(data[begin:end])))
        previous_end = end
        previous_name = name
    return tensors


def _span(path: Path, name: str, entry: object, size: int) -> tuple[int, int]:
    """Check one header entry and return where its bytes lie in the data.

    Parameters
    ----------
    path : Path
        The file, for messages.
    name : str
        The tensor's name.
    entry : object
        The entry as parsed from the header JSON.
    size : int
        How many bytes of data the file holds.

    Returns
    -------
    tuple of int
        The begin and end offsets into the data.

    Raises
    ------
    Failure
        When the entry is malformed, isn't float32, has no elements, or doesn't fit.
    """

    if not (
        isinstance(entry, dict)
        and isinstance(entry.get("dtype"), str)
        and isinstance(entry.get("shape"), list)
        and all(
            isinstance(d, int) and not isinstance(d, bool) and d >= 0
            for d in entry["shape"]
        )
        and isinstance(entry.get("data_offsets"), list)
        and len(entry["data_offsets"]) == 2
        and all(isinstance(o, int) for o in entry["data_offsets"])
    ):
        raise Failure(f"{path}: the header entry for {name} is malformed.")
    if entry["dtype"] != "F32":
        raise Failure(f"{path}: {name} is {entry['dtype']}, but only F32 is supported.")
    if not entry["shape"] or 0 in entry["shape"]:
        raise Failure(
            f"{path}: {name} has shape {entry['shape']}, which C arrays can't hold."
        )

    begin, end = entry["data_offsets"]
    if not 0 <= begin <= end <= size:
        raise Failure(
            f"{path}: {name} spans bytes {begin} to {end}, but the data holds {size} bytes."
        )
    elements = math.prod(entry["shape"])
    if end - begin != elements * FLOAT_BYTES:
        raise Failure(
            f"{path}: {name} has shape {entry['shape']}, which needs {elements * FLOAT_BYTES} bytes, but spans {end - begin}."
        )
    return begin, end
