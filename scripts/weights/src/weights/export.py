"""Reading a safetensors file and writing the weights as C source and a reference dump."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path


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


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The baseline never runs this head, so its weights stay out of the program
SKIPPED_PREFIXES = ("decoder_with_head.fft_softmax_mapping.",)

# Values per line in the generated initializers
VALUES_PER_LINE = 8

FLOAT_BYTES = 4

# Index of the `.tdump` dtype for float32
TDUMP_FLOAT32 = 1

TDUMP_MAGIC = b"TDMP"

TDUMP_VERSION = 1


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

    try:
        content = path.read_bytes()
    except OSError as error:
        raise Failure(f"Could not read {path}: {error}") from error

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

    try:
        header = json.loads(content[8:data_start])
    except ValueError as error:
        raise Failure(f"{path} has a header that is not JSON: {error}") from error
    if not isinstance(header, dict):
        raise Failure(f"{path} has a header that is not a JSON object.")
    header.pop("__metadata__", None)

    data = memoryview(content)[data_start:]
    spans = []
    for name, entry in header.items():
        spans.append((*_span(path, name, entry, len(data)), name, entry["shape"]))
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
        and all(isinstance(d, int) and d >= 0 for d in entry["shape"])
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


def designator(name: str) -> str:
    """Turn a PyTorch tensor name into a chain of C designators.

    A number after `mixers` indexes an array, so module code can loop over the mixers.
    Any other number names a member `_n`, because those modules are an `nn.Sequential` of different layers.

    Parameters
    ----------
    name : str
        The PyTorch name, such as `backbone.encoder_block.mixers.3.patch_mixer.mlp.fc1.weight`.

    Returns
    -------
    str
        The designators, starting with `.`, such as `.backbone.encoder_block.mixers[3].patch_mixer.mlp.fc1.weight`.
    """

    path = ""
    previous = ""
    for segment in name.split("."):
        if segment.isdigit():
            path += f"[{segment}]" if previous == "mixers" else f"._{segment}"
        else:
            path += f".{segment}"
        previous = segment
    return path


def hex_literal(raw: bytes) -> list[str]:
    """Write float32 values as exact C hex literals.

    Parameters
    ----------
    raw : bytes
        Little-endian float32 values.

    Returns
    -------
    list of str
        One literal per value, such as `0x1.8p+1f`.

    Raises
    ------
    Failure
        When a value is NaN or infinite, which have no hex literal.
    """

    values = struct.unpack(f"<{len(raw) // FLOAT_BYTES}f", raw)
    if not all(math.isfinite(v) for v in values):
        raise Failure("A tensor holds a NaN or an infinity, which have no hex literal.")
    return [float.hex(v) + "f" for v in values]


def initializer(literals: list[str], shape: tuple[int, ...], indent: str) -> list[str]:
    """Lay out values as nested braces, one level per dimension.

    Parameters
    ----------
    literals : list of str
        The values in row-major order.
    shape : tuple of int
        The dimensions.
    indent : str
        The indentation of the opening brace.

    Returns
    -------
    list of str
        The lines, from the opening brace to the closing one.
    """

    inner = indent + "    "
    lines = [indent + "{"]
    if len(shape) == 1:
        for i in range(0, len(literals), VALUES_PER_LINE):
            lines.append(inner + ", ".join(literals[i : i + VALUES_PER_LINE]) + ",")
    else:
        stride = math.prod(shape[1:])
        for i in range(shape[0]):
            lines += initializer(
                literals[i * stride : (i + 1) * stride], shape[1:], inner
            )
            lines[-1] += ","
    lines.append(indent + "}")
    return lines


def render_weights(tensors: list[Tensor]) -> str:
    """Write `weights.c`, which defines the weights as one const struct.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors to embed.

    Returns
    -------
    str
        The C source.
    """

    lines = [
        "// Generated by export-weights. Do not edit.",
        '#include "weights/weights.h"',
        "",
        "const struct tspulse_weights TSPULSE_WEIGHTS = {",
    ]
    for tensor in tensors:
        body = initializer(hex_literal(tensor.raw), tensor.shape, "    ")
        body[0] = f"    {designator(tensor.name)} = {{"
        lines += body
        lines[-1] += ","

    total = sum(len(t.raw) for t in tensors)
    lines += [
        "};",
        "",
        f'_Static_assert(sizeof(struct tspulse_weights) == {total}, "struct tspulse_weights and the exported tensors differ in size");',
        "",
    ]
    return "\n".join(lines)


def render_table(tensors: list[Tensor]) -> str:
    """Write `weights_table.c`, which lists every tensor by name, pointer and shape.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors in the weights struct.

    Returns
    -------
    str
        The C source.
    """

    lines = [
        "// Generated by export-weights. Do not edit.",
        '#include "weights/weights.h"',
        '#include "weights/table.h"',
        "",
        "const struct weight_entry WEIGHT_TABLE[] = {",
    ]
    for tensor in tensors:
        dims = ", ".join(str(d) for d in tensor.shape)
        lines.append(
            f'    {{ "{tensor.name}", (const float *)&TSPULSE_WEIGHTS{designator(tensor.name)}, {len(tensor.shape)}, {{ {dims} }} }},'
        )
    lines += [
        "};",
        "",
        f"const int WEIGHT_COUNT = {len(tensors)};",
        "",
    ]
    return "\n".join(lines)


def render_tdump(tensors: list[Tensor]) -> bytes:
    """Write `weights.tdump`, the reference the embedded weights are compared against.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors, one float32 record each.

    Returns
    -------
    bytes
        The dump: magic, version, then per record the name, dtype, shape and values, all little-endian.
    """

    out = [TDUMP_MAGIC, struct.pack("<I", TDUMP_VERSION)]
    for tensor in tensors:
        name = tensor.name.encode()
        out.append(struct.pack("<H", len(name)) + name)
        out.append(struct.pack("<BB", TDUMP_FLOAT32, len(tensor.shape)))
        out.append(struct.pack(f"<{len(tensor.shape)}I", *tensor.shape))
        out.append(tensor.raw)
    return b"".join(out)


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
