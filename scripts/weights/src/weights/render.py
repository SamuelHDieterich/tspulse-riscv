"""Rendering tensors as the C source files."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import struct
from dataclasses import dataclass

# External
from jinja2 import Environment, PackageLoader, StrictUndefined

# Internal
from weights.safetensors import FLOAT_BYTES, Failure, Tensor


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# One short row per line keeps the generated file readable in an editor
VALUES_PER_LINE = 8

# Template tags and the line breaks after them don't appear in the output, so the templates can be laid out like the C they produce
ENVIRONMENT = Environment(
    loader=PackageLoader("weights", "templates"),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


# A list of rows for a 1-D tensor, wrapped in one more list per extra dimension
type Rows = list[list[str]] | list[Rows]


@dataclass(frozen=True)
class Entry:
    """One tensor, prepared for the templates.

    Attributes
    ----------
    name : str
        Its PyTorch name.
    designator : str
        Its chain of C designators.
    shape : tuple of int
        Its dimensions.
    """

    name: str
    designator: str
    shape: tuple[int, ...]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def designator(name: str) -> str:
    """Turn a PyTorch tensor name into a chain of C designators.

    A number after `mixers` indexes an array, so module code can loop over the mixers.
    Any other number names a member `_n`, because those modules are an `nn.Sequential` of different layers.

    Parameters
    ----------
    name : str
        The PyTorch name.

    Returns
    -------
    str
        The designators, starting with `.`.

    Examples
    --------
    >>> designator("backbone.encoder_block.mixers.3.patch_mixer.mlp.fc1.weight")
    '.backbone.encoder_block.mixers[3].patch_mixer.mlp.fc1.weight'
    >>> designator("decoder_with_head.forecast_mapping.0.mixers.2.feature_mixer.norm.norm.bias")
    '.decoder_with_head.forecast_mapping._0.mixers[2].feature_mixer.norm.norm.bias'
    >>> designator("decoder_with_head.forecast_mapping.3.weight")
    '.decoder_with_head.forecast_mapping._3.weight'
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


def hex_literals(name: str, raw: bytes) -> list[str]:
    """Write float32 values as exact C hex literals.

    Parameters
    ----------
    name : str
        The tensor's name, for the failure message.
    raw : bytes
        Little-endian float32 values.

    Returns
    -------
    list of str
        One literal per value.

    Raises
    ------
    Failure
        When a value is NaN or infinite, which have no hex literal.

    Examples
    --------
    >>> hex_literals("x", struct.pack("<f", 3.0))
    ['0x1.8000000000000p+1f']
    """

    values = struct.unpack(f"<{len(raw) // FLOAT_BYTES}f", raw)
    if not all(math.isfinite(v) for v in values):
        raise Failure(f"{name} holds a NaN or an infinity, which have no hex literal.")
    return [float.hex(v) + "f" for v in values]


def nest(literals: list[str], shape: tuple[int, ...]) -> Rows:
    """Split values into rows, nested to match the shape.

    Parameters
    ----------
    literals : list of str
        The values in row-major order.
    shape : tuple of int
        The dimensions.

    Returns
    -------
    Rows
        For one dimension, the rows of up to `VALUES_PER_LINE` values.
        Otherwise one nested list per entry of the first dimension.

    Examples
    --------
    >>> nest(["a", "b", "c"], (3,))
    [['a', 'b', 'c']]
    >>> nest([str(i) for i in range(10)], (10,))
    [['0', '1', '2', '3', '4', '5', '6', '7'], ['8', '9']]
    >>> nest(["a", "b", "c", "d"], (2, 2))
    [[['a', 'b']], [['c', 'd']]]
    """

    if len(shape) == 1:
        return [
            literals[i : i + VALUES_PER_LINE]
            for i in range(0, len(literals), VALUES_PER_LINE)
        ]
    stride = math.prod(shape[1:])
    return [
        nest(literals[i * stride : (i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


def entries(tensors: list[Tensor]) -> list[Entry]:
    """Prepare tensors for the templates.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors to embed.

    Returns
    -------
    list of Entry
        One per tensor, in the same order.
    """

    return [Entry(t.name, designator(t.name), t.shape) for t in tensors]


def render_weights(tensors: list[Tensor]) -> str:
    """Render `weights.c`, which defines the weights as one const struct.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors to embed.

    Returns
    -------
    str
        The C source.
    """

    return ENVIRONMENT.get_template("weights.c.j2").render(
        initializers=[
            (
                Entry(t.name, designator(t.name), t.shape),
                nest(hex_literals(t.name, t.raw), t.shape),
            )
            for t in tensors
        ],
        total_bytes=sum(len(t.raw) for t in tensors),
    )


def render_table(tensors: list[Tensor]) -> str:
    """Render `weights_table.c`, which lists every tensor by name, pointer and shape.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors in the weights struct.

    Returns
    -------
    str
        The C source.
    """

    return ENVIRONMENT.get_template("weights_table.c.j2").render(
        tensors=entries(tensors)
    )
