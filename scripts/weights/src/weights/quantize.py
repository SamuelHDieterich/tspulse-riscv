"""Quantizing the mixer Linear weights to int8 and rendering the int8 set."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import struct
from dataclasses import dataclass

# Internal
from weights.render import (
    ENVIRONMENT,
    Entry,
    designator,
    hex_literals,
    nest,
)
from weights.safetensors import Failure, Tensor


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The Linear weights inside the mixers: the MLP's two layers and the gate, in the patch mixer and in the feature mixer
MIXER_WEIGHT_SUFFIXES = (
    "patch_mixer.mlp.fc1.weight",
    "patch_mixer.mlp.fc2.weight",
    "patch_mixer.gating_block.attn_layer.weight",
    "feature_mixer.mlp.fc1.weight",
    "feature_mixer.mlp.fc2.weight",
    "feature_mixer.gating_block.attn_layer.weight",
)

# 8 encoder layers, 2 decoder layers and 3 forecast layers, each with the six weights above
QUANTIZED_TENSOR_COUNT = 78

# The symmetric range: -128 stays unused, so a row's largest magnitude maps to exactly 127
INT8_MAX = 127

# The scale of a row of zeros, whose quantized values are all 0 whatever the scale is
ZERO_ROW_SCALE = 1.0

SCALE_SUFFIX = "scale"

WEIGHT_SUFFIX = "weight"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class QuantizedTensor:
    """One Linear weight as int8 values with a float32 scale per output row.

    Attributes
    ----------
    name : str
        Its PyTorch name.
    shape : tuple of int
        Its dimensions, `(out, in)`.
    values : bytes
        The int8 values in row-major order, as two's complement bytes.
    scales : bytes
        One little-endian float32 per output row.
    """

    name: str
    shape: tuple[int, ...]
    values: bytes
    scales: bytes

    @property
    def size(self) -> int:
        """The bytes the weight and its scales take in the C struct."""

        return len(self.values) + len(self.scales)

    def dequantized(self) -> Tensor:
        """Give the values the model computes with, `f32(scale * q)`.

        Returns
        -------
        Tensor
            The float32 tensor under the same name.
        """

        rows, columns = self.shape
        scales = struct.unpack(f"<{rows}f", self.scales)
        quantized = struct.unpack(f"{len(self.values)}b", self.values)
        values = [
            f32(scales[r] * quantized[r * columns + c])
            for r in range(rows)
            for c in range(columns)
        ]
        return Tensor(self.name, self.shape, struct.pack(f"<{len(values)}f", *values))


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def f32(value: float) -> float:
    """Round a float64 to the nearest float32.

    Parameters
    ----------
    value : float
        The value.

    Returns
    -------
    float
        The float32 value, held as a float64.

    Examples
    --------
    >>> f32(0.1)
    0.10000000149011612
    """

    return struct.unpack("<f", struct.pack("<f", value))[0]


def quantize_row(row: tuple[float, ...]) -> tuple[float, list[int]]:
    """Quantize one output row of a Linear weight.

    scale = f32(max |w| / 127), q = clamp(round(w / scale), -127, 127)
    with `round` taking ties to even, as in https://docs.python.org/3/library/functions.html#round.
    This is symmetric per-channel quantization, defined in Nagel et al. 2021, https://arxiv.org/abs/2106.08295 section 2.2.
    The division is done in float64 and rounded once to float32, which equals the float32 division because float64 carries more than twice float32's precision (53 >= 2 * 24 + 2).

    Parameters
    ----------
    row : tuple of float
        The row's float32 values.

    Returns
    -------
    tuple of float and list of int
        The float32 scale and the int8 values.

    Examples
    --------
    >>> quantize_row((0.0, 0.0))
    (1.0, [0, 0])
    >>> quantize_row((127.0, -63.5, 0.5))
    (1.0, [127, -64, 0])
    """

    largest = max(abs(v) for v in row)
    if largest == 0:
        return ZERO_ROW_SCALE, [0] * len(row)
    scale = f32(largest / INT8_MAX)
    return scale, [max(-INT8_MAX, min(INT8_MAX, round(v / scale))) for v in row]


def is_quantized(name: str) -> bool:
    """Tell whether a tensor is one of the mixer Linear weights.

    Parameters
    ----------
    name : str
        The PyTorch name.

    Returns
    -------
    bool
        Whether the name ends with one of `MIXER_WEIGHT_SUFFIXES`.

    Examples
    --------
    >>> is_quantized("backbone.encoder_block.mixers.0.patch_mixer.mlp.fc1.weight")
    True
    >>> is_quantized("backbone.encoder_block.mixers.0.patch_mixer.mlp.fc1.bias")
    False
    """

    return name.endswith(MIXER_WEIGHT_SUFFIXES)


def quantize(tensors: list[Tensor]) -> list[Tensor | QuantizedTensor]:
    """Quantize the mixer Linear weights and keep every other tensor as it is.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors the program runs.

    Returns
    -------
    list of Tensor or QuantizedTensor
        The same tensors in the same order.

    Raises
    ------
    Failure
        When the number of matching tensors is not `QUANTIZED_TENSOR_COUNT`, or one isn't 2-D.
    """

    result: list[Tensor | QuantizedTensor] = []
    count = 0
    for tensor in tensors:
        if not is_quantized(tensor.name):
            result.append(tensor)
            continue
        if len(tensor.shape) != 2:
            raise Failure(f"{tensor.name} has shape {tensor.shape}, not [out, in].")
        count += 1
        rows, columns = tensor.shape
        flat = struct.unpack(f"<{rows * columns}f", tensor.raw)
        scales = []
        values = []
        for r in range(rows):
            scale, quantized = quantize_row(flat[r * columns : (r + 1) * columns])
            scales.append(scale)
            values.extend(quantized)
        result.append(
            QuantizedTensor(
                tensor.name,
                tensor.shape,
                struct.pack(f"{len(values)}b", *values),
                struct.pack(f"<{rows}f", *scales),
            )
        )
    if count != QUANTIZED_TENSOR_COUNT:
        raise Failure(
            f"{count} tensors end with {' or '.join(MIXER_WEIGHT_SUFFIXES)}, not {QUANTIZED_TENSOR_COUNT}, so this is not the expected file."
        )
    return result


def dequantize(tensors: list[Tensor | QuantizedTensor]) -> list[Tensor]:
    """Give every tensor as float32, quantized ones with the values the model computes with.

    Parameters
    ----------
    tensors : list of Tensor or QuantizedTensor
        The output of `quantize`.

    Returns
    -------
    list of Tensor
        The float32 tensors, for the `.tdump`.
    """

    return [t.dequantized() if isinstance(t, QuantizedTensor) else t for t in tensors]


def scale_designator(name: str) -> str:
    """Give the C designators of a quantized weight's scales, which sit beside it.

    Parameters
    ----------
    name : str
        The weight's PyTorch name.

    Returns
    -------
    str
        The designators, starting with `.`.

    Examples
    --------
    >>> scale_designator("backbone.encoder_block.mixers.3.patch_mixer.mlp.fc1.weight")
    '.backbone.encoder_block.mixers[3].patch_mixer.mlp.fc1.scale'
    """

    return designator(name.removesuffix(WEIGHT_SUFFIX) + SCALE_SUFFIX)


def render_int8_weights(tensors: list[Tensor | QuantizedTensor]) -> str:
    """Render `weights_int8.c`, which defines the weights as one const struct with int8 mixer weights.

    Parameters
    ----------
    tensors : list of Tensor or QuantizedTensor
        The output of `quantize`.

    Returns
    -------
    str
        The C source.
    """

    initializers = []
    for tensor in tensors:
        if isinstance(tensor, QuantizedTensor):
            rows, _ = tensor.shape
            values = [
                str(v) for v in struct.unpack(f"{len(tensor.values)}b", tensor.values)
            ]
            initializers.append(
                (designator(tensor.name), tensor.shape, nest(values, tensor.shape))
            )
            initializers.append(
                (
                    scale_designator(tensor.name),
                    (rows,),
                    nest(hex_literals(tensor.name, tensor.scales), (rows,)),
                )
            )
        else:
            initializers.append(
                (
                    designator(tensor.name),
                    tensor.shape,
                    nest(hex_literals(tensor.name, tensor.raw), tensor.shape),
                )
            )

    total_bytes = sum(
        t.size if isinstance(t, QuantizedTensor) else len(t.raw) for t in tensors
    )
    return ENVIRONMENT.get_template("weights_int8.c.j2").render(
        initializers=initializers, total_bytes=total_bytes
    )


def render_int8_table(tensors: list[Tensor | QuantizedTensor]) -> str:
    """Render `weights_int8_table.c`, which lists every tensor by name, pointer and shape.

    Parameters
    ----------
    tensors : list of Tensor or QuantizedTensor
        The output of `quantize`.

    Returns
    -------
    str
        The C source.
    """

    prepared = [
        (
            Entry(t.name, designator(t.name), t.shape),
            scale_designator(t.name) if isinstance(t, QuantizedTensor) else None,
        )
        for t in tensors
    ]
    return ENVIRONMENT.get_template("weights_int8_table.c.j2").render(tensors=prepared)
