"""Rendering tensors as a `.tdump` reference file."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import struct

# Internal
from weights.safetensors import Failure, Tensor


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# The `.tdump` code for float32
TDUMP_FLOAT32 = 1

TDUMP_MAGIC = b"TDMP"

TDUMP_VERSION = 1

TDUMP_MAX_NAME = 65535

TDUMP_MAX_NDIM = 255


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def render_tdump(tensors: list[Tensor]) -> bytes:
    """Render `weights.tdump`, the reference the embedded weights are compared against.

    Parameters
    ----------
    tensors : list of Tensor
        The tensors, one float32 record each.

    Returns
    -------
    bytes
        The dump: magic, version, then per record the name, dtype, shape and values, all little-endian.

    Raises
    ------
    Failure
        If a name is longer than 65535 UTF-8 bytes or a tensor has more than 255 dimensions.
    """

    out = [TDUMP_MAGIC, struct.pack("<I", TDUMP_VERSION)]
    for tensor in tensors:
        name = tensor.name.encode()
        if len(name) > TDUMP_MAX_NAME:
            raise Failure(
                f"{tensor.name} has a name longer than {TDUMP_MAX_NAME} bytes."
            )
        if len(tensor.shape) > TDUMP_MAX_NDIM:
            raise Failure(f"{tensor.name} has more than {TDUMP_MAX_NDIM} dimensions.")
        out.append(struct.pack("<H", len(name)) + name)
        out.append(struct.pack("<BB", TDUMP_FLOAT32, len(tensor.shape)))
        out.append(struct.pack(f"<{len(tensor.shape)}I", *tensor.shape))
        out.append(tensor.raw)
    return b"".join(out)
