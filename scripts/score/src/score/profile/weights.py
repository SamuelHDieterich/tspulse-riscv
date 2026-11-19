"""The weight tensors of the generated table: names and byte sizes."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
import re
import subprocess
from pathlib import Path

# Internal
from score.profile.errors import ProfileError
from score.profile.symbols import NM


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# A float weight is a 32-bit float
FLOAT_BYTES = 4

# The symbol that holds every tensor, in `.rodata`
WEIGHTS_SYMBOL = "TSPULSE_WEIGHTS"

# `nm --print-size` prints address, size in hex, type and name for a sized symbol
NM_ARGUMENTS = ("--print-size",)
NM_FIELDS_WITH_SIZE = 4

# The text of a table row's column that holds no pointer
NULL_COLUMN = "NULL"

# An int8 weight is one byte, and each of its output rows carries one 32-bit float scale
INT8_BYTES = 1
SCALE_BYTES = 4

# A table row: `{ "name", (const float *)&..., <rank>, { <dims> } },`.
# The int8 table's rows add two columns after the dimensions, `(const int8_t *)&..., <scale>` for a quantized tensor and `NULL, NULL` for a float one
ENTRY_LINE = re.compile(
    r'^\s*\{\s*"(?P<name>[^"]+)",.*,\s*\d+,\s*\{(?P<dims>[\d,\s]+)\}'
    r"(?:,\s*(?P<q>\(const int8_t \*\)[^,]+|NULL),\s*(?:[^,}]+|NULL))?\s*\}"
)

# Tensors of the weights table by name, in bytes
TensorSizes = dict[str, int]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_tensor_sizes(table: Path) -> TensorSizes:
    """Read each tensor's size from the generated `weights_table.c` or `weights_int8_table.c`.

    Parameters
    ----------
    table : Path
        The generated table of tensor names, ranks and dimensions.

    Returns
    -------
    TensorSizes
        Bytes of each tensor: 4 per element for a float one, and for a quantized one (its row holds the `(const int8_t *)` cast) 1 per element plus 4 per output row for the scales.

    Raises
    ------
    ProfileError
        If the table holds no tensor, or a name appears twice.
    """

    sizes: TensorSizes = {}
    for line in table.read_text().splitlines():
        match = ENTRY_LINE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name in sizes:
            raise ProfileError(f"{table}: tensor {name} appears twice")
        dimensions = [
            int(part) for part in match.group("dims").split(",") if part.strip()
        ]
        if match.group("q") not in (None, NULL_COLUMN):
            # One scale per output row, which is the first dimension
            sizes[name] = (
                math.prod(dimensions) * INT8_BYTES + dimensions[0] * SCALE_BYTES
            )
        else:
            sizes[name] = math.prod(dimensions) * FLOAT_BYTES
    if not sizes:
        raise ProfileError(f"{table}: no tensor found")
    return sizes


def weights_symbol_size(elf: Path) -> int:
    """Read the size of the weights symbol from the ELF.

    Parameters
    ----------
    elf : Path
        The linked program.

    Returns
    -------
    int
        Bytes of `TSPULSE_WEIGHTS`.

    Raises
    ------
    ProfileError
        If `nm` fails or the symbol is missing.
    """

    result = subprocess.run(
        [NM, *NM_ARGUMENTS, str(elf)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ProfileError(f"{NM} failed on {elf}: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == NM_FIELDS_WITH_SIZE and fields[-1] == WEIGHTS_SYMBOL:
            return int(fields[1], 16)
    raise ProfileError(f"{elf}: no {WEIGHTS_SYMBOL} symbol with a size")
