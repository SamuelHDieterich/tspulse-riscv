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

# The weights are stored as 32-bit floats
FLOAT_BYTES = 4

# The symbol that holds every tensor, in `.rodata`
WEIGHTS_SYMBOL = "TSPULSE_WEIGHTS"

# `nm --print-size` prints address, size in hex, type and name for a sized symbol
NM_ARGUMENTS = ("--print-size",)
NM_FIELDS_WITH_SIZE = 4

# A table row: `{ "name", (const float *)&..., <rank>, { <dims> } },`
ENTRY_LINE = re.compile(r'^\s*\{\s*"([^"]+)",.*,\s*\d+,\s*\{([\d,\s]+)\}\s*\}')

# Tensors of the weights table by name, in bytes
TensorSizes = dict[str, int]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_tensor_sizes(table: Path) -> TensorSizes:
    """Read each tensor's size from the generated `weights_table.c`.

    Parameters
    ----------
    table : Path
        The generated table of tensor names, ranks and dimensions.

    Returns
    -------
    TensorSizes
        Bytes of each tensor, as 4 bytes per element.

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
        name = match.group(1)
        if name in sizes:
            raise ProfileError(f"{table}: tensor {name} appears twice")
        dimensions = [int(part) for part in match.group(2).split(",") if part.strip()]
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
