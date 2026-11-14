"""What the program keeps in memory: weights, code, static buffers, the stack bound and the heap."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Internal
from score.profile.errors import ProfileError
from score.profile.heap import HEAP_ALLOCATIONS
from score.profile.histogram import Histogram
from score.profile.stack import StackBound, stack_bound
from score.profile.symbols import PK_SLICE, FunctionTable, split_by_function
from score.profile.weights import read_tensor_sizes, weights_symbol_size


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The binutils that read section sizes
SIZE = "riscv64-none-elf-size"
SIZE_ARGUMENTS = ("-A",)

# The sections of the program, by what lives in them
TEXT_SECTION = ".text"
RODATA_SECTION = ".rodata"
STATIC_SECTIONS = (".bss", ".sbss", ".data", ".sdata", RODATA_SECTION)

# The function the stack bound starts from
STACK_ROOT = "window_run"

# What a row measures
MemoryKind = Literal["weights", "code", "static", "stack", "heap"]

# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class MemoryRow:
    """One line of the memory table.

    Attributes
    ----------
    kind : MemoryKind
        What the row measures.
    item : str
        The row's name within its kind.
    bytes : int
        Its size in bytes; a heap row's value for 1 channel.
    formula : str
        A heap row's size as a formula of `channels`, empty for the rest.
    source : str
        Where the number comes from.
    """

    kind: MemoryKind
    item: str
    bytes: int
    formula: str
    source: str


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_section_sizes(elf: Path) -> dict[str, int]:
    """Read the size of each section of the ELF.

    Parameters
    ----------
    elf : Path
        The linked program.

    Returns
    -------
    dict[str, int]
        Bytes by section name.

    Raises
    ------
    ProfileError
        If `size` fails.
    """

    result = subprocess.run(
        [SIZE, *SIZE_ARGUMENTS, str(elf)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ProfileError(f"{SIZE} failed on {elf}: {result.stderr.strip()}")
    sizes: dict[str, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0].startswith("."):
            sizes[fields[0]] = int(fields[1])
    return sizes


def weights_row(elf: Path, weights_table: Path) -> MemoryRow:
    """Size the weights from the ELF and check the generated table agrees.

    Parameters
    ----------
    elf : Path
        The linked program.
    weights_table : Path
        The generated `weights_table.c`.

    Returns
    -------
    MemoryRow
        The weights' bytes.

    Raises
    ------
    ProfileError
        If the symbol's size differs from the tensors' sum.
    """

    symbol = weights_symbol_size(elf)
    tensors = sum(read_tensor_sizes(weights_table).values())
    if symbol != tensors:
        raise ProfileError(
            f"weights symbol is {symbol} bytes but the table's tensors add up to {tensors}"
        )
    return MemoryRow(
        "weights", "weights", symbol, "", "size of the weights symbol in .rodata"
    )


def code_rows(
    elf: Path, table: FunctionTable, difference: Histogram
) -> list[MemoryRow]:
    """Size the code, and the share of it that a window ran.

    Parameters
    ----------
    elf : Path
        The linked program.
    table : FunctionTable
        The functions of `elf`.
    difference : Histogram
        One window's instructions per address, from the differential runs.

    Returns
    -------
    list[MemoryRow]
        `.text`, then the functions the window ran, pk's slice left out.

    Raises
    ------
    ProfileError
        If the ELF has no `.text`, or an address is in no function.
    """

    sections = read_section_sizes(elf)
    if TEXT_SECTION not in sections:
        raise ProfileError(f"{elf}: no {TEXT_SECTION} section")
    ran = {
        name
        for name, count in split_by_function(difference, table).items()
        if count > 0 and name != PK_SLICE
    }
    ran_bytes = sum(f.end - f.start for f in table.functions if f.name in ran)
    return [
        MemoryRow("code", TEXT_SECTION, sections[TEXT_SECTION], "", "section size"),
        MemoryRow(
            "code",
            "functions the window ran",
            ran_bytes,
            "",
            f"symbol sizes of the {len(ran)} functions the histogram saw",
        ),
    ]


def static_rows(elf: Path, weights_bytes: int) -> list[MemoryRow]:
    """Size the static buffers, one row per section.

    Parameters
    ----------
    elf : Path
        The linked program.
    weights_bytes : int
        The weights, which sit in `.rodata` and are counted in their own row.

    Returns
    -------
    list[MemoryRow]
        One row for each of `.bss`, `.sbss`, `.data`, `.sdata` and `.rodata` without the weights.

    Raises
    ------
    ProfileError
        If a section is missing.
    """

    sections = read_section_sizes(elf)
    rows = []
    for name in STATIC_SECTIONS:
        if name not in sections:
            raise ProfileError(f"{elf}: no {name} section")
        size = sections[name]
        source = "section size"
        if name == RODATA_SECTION:
            size -= weights_bytes
            source = "section size minus the weights"
        rows.append(MemoryRow("static", name, size, "", source))
    return rows


def stack_row(bound: StackBound) -> MemoryRow:
    """Turn a stack bound into a table row, with its path as the source.

    Parameters
    ----------
    bound : StackBound
        The worst chain below `STACK_ROOT`.

    Returns
    -------
    MemoryRow
        The chain's total, naming each function with its frame.
    """

    path = " > ".join(f"{frame.name} {frame.size}" for frame in bound.frames)
    return MemoryRow(
        "stack",
        f"worst path from {STACK_ROOT}",
        bound.total,
        "",
        f"frames of {path}",
    )


def heap_rows(source_folder: Path) -> list[MemoryRow]:
    """Write down the heap as formulas of the channel count, with the line each comes from.

    Parameters
    ----------
    source_folder : Path
        The `tspulse` source folder.

    Returns
    -------
    list[MemoryRow]
        One row per allocation, valued for 1 channel.

    Raises
    ------
    ProfileError
        If an allocation's line is no longer in its source file.
    """

    rows = []
    for allocation in HEAP_ALLOCATIONS:
        lines = (source_folder / allocation.file).read_text().splitlines()
        found = [i + 1 for i, line in enumerate(lines) if allocation.call in line]
        if len(found) != 1:
            raise ProfileError(
                f"{allocation.file}: expected one `{allocation.call}`, found {len(found)}"
            )
        rows.append(
            MemoryRow(
                "heap",
                allocation.what,
                allocation.per_channel,
                allocation.formula,
                f"{allocation.file}:{found[0]}",
            )
        )
    return rows


def memory_rows(
    elf: Path,
    table: FunctionTable,
    weights_table: Path,
    stack_folder: Path,
    source_folder: Path,
    difference: Histogram,
) -> list[MemoryRow]:
    """Build the whole memory table of one build.

    Parameters
    ----------
    elf : Path
        The clean rv64 program.
    table : FunctionTable
        The functions of `elf`.
    weights_table : Path
        The generated `weights_table.c`.
    stack_folder : Path
        The `.ci` files of the same objects.
    source_folder : Path
        The `tspulse` source folder.
    difference : Histogram
        One window's instructions per address.

    Returns
    -------
    list[MemoryRow]
        Weights, code, static buffers, stack and heap rows.
    """

    weights = weights_row(elf, weights_table)
    return [
        weights,
        *code_rows(elf, table, difference),
        *static_rows(elf, weights.bytes),
        stack_row(stack_bound(elf, table, stack_folder, STACK_ROOT)),
        *heap_rows(source_folder),
    ]
