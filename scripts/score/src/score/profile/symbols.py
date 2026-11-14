"""The program's functions, from the ELF's symbol table, and where an address belongs among them."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import bisect
import subprocess
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

# Internal
from score.profile.errors import ProfileError
from score.profile.histogram import Histogram
from score.spike_check import PK_BASE


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The binutils that read the rv64 ELF
NM = "riscv64-none-elf-nm"

# `--print-size` adds each symbol's byte size, and `--numeric-sort` orders them by address
NM_ARGUMENTS = ("--print-size", "--numeric-sort")

# nm's symbol types for code: global, local and weak
CODE_TYPES = frozenset("TtWw")

# The slice every address from `PK_BASE` up goes to
PK_SLICE = "pk"

# nm prints no size for a symbol of size 0, such as the C runtime's startup helpers, leaving address, type and name
NM_FIELDS_WITHOUT_SIZE = 3


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Function:
    """One function's address range.

    Attributes
    ----------
    name : str
        The symbol name.
    start : int
        Address of its first instruction.
    end : int
        Address just past its last byte.
    """

    name: str
    start: int
    end: int


class FunctionTable:
    """The functions of one ELF, sorted by address, answering which function owns an address."""

    def __init__(self, functions: list[Function]) -> None:
        """Index the functions by start address.

        Parameters
        ----------
        functions : list[Function]
            Non-overlapping ranges, in address order.
        """

        self.functions = functions
        self._starts = [function.start for function in functions]

    def owner(self, address: int) -> str:
        """Name the slice an address belongs to.

        Parameters
        ----------
        address : int
            A program counter from the histogram.

        Returns
        -------
        str
            `PK_SLICE` for pk's addresses, else the function that holds the address.

        Raises
        ------
        ProfileError
            If an address in the program's range is outside every function.
        """

        if address >= PK_BASE:
            return PK_SLICE
        index = bisect.bisect_right(self._starts, address) - 1
        if index >= 0 and address < self.functions[index].end:
            return self.functions[index].name
        raise ProfileError(f"address {address:#x} is in the program but in no function")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_functions(elf: Path) -> FunctionTable:
    """List the functions of an ELF with `nm`.

    A symbol of size 0 runs to the next symbol, since nothing else says where it ends.

    Parameters
    ----------
    elf : Path
        The linked program.

    Returns
    -------
    FunctionTable
        Every code symbol with its range.

    Raises
    ------
    ProfileError
        If `nm` fails, or two functions overlap.
    """

    # Step 1: ask nm, which sorts by address
    result = subprocess.run(
        [NM, *NM_ARGUMENTS, str(elf)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ProfileError(f"{NM} failed on {elf}: {result.stderr.strip()}")

    # Step 2: keep the code symbols as (start, size, name)
    symbols: list[tuple[int, int | None, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        # An undefined weak symbol has no address, only type and name
        if len(fields) < NM_FIELDS_WITHOUT_SIZE or fields[-2] not in CODE_TYPES:
            continue
        size = None if len(fields) == NM_FIELDS_WITHOUT_SIZE else int(fields[1], 16)
        symbols.append((int(fields[0], 16), size, fields[-1]))

    # Step 3: give each a range, closing sizeless ones at the next symbol's start
    # Two functions can share a name (newlib has two `__sbprintf`), so each of them gets its own slice, named with its address, rather than being merged into one
    name_counts = Counter(name for _, _, name in symbols)
    functions: list[Function] = []
    for index, (start, size, name) in enumerate(symbols):
        if name_counts[name] > 1:
            name = f"{name}@{start:#x}"
        if size is None:
            following = symbols[index + 1][0] if index + 1 < len(symbols) else start
            size = following - start
        functions.append(Function(name, start, start + size))
    for before, after in pairwise(functions):
        if before.end > after.start:
            raise ProfileError(f"{before.name} and {after.name} overlap")
    return FunctionTable(functions)


def split_by_function(difference: Histogram, table: FunctionTable) -> dict[str, int]:
    """Add a histogram's counts up per function, "self" instructions only.

    Parameters
    ----------
    difference : Histogram
        Instructions per address, possibly negative at single addresses.
    table : FunctionTable
        The program's functions.

    Returns
    -------
    dict[str, int]
        Instructions per function, plus `PK_SLICE`.

    Raises
    ------
    ProfileError
        If an address is in no function, or a function's total is negative (pk's slice excepted).
    """

    totals: dict[str, int] = {}
    for address, count in difference.items():
        owner = table.owner(address)
        totals[owner] = totals.get(owner, 0) + count
    # pk's slice is its own syscall and page-fault work, which varies slightly between runs, so it may be slightly negative and is kept as measured instead of failing the split
    negative = {
        name: count for name, count in totals.items() if count < 0 and name != PK_SLICE
    }
    if negative:
        raise ProfileError(f"negative instruction counts per function: {negative}")
    return totals
