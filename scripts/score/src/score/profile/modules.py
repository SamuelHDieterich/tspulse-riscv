"""Instructions per PyTorch module for one window, from the profile build's CSV."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
from dataclasses import dataclass
from pathlib import Path

# Internal
from score.profile.errors import ProfileError


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The profile CSV's header
PROFILE_HEADER = ["window", "path", "depth", "instructions", "calls"]

# The two rows that cover the whole window: its count with every hook's cost subtracted, and with nothing subtracted
WINDOW_ROOT = "window"
RAW_ROOT = "raw"

# The copy (`hidden` or `plain`) is the first push, so its depth is 1, and a module is a child of the row one depth up
COPY_DEPTH = 1

# The copy ends with `/`, and the PyTorch names after it are joined with `.`
COPY_SEPARATOR = "/"
NAME_SEPARATOR = "."


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class ModuleRow:
    """One module's instructions in one window.

    Attributes
    ----------
    path : str
        `WINDOW_ROOT`, `RAW_ROOT`, or the copy then `/` then the PyTorch names, such as `hidden/backbone.encoder_block`.
    depth : int
        Pushes open when the module was entered: 0 for the roots, 1 for a copy.
    instructions : int
        Instructions inside the module, children included.
    calls : int
        Times the module was entered; for `RAW_ROOT`, the number of hook calls.
    own : int or None
        `instructions` minus the children's, so the leaves add up to the total; None for `RAW_ROOT`, which has no children.
        The table writes it as `self`.
    """

    path: str
    depth: int
    instructions: int
    calls: int
    own: int | None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_profile(path: Path, window: int) -> list[ModuleRow]:
    """Read one window's rows from the profile build's CSV, as they come.

    The profile run covers the window before the measured one too, so the other window's rows are skipped.

    Parameters
    ----------
    path : Path
        The CSV written by `--profile`.
    window : int
        The measured window.

    Returns
    -------
    list[ModuleRow]
        The window's rows in the order the program wrote them, with `own` still unset.

    Raises
    ------
    ProfileError
        If the header is wrong, a line is malformed, or the window has no rows.
    """

    with path.open(newline="") as file:
        lines = list(csv.reader(file))
    if not lines or lines[0] != PROFILE_HEADER:
        raise ProfileError(f"{path}: header is not {','.join(PROFILE_HEADER)}")
    rows = []
    for line in lines[1:]:
        try:
            row_window, module, depth, instructions, calls = line
            if int(row_window) == window:
                rows.append(
                    ModuleRow(module, int(depth), int(instructions), int(calls), None)
                )
        except ValueError:
            raise ProfileError(f"{path}: not a profile line: {line}") from None
    if not rows:
        raise ProfileError(f"{path}: no rows for window {window}")
    return rows


def nest_modules(rows: list[ModuleRow]) -> list[ModuleRow]:
    """Give each module its own instructions: its count minus its children's.

    The window root's children are the two copies, and what is left is the work outside any module, such as reading and pushing the row.

    Parameters
    ----------
    rows : list[ModuleRow]
        One window's rows from `read_profile`.

    Returns
    -------
    list[ModuleRow]
        `WINDOW_ROOT` and `RAW_ROOT` first, then the modules, each with `own` set except for `RAW_ROOT`.

    Raises
    ------
    ProfileError
        If a root is missing or repeated, a module has no single parent, or a module's count is below its children's.
    """

    # Step 1: the roots, then the modules
    roots = {row.path: row for row in rows if row.depth == 0}
    modules = [row for row in rows if row.depth != 0]
    if set(roots) != {WINDOW_ROOT, RAW_ROOT} or len(rows) - len(modules) != 2:
        raise ProfileError(f"expected one {WINDOW_ROOT} and one {RAW_ROOT} row")

    # Step 2: each module's children, found by path prefix one depth down
    # A push can carry several dotted names at once (`mixers.0`), so the depth, not the dots, says who the parent is
    children: dict[str, int] = {WINDOW_ROOT: 0}
    for module in modules:
        children[module.path] = 0
    for module in modules:
        if module.depth == COPY_DEPTH:
            children[WINDOW_ROOT] += module.instructions
            continue
        separator = COPY_SEPARATOR if module.depth == COPY_DEPTH + 1 else NAME_SEPARATOR
        parents = [
            row
            for row in modules
            if row.depth == module.depth - 1
            and module.path.startswith(row.path + separator)
        ]
        if len(parents) != 1:
            raise ProfileError(
                f"{module.path}: expected one parent at depth {module.depth - 1}, found {len(parents)}"
            )
        children[parents[0].path] += module.instructions

    # Step 3: own = inclusive - children's inclusive; a negative one means the nesting is wrong
    window_own = roots[WINDOW_ROOT].instructions - children[WINDOW_ROOT]
    if window_own < 0:
        raise ProfileError(f"{WINDOW_ROOT}: the copies add up to more than it")
    nested = [
        ModuleRow(
            roots[WINDOW_ROOT].path,
            0,
            roots[WINDOW_ROOT].instructions,
            roots[WINDOW_ROOT].calls,
            window_own,
        ),
        roots[RAW_ROOT],
    ]
    for module in modules:
        own = module.instructions - children[module.path]
        if own < 0:
            raise ProfileError(f"{module.path}: its children add up to more than it")
        nested.append(
            ModuleRow(module.path, module.depth, module.instructions, module.calls, own)
        )
    return nested
