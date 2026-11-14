"""A bound on the stack the model needs, from GCC's frame sizes and the linked program's prologues."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Internal
from score.profile.disassembly import OBJDUMP, OBJDUMP_ARGUMENTS
from score.profile.errors import ProfileError
from score.profile.symbols import FunctionTable


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Where each frame size comes from
SOURCE_CALLGRAPH = "callgraph"
SOURCE_PROLOGUE = "prologue"

# GCC writes a node's frame as `<n> bytes (static)`, with `dynamic` instead of `static` for a size known only at run time, and `bounded` after it when a bound is known
NODE_LINE = re.compile(r'^node: \{ title: "([^"]+)" label: "([^"]*)"')
FRAME_LABEL = re.compile(r"(\d+) bytes \((\w+)(?:,\w+)?\)")
STATIC_FRAME = "static"

# One instruction line: address, mnemonic, then the operands without objdump's `<symbol>` and `#` annotations
INSTRUCTION_LINE = re.compile(
    r"^\s*([0-9a-f]+):\t[0-9a-f ]+\t(\S+)(?:\s+([^<#]*?))?\s*(?:<[^>]*>)?\s*(?:#.*)?$"
)

# Instructions that move the stack pointer by an immediate: `addi sp,sp,-N` and its compressed forms
SP_ADJUST_MNEMONICS = frozenset({"addi", "c.addi", "c.addi16sp"})

# Stores name the stack pointer only as an address base, never as the register they write
STORE_MNEMONICS = frozenset(
    {"sb", "sh", "sw", "sd", "fsw", "fsd", "c.sw", "c.sd", "c.swsp", "c.sdsp"}
    | {"c.fsw", "c.fsd", "c.fswsp", "c.fsdsp"}
)

# Jumps that leave the function: calls that link in `ra`, and tail calls that link in `zero`
DIRECT_JUMPS = frozenset({"jal", "c.j"})
RETURN_REGISTER = "ra"
RETURN_MNEMONICS = frozenset({"c.jr", "jalr"})
STACK_POINTER = "sp"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class StackFrame:
    """One function's frame on the worst path.

    Attributes
    ----------
    name : str
        The function.
    size : int
        Bytes of its frame.
    source : str
        `SOURCE_CALLGRAPH` or `SOURCE_PROLOGUE`.
    """

    name: str
    size: int
    source: str


@dataclass(frozen=True)
class StackBound:
    """The deepest call chain below a root function.

    Attributes
    ----------
    frames : list[StackFrame]
        The chain from the root down, one frame per function.
    total : int
        The frames' sizes added up, in bytes.
    """

    frames: list[StackFrame]
    total: int


@dataclass
class _Body:
    """What one function's instructions say about its frame and its callees.

    Attributes
    ----------
    start : int
        Address of its first instruction.
    end : int
        Address just past its last instruction.
    callees : set[str]
        Functions it calls or tail-calls.
    prologue : int
        The deepest the stack pointer goes below its value on entry, from immediate adjustments, with the running sum starting again after each return.
    calls : bool
        Whether it calls with a return address, which needs a saved `ra` and so a frame.
    problems : list[str]
        Reasons the frame is not a fixed size: a dynamic stack pointer change, an indirect call or jump.
    """

    start: int
    end: int
    callees: set[str]
    prologue: int
    calls: bool
    problems: list[str]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_callgraph_frames(stack_folder: Path) -> dict[str, int]:
    """Read the frame sizes of our own functions from the `.ci` files GCC left beside the objects.

    Library functions appear in the graph without a size and are skipped.

    Parameters
    ----------
    stack_folder : Path
        The folder `make TARGET=rv64` fills, one `.ci` per object.

    Returns
    -------
    dict[str, int]
        Frame bytes by function name.

    Raises
    ------
    ProfileError
        If no `.ci` file is found, a frame is dynamic, or two files size the same name.
    """

    files = sorted(stack_folder.rglob("*.ci"))
    if not files:
        raise ProfileError(f"{stack_folder}: no .ci file; run `make TARGET=rv64`")
    frames: dict[str, int] = {}
    for path in files:
        for line in path.read_text().splitlines():
            node = NODE_LINE.match(line)
            if node is None:
                continue
            label = FRAME_LABEL.search(node.group(2).replace("\\n", "\n"))
            if label is None:
                continue
            name = node.group(1)
            if label.group(2) != STATIC_FRAME:
                raise ProfileError(f"{path}: {name} has a dynamic frame")
            if name in frames:
                raise ProfileError(f"{path}: {name} is sized in two .ci files")
            frames[name] = int(label.group(1))
    return frames


def _read_bodies(elf: Path, table: FunctionTable) -> dict[str, _Body]:
    """Disassemble the program and read each function's callees and stack pointer changes.

    Parameters
    ----------
    elf : Path
        The linked program.
    table : FunctionTable
        The program's functions.

    Returns
    -------
    dict[str, _Body]
        One body per function.

    Raises
    ------
    ProfileError
        If `objdump` fails.
    """

    result = subprocess.run(
        [OBJDUMP, *OBJDUMP_ARGUMENTS, str(elf)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProfileError(f"{OBJDUMP} failed on {elf}: {result.stderr.strip()}")

    bodies = {
        function.name: _Body(function.start, function.end, set(), 0, False, [])
        for function in table.functions
    }
    # The stack pointer's depth below its value on entry, summed in address order
    # It starts again at each return, so a function's epilogues, which give the frame back, never cancel the next prologue's allocation
    depth: dict[str, int] = {}
    for line in result.stdout.splitlines():
        match = INSTRUCTION_LINE.match(line)
        if match is None:
            continue
        address = int(match.group(1), 16)
        owner = _owner_or_none(table, address)
        if owner is None:
            continue
        body = bodies[owner]
        mnemonic = match.group(2)
        fields = (match.group(3) or "").split(",")
        where = f"{address:#x} {mnemonic} {','.join(fields)}"

        # Step 1: the stack pointer moves only by immediates
        if fields[0] == STACK_POINTER and mnemonic not in STORE_MNEMONICS:
            immediate_form = mnemonic in SP_ADJUST_MNEMONICS and (
                len(fields) == 2 or fields[1] == STACK_POINTER
            )
            if immediate_form:
                depth[owner] = depth.get(owner, 0) - int(fields[-1])
                body.prologue = max(body.prologue, depth[owner])
            elif mnemonic not in DIRECT_JUMPS:
                body.problems.append(f"{where}: stack pointer set from a register")

        # Step 2: calls and jumps that leave the function
        if mnemonic in DIRECT_JUMPS:
            target = int(fields[-1], 16)
            if not body.start <= target < body.end:
                body.callees.add(table.owner(target))
            if mnemonic == "jal" and fields[0] == RETURN_REGISTER:
                body.calls = True
        elif mnemonic == "c.jalr" or (
            mnemonic == "jalr" and fields[0] == RETURN_REGISTER
        ):
            body.calls = True
            body.problems.append(f"{where}: indirect call")
        elif mnemonic == "c.jr" and fields[0] != RETURN_REGISTER:
            body.problems.append(f"{where}: indirect jump")
        elif mnemonic == "jalr" and fields[1] != f"0({RETURN_REGISTER})":
            body.problems.append(f"{where}: indirect jump")
        elif mnemonic in RETURN_MNEMONICS:
            depth[owner] = 0
    return bodies


def _owner_or_none(table: FunctionTable, address: int) -> str | None:
    """Name the function holding an address, or None for padding between functions.

    Parameters
    ----------
    table : FunctionTable
        The program's functions.
    address : int
        An instruction address.

    Returns
    -------
    str | None
        The function's name.
    """

    try:
        return table.owner(address)
    except ProfileError:
        return None


def _frame_of(name: str, bodies: dict[str, _Body], sized: dict[str, int]) -> StackFrame:
    """Give one function's frame, from the call graph when GCC sized it, else from its prologue.

    Parameters
    ----------
    name : str
        The function.
    bodies : dict[str, _Body]
        What each function's instructions say.
    sized : dict[str, int]
        Frame bytes GCC wrote, by function name.

    Returns
    -------
    StackFrame
        The function's frame and where its size came from.

    Raises
    ------
    ProfileError
        If the function has no fixed frame, or calls without a frame to save `ra` in.
    """

    body = bodies[name]
    if body.problems:
        raise ProfileError(f"{name} has no fixed frame: {'; '.join(body.problems)}")
    if name in sized:
        return StackFrame(name, sized[name], SOURCE_CALLGRAPH)
    if body.prologue == 0 and body.calls:
        raise ProfileError(f"{name} calls but its prologue has no fixed frame")
    return StackFrame(name, body.prologue, SOURCE_PROLOGUE)


def _deepest_chain(
    name: str,
    bodies: dict[str, _Body],
    sized: dict[str, int],
    deepest: dict[str, StackBound],
    walking: list[str],
) -> StackBound:
    """Find the deepest call chain starting at a function, depth first.

    A function's chain is its own frame, then the deepest of its callees' chains.
    Each answer is kept in `deepest`, so a function called from many places is walked once.

    Parameters
    ----------
    name : str
        The function the chain starts at.
    bodies : dict[str, _Body]
        What each function's instructions say.
    sized : dict[str, int]
        Frame bytes GCC wrote, by function name.
    deepest : dict[str, StackBound]
        The chains already found, by their first function.
    walking : list[str]
        The functions between the root and this one, kept to refuse recursion.

    Returns
    -------
    StackBound
        The chain and its total.

    Raises
    ------
    ProfileError
        If the chain is recursive, or a function on it has no fixed frame.
    """

    # Step 1: a chain already found is reused
    if name in deepest:
        return deepest[name]

    # Step 2: a function already open on the way down is calling itself, however far round
    if name in walking:
        cycle = " -> ".join([*walking[walking.index(name) :], name])
        raise ProfileError(f"recursion: {cycle}")

    # Step 3: this function's frame, then the chain of each callee
    walking.append(name)
    frame = _frame_of(name, bodies, sized)
    below = [
        _deepest_chain(callee, bodies, sized, deepest, walking)
        for callee in sorted(bodies[name].callees)
    ]
    walking.pop()

    # Step 4: the deepest callee's chain goes under this frame
    worst = max(below, key=lambda bound: bound.total, default=StackBound([], 0))
    deepest[name] = StackBound([frame, *worst.frames], frame.size + worst.total)
    return deepest[name]


def _check_prologue_reader(bodies: dict[str, _Body], sized: dict[str, int]) -> None:
    """Check that the prologue reader, which sizes the library's frames, agrees with GCC on our own.

    Parameters
    ----------
    bodies : dict[str, _Body]
        What each function's instructions say.
    sized : dict[str, int]
        Frame bytes GCC wrote, by function name.

    Raises
    ------
    ProfileError
        If a function GCC sized has a different frame in its prologue.
    """

    for name, size in sized.items():
        if name in bodies and bodies[name].prologue != size:
            raise ProfileError(
                f"{name}: GCC sized the frame at {size} bytes but its prologue says {bodies[name].prologue}"
            )


def stack_bound(
    elf: Path, table: FunctionTable, stack_folder: Path, root: str
) -> StackBound:
    """Find the deepest call chain below a function and add up its frames.

    Our own functions take their frame from GCC's call graph, and library functions from the `addi sp,sp,-N` of their prologue in the linked program.
    Callees come from the linked program's `jal` targets for both.
    A tail call is counted as a call, which can only overestimate.

    Parameters
    ----------
    elf : Path
        The linked program, built from the same objects as the `.ci` files.
    table : FunctionTable
        The functions of `elf`.
    stack_folder : Path
        The folder of `.ci` files.
    root : str
        The function to start from.

    Returns
    -------
    StackBound
        The chain and its total, which does not include the frames above the root.

    Raises
    ------
    ProfileError
        If a function on the way has a dynamic frame, an indirect call or jump, a call but no fixed frame, or the chain is recursive, or the prologue reader disagrees with GCC on one of our own functions.
    """

    # Step 1: frames and bodies
    sized = read_callgraph_frames(stack_folder)
    bodies = _read_bodies(elf, table)
    if root not in bodies:
        raise ProfileError(f"{elf}: no function {root}")

    # Step 2: the prologue reader sizes the library's frames, so it is held to GCC's numbers on ours
    _check_prologue_reader(bodies, sized)

    # Step 3: the deepest chain below the root
    return _deepest_chain(root, bodies, sized, {}, [])
