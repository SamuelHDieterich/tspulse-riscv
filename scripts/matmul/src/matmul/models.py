"""The setups, the program's parsed output, and the error the command reports."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from dataclasses import dataclass
from pathlib import Path


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Failure(Exception):
    """A step failed in a way the user can act on, and the message says how."""


@dataclass(frozen=True)
class Setup:
    """One way of building and running the program.

    Attributes
    ----------
    name : str
        The unit's folder name.
    compiler : str
        The compiler on `PATH`.
    flags : tuple[str, ...]
        Every flag beside the sources and output.
    isa : str | None
        Spike's `--isa`, or `None` for the host build, which runs natively.
    vector : bool
        Whether the build targets the vector extension, so its kernel must vectorize.
    """

    name: str
    compiler: str
    flags: tuple[str, ...]
    isa: str | None
    vector: bool = False

    @property
    def binary(self) -> str:
        """The file the compiler writes."""

        return "matmul" if self.isa is None else "matmul.elf"

    @property
    def raw_output(self) -> str:
        """The file the program's raw output goes to."""

        return "run.out" if self.isa is None else "spike.out"


@dataclass(frozen=True)
class Shape:
    """One shape's line of program output.

    Attributes
    ----------
    rows : int
        The input rows.
    in_ : int
        The layer's input size.
    out : int
        The layer's output size.
    calls : int
        How many times one pass makes the call.
    macs : int
        Multiply-adds over all the calls.
    instret : int
        Instructions over all the calls, 0 on the host.
    abssum : float
        The sum of `|y|` after the last call.
    """

    rows: int
    in_: int
    out: int
    calls: int
    macs: int
    instret: int
    abssum: float


@dataclass(frozen=True)
class Result:
    """What one unit measured, as `result.json` stores it.

    Attributes
    ----------
    setup : str
        The setup's name.
    wall_s : float
        Seconds the program run took, including Spike and pk startup.
    shapes : list[Shape]
        One per shape, in the program's order.
    macs : int
        Multiply-adds over the whole pass.
    instret : int
        Instructions over the whole pass.
    """

    setup: str
    wall_s: float
    shapes: list[Shape]
    macs: int
    instret: int

    @property
    def spike_ips(self) -> float:
        """Instructions Spike ran per second of wall time."""

        return self.instret / self.wall_s

    @classmethod
    def load(cls, path: Path) -> "Result":
        """Read a `result.json`."""

        data = json.loads(path.read_text())
        data["shapes"] = [Shape(**shape) for shape in data["shapes"]]
        return cls(**data)
