"""What `check-spike` keeps of each unit, and the `results.csv` built from it.

Each unit leaves a `result.json` in its own folder under the machine's folder.
`results.csv` is rebuilt from every `result.json` there, so it can be read while a run goes on, and a run split across calls or computers still ends with one table per machine.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# A reference window run with the dump build, or one window of a working series run with the clean build
UnitKind = Literal["reference", "series"]

# The order units are listed in
UNIT_KINDS: tuple[UnitKind, ...] = get_args(UnitKind)

RESULT_FILE = "result.json"
RESULTS_FILE = "results.csv"

# The `result.json` fields that go into `results.csv`, in column order. The commands, hashes and machine stay in `result.json` only
COLUMNS = (
    "label",
    "kind",
    "series",
    "window",
    "channels",
    "host_seconds",
    "spike_seconds",
    "program_instructions",
    "pk_instructions",
    "instructions",
    "errors_match",
    "dumps_match",
    "first_difference",
    "commit",
    "tree",
    "dirty",
    "jobs",
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class UnitResult:
    """One unit's outcome: the same window run on the host and under Spike.

    Attributes
    ----------
    label : str
        The unit, such as `flat/w0` or `series/004_NAB_id_4_Facility_tr_1007_1st_1437`.
    kind : UnitKind
        `reference` for a reference window run with the dump build, `series` for a series' window run with the clean build.
    series : str
        The series' CSV name without `.csv`, or `flat` for the made-up window of zeros.
    window : int
        The window's index in the series.
    channels : int
        How many channels the series has.
    machine : str
        The label of the computer that ran the unit.
    host_command : str
        The host run's exact command line, shell-quoted.
    spike_command : str
        The Spike run's exact command line, shell-quoted.
    isa : str
        The ISA string Spike ran with.
    host_seconds : float
        The host run's wall time.
    spike_seconds : float
        The Spike run's wall time, histogram included.
    program_instructions : int
        Instructions at addresses below pk's, from Spike's histogram: startup, reading the skipped rows and the window.
    pk_instructions : int
        Instructions at pk's addresses and above.
    instructions : int
        Their sum.
    errors_match : bool
        Whether the two errors CSVs are byte-identical.
    dumps_match : bool or None
        Whether the two dumps are byte-identical, None for a series unit, which writes no dump.
    first_difference : str or None
        The first record, in the host dump's order, whose bytes differ, or None when the dumps match or there are none.
    commit : str
        `git rev-parse HEAD` of the worktree the command ran in.
    tree : str
        `git rev-parse HEAD^{tree}`, which a rebase with an empty diff leaves unchanged.
    dirty : bool
        Whether the worktree had uncommitted changes.
    sha256 : dict[str, str]
        The four binaries' hashes, by role: `host/tspulse`, `host/tspulse-dump`, `rv64/tspulse`, `rv64/tspulse-dump`.
    jobs : int
        How many units the call allowed at once, which bears on the wall times.
    """

    label: str
    kind: UnitKind
    series: str
    window: int
    channels: int
    machine: str
    host_command: str
    spike_command: str
    isa: str
    host_seconds: float
    spike_seconds: float
    program_instructions: int
    pk_instructions: int
    instructions: int
    errors_match: bool
    dumps_match: bool | None
    first_difference: str | None
    commit: str
    tree: str
    dirty: bool
    sha256: dict[str, str]
    jobs: int

    @property
    def identical(self) -> bool:
        """Whether every file compared is byte-identical: the errors, and the dumps for a reference unit.

        Returns
        -------
        bool
            True when the errors match and the dumps match or there are none.
        """

        return self.errors_match and self.dumps_match is not False

    def summary(self) -> str:
        """The unit's printed line: label, channels, Spike seconds, program instructions and the verdict.

        Returns
        -------
        str
            The line, ending in `identical`, or in `DIFFERENT` with what differs and the first differing record if any.
        """

        if self.identical:
            verdict = "identical"
        else:
            differing = [
                name
                for name, match in (
                    ("errors", self.errors_match),
                    ("dumps", self.dumps_match),
                )
                if match is False
            ]
            verdict = f"DIFFERENT ({' and '.join(differing)})"
            if self.first_difference is not None:
                verdict += f", first differing record {self.first_difference}"
        return (
            f"{self.label}: {self.channels} channels, Spike {self.spike_seconds:.1f} s, "
            f"{self.program_instructions:,} program instructions, {verdict}"
        )


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _replace(path: Path, text: str) -> None:
    """Write a text file through a temporary one, so a reader or a cut-short run never sees half of it.

    Parameters
    ----------
    path : Path
        The file to write.
    text : str
        Its whole content.
    """

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_result(folder: Path, result: UnitResult) -> None:
    """Save a unit's `result.json`, whose presence marks the unit as done.

    Parameters
    ----------
    folder : Path
        The unit's result folder.
    result : UnitResult
        What the unit found.
    """

    _replace(folder / RESULT_FILE, json.dumps(asdict(result), indent=2) + "\n")


def read_result(path: Path) -> UnitResult:
    """Read a unit's `result.json`.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    UnitResult
        What the unit found.

    Raises
    ------
    ValueError
        When the file isn't JSON or its fields don't match `UnitResult`'s.
    OSError
        When the file can't be read.
    """

    try:
        return UnitResult(**json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from error


def write_results_csv(machine_folder: Path) -> int:
    """Rebuild a machine's `results.csv` from every `result.json` under its folder.

    Parameters
    ----------
    machine_folder : Path
        The machine's folder.

    Returns
    -------
    int
        How many units the table holds.

    Raises
    ------
    ValueError
        When a `result.json` can't be read.
    """

    # Step 1: every result. A `--redo` unit may be deleting its `result.json` between the glob and the read, and is then left out
    results = []
    for path in machine_folder.rglob(RESULT_FILE):
        try:
            results.append(read_result(path))
        except FileNotFoundError:
            continue

    # Step 2: reference units first, then series, each by label, so the order doesn't depend on which unit finished first
    results.sort(key=lambda result: (UNIT_KINDS.index(result.kind), result.label))
    lines: list[list[object]] = []
    for result in results:
        fields = asdict(result)
        lines.append(["" if fields[name] is None else fields[name] for name in COLUMNS])

    # Step 3: through a string, so the file is replaced in one step
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(lines)
    _replace(machine_folder / RESULTS_FILE, buffer.getvalue())
    return len(results)
