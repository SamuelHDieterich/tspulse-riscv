"""The `check-spike` command: does the RISC-V build under Spike compute exactly what the host build does?

Each unit runs one window on the host and under Spike and compares the files the two write, byte for byte.
A reference unit runs the dump build on one of the 7 reference windows and compares every inner output; a series unit runs the clean build on the middle window of a working series and compares its three errors.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import filecmp
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from score import rows, tdump
from score.capture import flat
from score.check import FLAT_FOLDER
from score.layout import DATASETS_FOLDER, SPIKE_CHECK_FOLDER
from score.processes import Processes
from score.reference import WINDOWS
from score.series_check import _find_series, write_rows
from score.spike_results import (
    RESULT_FILE,
    RESULTS_FILE,
    UnitKind,
    UnitResult,
    read_result,
    write_result,
    write_results_csv,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The ISA string the Makefile and `spike-matmul` run Spike with
ISA = "rv64gc_zicntr"

# The environment variable the dev shell puts pk's path in, since Spike does not search PATH for it
PK_VARIABLE = "RISCV_PK"

# The two builds each build folder holds
CLEAN_BINARY = "tspulse"
DUMP_BINARY = "tspulse-dump"

# The roles the binaries' hashes are recorded under
HOST_ROLE = "host"
RV64_ROLE = "rv64"

# Where the row files, errors and dumps sit while a unit runs, under the git top level
WORK_FOLDER = Path("tspulse/build/check-spike")

# The benchmark's series table, under the git top level
SERIES_TABLE = Path("benchmark/series.csv")
WORKING_GROUP = "working"

# The folder a series unit's label starts with
SERIES_FOLDER = "series"

# What each side's files are called in a unit's folders
HOST_ERRORS = "host.errors.csv"
SPIKE_ERRORS = "spike.errors.csv"
HOST_DUMP = "host.tdump"
SPIKE_DUMP = "spike.tdump"
HOST_STDERR = "host.stderr"
SPIKE_STDERR = "spike.stderr"

# Spike's `-g` histogram: a header, then one line per program counter in hex with how many instructions ran there
HISTOGRAM_HEADER = "PC Histogram size:"
HISTOGRAM_LINE = re.compile(r"^([0-9a-f]+) ([0-9]+)$")

# pk loads at the start of RAM and maps itself high, so every address from here up is pk's and every address below is the program's
PK_BASE = 0x80000000

# Read in 1 MiB pieces, so hashing a binary never holds it whole
HASH_CHUNK = 1 << 20


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Stage(StrEnum):
    """Which units a call runs.

    Attributes
    ----------
    REFERENCE : str
        The 7 reference windows, with the dump build.
    SERIES : str
        One window per working series, with the clean build.
    ALL : str
        Both.
    """

    REFERENCE = "reference"
    SERIES = "series"
    ALL = "all"


@dataclass(frozen=True)
class Unit:
    """One window to run on both builds.

    Attributes
    ----------
    label : str
        `<series stem>/w<index>` or `flat/w0` for a reference window, as `check-model` labels them, and `series/<stem>` for a series' window. Also the unit's folder, under the work folder and the machine's folder.
    kind : UnitKind
        `reference` or `series`.
    series : str
        The series' CSV name without `.csv`, or `flat`.
    window : int
        The window's index in the series.
    channels : int
        How many channels the series has, which orders the units by cost.
    csv_path : Path or None
        The series' CSV, None for the flat window.
    rows_path : Path
        The `.rows` file both builds read.
    """

    label: str
    kind: UnitKind
    series: str
    window: int
    channels: int
    csv_path: Path | None
    rows_path: Path


@dataclass(frozen=True)
class Settings:
    """What every unit of one call shares.

    Attributes
    ----------
    host_build : Path
        The host build folder.
    rv64_build : Path
        The rv64 build folder.
    pk : Path
        The proxy kernel Spike runs the program under.
    work : Path
        The work folder.
    machine : str
        The label of this computer.
    machine_folder : Path
        Where the units' results go.
    jobs : int
        How many units run at once.
    commit : str
        `git rev-parse HEAD`.
    tree : str
        `git rev-parse HEAD^{tree}`.
    dirty : bool
        Whether the worktree has uncommitted changes.
    sha256 : dict[str, str]
        The four binaries' hashes by role.
    processes : Processes
        The programs started so far.
    """

    host_build: Path
    rv64_build: Path
    pk: Path
    work: Path
    machine: str
    machine_folder: Path
    jobs: int
    commit: str
    tree: str
    dirty: bool
    sha256: dict[str, str]
    processes: Processes = field(default_factory=Processes)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _git(top: Path, *arguments: str) -> str:
    """Run a git command in the worktree and return its output.

    Parameters
    ----------
    top : Path
        The worktree's top level.
    *arguments : str
        The git subcommand and its arguments.

    Returns
    -------
    str
        Its stdout, stripped.

    Raises
    ------
    typer.Exit
        With code 1 when git fails.
    """

    process = subprocess.run(
        ["git", *arguments], cwd=top, capture_output=True, text=True
    )
    if process.returncode:
        typer.echo(
            f"git {' '.join(arguments)} failed: {process.stderr.strip()}", err=True
        )
        raise typer.Exit(1)
    return process.stdout.strip()


def _top_level() -> Path:
    """Find the top level of the git worktree the command runs in.

    Returns
    -------
    Path
        The worktree's top level, so the defaults don't depend on the current folder.

    Raises
    ------
    typer.Exit
        With code 1 when the current folder is not inside a git worktree.
    """

    return Path(_git(Path.cwd(), "rev-parse", "--show-toplevel"))


def _sha256(path: Path) -> str:
    """Hash a file.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    str
        Its SHA-256, in hex.
    """

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _instruction_totals(histogram: str) -> tuple[int, int]:
    """Sum Spike's `-g` histogram into the program's and pk's instructions.

    Parameters
    ----------
    histogram : str
        Spike's stderr, which also holds whatever the program printed there.

    Returns
    -------
    tuple[int, int]
        Instructions at addresses below `PK_BASE`, then at `PK_BASE` and above.
    """

    program = pk = 0
    for line in histogram.splitlines():
        match = HISTOGRAM_LINE.match(line)
        if match is None:
            continue
        count = int(match.group(2))
        if int(match.group(1), 16) < PK_BASE:
            program += count
        else:
            pk += count
    return program, pk


def _program_message(stderr_path: Path) -> str:
    """Read what a failed program printed, without Spike's histogram.

    Parameters
    ----------
    stderr_path : Path
        The file its stderr went to.

    Returns
    -------
    str
        The lines that are not histogram lines, joined with `; `.
    """

    lines = stderr_path.read_text(errors="replace").splitlines()
    kept = [
        line
        for line in lines
        if line.strip()
        and not HISTOGRAM_LINE.match(line)
        and not line.startswith(HISTOGRAM_HEADER)
    ]
    return "; ".join(kept)


def _first_difference(host_path: Path, spike_path: Path) -> str:
    """Find the first record, in the host dump's order, whose bytes differ under Spike.

    Parameters
    ----------
    host_path : Path
        The host's dump.
    spike_path : Path
        Spike's dump.

    Returns
    -------
    str
        The record's name, or what else sets the two files apart when no record does.
    """

    # Step 1: both dumps, of which a truncated one already names the problem
    try:
        host = tdump.read(host_path)
        spike = tdump.read(spike_path)
    except (OSError, ValueError) as error:
        return f"unreadable dump: {error}"

    # Step 2: the host's records in order. Comparing raw bytes tells NaNs and signed zeros apart, so this agrees with the byte-for-byte file check
    for name, values in host.items():
        other = spike.get(name)
        if (
            other is None
            or other.dtype != values.dtype
            or other.shape != values.shape
            or other.tobytes() != values.tobytes()
        ):
            return name

    # Step 3: a record only Spike wrote, or the same records in another order
    for name in spike:
        if name not in host:
            return name
    return "the same records in another order"


def _select_units(
    data: Path,
    top: Path,
    work: Path,
    stage: Stage,
    references: list[str],
    series: list[str],
) -> list[Unit]:
    """List the units a call asks for, in reference then `series.csv` order.

    Parameters
    ----------
    data : Path
        The folder with the data.
    top : Path
        The worktree's top level.
    work : Path
        The work folder, which holds the row files.
    stage : Stage
        Which stages to run.
    references : list[str]
        Reference labels to keep, or all when empty.
    series : list[str]
        Series stems to keep, or all working series when empty.

    Returns
    -------
    list[Unit]
        The units.

    Raises
    ------
    typer.BadParameter
        When a label or stem is not one the stage has.
    typer.Exit
        With code 1 when the series table or a series' CSV is missing.
    """

    # Step 1: the series table, which gives every series' archive, channels and window count
    table_path = top / SERIES_TABLE
    try:
        with table_path.open(newline="") as file:
            table = list(csv.DictReader(file))
    except OSError as error:
        typer.echo(f"Cannot read {table_path}: {error}", err=True)
        raise typer.Exit(1) from error
    channels_by_name = {row["file_name"]: int(row["channels"]) for row in table}

    # Step 2: the reference windows, labelled as `check-model` labels them
    reference_units = []
    for reference_series, index in WINDOWS:
        folder = reference_series.stem if reference_series else FLAT_FOLDER
        reference_units.append(
            Unit(
                label=f"{folder}/w{index}",
                kind="reference",
                series=folder,
                window=index,
                channels=channels_by_name[reference_series.name]
                if reference_series
                else flat().raw.shape[1],
                csv_path=data / DATASETS_FOLDER / reference_series
                if reference_series
                else None,
                rows_path=work / f"{folder}.rows",
            )
        )
    known = {unit.label for unit in reference_units}
    unknown = sorted(set(references) - known)
    if unknown:
        raise typer.BadParameter(
            f"Not a reference window: {', '.join(unknown)}. They are {', '.join(sorted(known))}.",
            param_hint="--reference",
        )

    # Step 3: the working series. The middle window also exercises reading and skipping the first half of the rows
    working = [row for row in table if row["group"] == WORKING_GROUP]
    stems = [Path(row["file_name"]).stem for row in working]
    unknown = sorted(set(series) - set(stems))
    if unknown:
        raise typer.BadParameter(
            f"Not a working series in {SERIES_TABLE}: {', '.join(unknown)}",
            param_hint="--series",
        )
    series_units = []
    if stage in (Stage.SERIES, Stage.ALL):
        for row, stem in zip(working, stems, strict=True):
            if series and stem not in series:
                continue
            csv_path = _find_series(data, stem)
            if csv_path is None:
                typer.echo(f"No series {stem} under {data}.", err=True)
                raise typer.Exit(1)
            series_units.append(
                Unit(
                    label=f"{SERIES_FOLDER}/{stem}",
                    kind="series",
                    series=stem,
                    window=int(row["windows"]) // 2,
                    channels=int(row["channels"]),
                    csv_path=csv_path,
                    rows_path=work / f"{stem}.rows",
                )
            )

    # Step 4: the stages asked for
    if stage == Stage.SERIES:
        reference_units = []
    elif references:
        reference_units = [unit for unit in reference_units if unit.label in references]
    return reference_units + series_units


def _write_rows(units: list[Unit]) -> None:
    """Write the row files the units need and the work folder lacks, before any unit starts, so no two units write the same file.

    Parameters
    ----------
    units : list[Unit]
        The units about to run.

    Raises
    ------
    typer.Exit
        With code 1 when a CSV cannot be turned into a row file.
    """

    for unit in units:
        # A row file is only replaced whole, so one that exists is complete and is kept for reuse
        if unit.rows_path.is_file():
            continue
        if unit.csv_path is None:
            window = flat()
            rows.write(unit.rows_path, window.raw, window.mean, window.std)
            continue
        try:
            write_rows(unit.csv_path, unit.rows_path)
        except (OSError, ValueError, KeyError) as error:
            typer.echo(f"Cannot read {unit.csv_path}: {error}", err=True)
            raise typer.Exit(1) from error


def _dump_arguments(unit: Unit, dump: Path) -> list[str]:
    """The dump build's extra arguments for a unit.

    Parameters
    ----------
    unit : Unit
        The unit being run.
    dump : Path
        Where its dump goes.

    Returns
    -------
    list[str]
        `--dump-window` and `--dump` for a reference unit, which dumps its one window, and nothing for a series unit, whose clean build takes no dump options.
    """

    if unit.kind != "reference":
        return []
    return ["--dump-window", str(unit.window), "--dump", str(dump)]


def _run_unit(unit: Unit, settings: Settings) -> UnitResult | str | None:
    """Run one window on the host, then under Spike, compare their files and save the unit's result.

    Parameters
    ----------
    unit : Unit
        The window.
    settings : Settings
        What every unit shares.

    Returns
    -------
    UnitResult, str or None
        The result, saved as `result.json`; a message when a program can't start or fails, which saves no result so the unit reruns; or None when the call is stopping, which also saves nothing.
    """

    # Step 1: the unit's folders. An earlier result goes first, so a redo cut short leaves the unit to rerun
    result_folder = settings.machine_folder / unit.label
    scratch = settings.work / unit.label
    result_folder.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    (result_folder / RESULT_FILE).unlink(missing_ok=True)

    # An earlier run's files would otherwise sit next to this run's: its kept dumps beside an identical result, or its errors beside a failed run's `spike.stderr`
    for name in (HOST_DUMP, SPIKE_DUMP, HOST_ERRORS, SPIKE_ERRORS):
        (result_folder / name).unlink(missing_ok=True)

    # Step 2: the two command lines, the same program arguments on both sides. A reference unit dumps its one window
    binary = DUMP_BINARY if unit.kind == "reference" else CLEAN_BINARY
    host_errors, spike_errors = scratch / HOST_ERRORS, scratch / SPIKE_ERRORS
    host_dump, spike_dump = scratch / HOST_DUMP, scratch / SPIKE_DUMP
    window_arguments = ["--windows", f"{unit.window}:{unit.window + 1}"]

    host_command = [
        str(settings.host_build / binary),
        str(unit.rows_path),
        str(host_errors),
        *window_arguments,
        *_dump_arguments(unit, host_dump),
    ]
    spike_command = [
        "spike",
        "-g",
        f"--isa={ISA}",
        str(settings.pk),
        str(settings.rv64_build / binary),
        str(unit.rows_path),
        str(spike_errors),
        *window_arguments,
        *_dump_arguments(unit, spike_dump),
    ]

    # Step 3: the host, then Spike, each with its stderr kept. Spike's holds the `-g` histogram
    runs = []
    for side, command, stderr_path in (
        (HOST_ROLE, host_command, scratch / HOST_STDERR),
        (RV64_ROLE, spike_command, result_folder / SPIKE_STDERR),
    ):
        try:
            run = settings.processes.run(command, stderr_path)
        except OSError as error:
            return f"cannot run {command[0]}: {error}"
        if run is None:
            return None
        code, seconds = run
        if code:
            return f"{side} exit {code}: {_program_message(stderr_path)}"
        runs.append(seconds)
    host_seconds, spike_seconds = runs

    # Step 4: the errors byte for byte, then the dumps. Matching dumps are deleted, at about 50 MB per channel each; differing ones are kept with the result
    errors_match = filecmp.cmp(host_errors, spike_errors, shallow=False)
    dumps_match = None
    first_difference = None
    if unit.kind == "reference":
        dumps_match = filecmp.cmp(host_dump, spike_dump, shallow=False)
        if dumps_match:
            host_dump.unlink()
            spike_dump.unlink()
        else:
            first_difference = _first_difference(host_dump, spike_dump)
            shutil.move(host_dump, result_folder / HOST_DUMP)
            shutil.move(spike_dump, result_folder / SPIKE_DUMP)
    shutil.move(host_errors, result_folder / HOST_ERRORS)
    shutil.move(spike_errors, result_folder / SPIKE_ERRORS)

    # Step 5: the instruction totals and the result, whose file marks the unit done
    program, pk = _instruction_totals(
        (result_folder / SPIKE_STDERR).read_text(errors="replace")
    )
    result = UnitResult(
        label=unit.label,
        kind=unit.kind,
        series=unit.series,
        window=unit.window,
        channels=unit.channels,
        machine=settings.machine,
        host_command=shlex.join(host_command),
        spike_command=shlex.join(spike_command),
        isa=ISA,
        host_seconds=host_seconds,
        spike_seconds=spike_seconds,
        program_instructions=program,
        pk_instructions=pk,
        instructions=program + pk,
        errors_match=errors_match,
        dumps_match=dumps_match,
        first_difference=first_difference,
        commit=settings.commit,
        tree=settings.tree,
        dirty=settings.dirty,
        sha256=settings.sha256,
        jobs=settings.jobs,
    )
    write_result(result_folder, result)
    return result


def _run_units(units: list[Unit], settings: Settings) -> tuple[int, int]:
    """Run the pending units up to `jobs` at once, printing each and rewriting `results.csv` as it completes.

    Parameters
    ----------
    units : list[Unit]
        The units to run.
    settings : Settings
        What every unit shares.

    Returns
    -------
    tuple[int, int]
        How many units differed, then how many failed.

    Raises
    ------
    KeyboardInterrupt
        On Ctrl-C or SIGTERM, after every program started has been killed.
    """

    # Longest first, by channel count, so the 55-channel series don't finish last. The sort is stable, so ties keep their listed order
    ordered = sorted(units, key=lambda unit: unit.channels, reverse=True)
    differing = failed = 0
    with ThreadPoolExecutor(max_workers=settings.jobs) as pool:
        futures: dict[Future[UnitResult | str | None], Unit] = {}
        try:
            for unit in ordered:
                futures[pool.submit(_run_unit, unit, settings)] = unit

            # Only this thread prints and writes `results.csv`, so they have one writer
            for future in as_completed(futures):
                unit = futures[future]
                outcome = future.result()
                if isinstance(outcome, UnitResult):
                    differing += not outcome.identical
                    typer.echo(outcome.summary())
                elif isinstance(outcome, str):
                    failed += 1
                    typer.echo(f"{unit.label}: error: {outcome}")
                write_results_csv(settings.machine_folder)

        # On Ctrl-C or any error, no Spike is left running. A unit cut short wrote no `result.json`, so it reruns on the next call
        except BaseException:
            for future in futures:
                future.cancel()
            settings.processes.stop()
            raise
    return differing, failed


def _matching_result(path: Path, sha256: dict[str, str]) -> UnitResult | None:
    """Read a unit's stored result if it was made by the same four binaries.

    Parameters
    ----------
    path : Path
        The unit's `result.json`.
    sha256 : dict[str, str]
        This call's four binary hashes, by role.

    Returns
    -------
    UnitResult or None
        The stored result, or None when its recorded hashes differ, are missing or are malformed, so the unit reruns.

    Raises
    ------
    ValueError
        When the file isn't JSON, or its hashes match but its other fields don't fit `UnitResult`.
    OSError
        When the file can't be read.
    """

    stored = json.loads(path.read_text())
    if not isinstance(stored, dict) or stored.get("sha256") != sha256:
        return None
    return read_result(path)


def _check_spike(
    host_build: Annotated[
        Path,
        typer.Argument(
            help="The host build folder, holding `tspulse` and `tspulse-dump`."
        ),
    ],
    rv64_build: Annotated[
        Path,
        typer.Argument(
            help="The rv64 build folder, holding `tspulse` and `tspulse-dump`."
        ),
    ],
    stage: Annotated[
        Stage, typer.Option(help="The reference windows, the series, or both.")
    ] = Stage.ALL,
    reference: Annotated[
        list[str] | None,
        typer.Option(
            help="Only this reference window, labelled as `check-model` labels it, such as `flat/w0`. Repeatable."
        ),
    ] = None,
    series: Annotated[
        list[str] | None,
        typer.Option(
            help="Only this working series, by CSV name without `.csv`. Repeatable."
        ),
    ] = None,
    jobs: Annotated[
        int,
        typer.Option(
            min=1, help="Units run at once, each as a host run and then a Spike run."
        ),
    ] = 1,
    machine: Annotated[
        str | None,
        typer.Option(
            help="A label for this computer, so runs from several can share one folder. Defaults to the hostname."
        ),
    ] = None,
    redo: Annotated[
        bool,
        typer.Option(
            "--redo",
            help="Rerun every selected unit, even one already run with the same binaries.",
        ),
    ] = False,
    work: Annotated[
        Path | None,
        typer.Option(
            help="The folder for the row files, and for the errors and dumps while a unit runs. Defaults to tspulse/build/check-spike in this worktree."
        ),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
) -> None:
    """Run windows on the host build and under Spike on the rv64 build, and check they write the same bytes, skipping units already run with the same four binaries.

    Parameters
    ----------
    host_build : Path
        The host build folder.
    rv64_build : Path
        The rv64 build folder.
    stage : Stage
        Which stages to run.
    reference : list[str] or None
        Reference labels to keep.
    series : list[str] or None
        Series stems to keep.
    jobs : int
        How many units run at once.
    machine : str or None
        The label of this computer.
    redo : bool
        Whether to rerun units that already have a result, even from the same binaries.
    work : Path or None
        The work folder.
    data : Path or None
        The folder with the data.

    Raises
    ------
    typer.Exit
        With code 1 when something needed is missing, any program fails, or any unit asked for differs, whether run now or already on disk.
    """

    # SIGTERM otherwise ends the parent without unwinding, and its Spikes, which can run 45 min, keep running. As Ctrl-C it kills them first
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    # Step 1: the data folder, pk, Spike and the four binaries
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    pk = os.environ.get(PK_VARIABLE)
    if not pk:
        typer.echo(
            f"${PK_VARIABLE} is not set. Run this inside the dev shell, from `nix develop`.",
            err=True,
        )
        raise typer.Exit(1)
    if shutil.which("spike") is None:
        typer.echo(
            "spike is not on PATH. Run this inside the dev shell, from `nix develop`.",
            err=True,
        )
        raise typer.Exit(1)
    binaries = {
        f"{role}/{binary}": build.resolve() / binary
        for role, build in ((HOST_ROLE, host_build), (RV64_ROLE, rv64_build))
        for binary in (CLEAN_BINARY, DUMP_BINARY)
    }
    for path in [Path(pk), *binaries.values()]:
        if not path.is_file():
            typer.echo(f"Not found: {path}", err=True)
            raise typer.Exit(1)

    # Hashed once, so the skip decision and every saved result use the same value
    sha256 = {role: _sha256(path) for role, path in binaries.items()}

    # Step 2: the worktree's state, recorded with every result
    top = _top_level()
    commit = _git(top, "rev-parse", "HEAD")
    tree = _git(top, "rev-parse", "HEAD^{tree}")
    dirty = bool(_git(top, "status", "--porcelain"))

    # Step 3: the units, and which of them already have a result from these same four binaries
    work = (work or top / WORK_FOLDER).resolve()
    work.mkdir(parents=True, exist_ok=True)
    machine = machine or socket.gethostname()
    machine_folder = data / SPIKE_CHECK_FOLDER / machine
    machine_folder.mkdir(parents=True, exist_ok=True)
    units = _select_units(data, top, work, stage, reference or [], series or [])
    pending = []
    differing = 0
    for unit in units:
        result_path = machine_folder / unit.label / RESULT_FILE
        if redo or not result_path.is_file():
            pending.append(unit)
            continue
        try:
            done = _matching_result(result_path, sha256)
        except (OSError, ValueError) as error:
            typer.echo(f"Cannot read {result_path}: {error}", err=True)
            raise typer.Exit(1) from error
        if done is None:
            typer.echo(
                f"{unit.label}: binaries changed since the stored result, rerunning"
            )
            pending.append(unit)
            continue
        differing += not done.identical
        typer.echo(f"{done.summary()}, already done, skipped")

    # Step 4: the row files, then the units
    _write_rows(pending)
    settings = Settings(
        host_build=host_build.resolve(),
        rv64_build=rv64_build.resolve(),
        pk=Path(pk),
        work=work,
        machine=machine,
        machine_folder=machine_folder,
        jobs=jobs,
        commit=commit,
        tree=tree,
        dirty=dirty,
        sha256=sha256,
    )
    try:
        newly_differing, failed = _run_units(pending, settings)
    except KeyboardInterrupt:
        typer.echo(
            "Interrupted. Every Spike started was stopped; run the same command again to resume.",
            err=True,
        )
        raise typer.Exit(1) from None

    # Step 5: the table and the verdict
    count = write_results_csv(machine_folder)
    typer.echo(f"{machine_folder / RESULTS_FILE}: {count} units")
    if differing + newly_differing or failed:
        raise typer.Exit(1)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `check-spike`."""

    typer.run(_check_spike)
