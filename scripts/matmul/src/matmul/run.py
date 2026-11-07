"""Build and run the matrix multiply program for each setup, and turn the results into tables."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import json
import os
import platform
import re
import socket
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Internal
from matmul.models import Failure, Result, Setup, Shape


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


SOURCES = ("main.c", "matmul.c", "random.c")

# Vector and fused builds round differently from the host, so their sums only agree this closely
TOLERANCE = 1e-5

SHAPE_LINE = re.compile(
    r"shape rows=(\d+) in=(\d+) out=(\d+) calls=(\d+) macs=(\d+) instret=(\d+) abssum=(\S+)"
)
TOTAL_LINE = re.compile(r"total macs=(\d+) instret=(\d+)")

# GCC names the file and line of every loop it vectorized
KERNEL_VECTORIZED = re.compile(
    r"(^|/)matmul\.c:\d+:\d+: optimized: loop vectorized", re.MULTILINE
)

# Checks every RISC-V build's numbers, so it runs with whatever flags make it fast
HOST = Setup("host", "gcc", ("-O2",), None)

# Spike's default ISA has the counters, but an explicit --isa drops them unless zicntr is named, and rdinstret then traps
SETUPS = (
    Setup(
        "rv64gc-O2",
        "riscv64-none-elf-gcc",
        ("-march=rv64gc", "-mabi=lp64d", "-O2"),
        "rv64gc_zicntr",
    ),
    Setup(
        "rv64gc-O3",
        "riscv64-none-elf-gcc",
        ("-march=rv64gc", "-mabi=lp64d", "-O3"),
        "rv64gc_zicntr",
    ),
    Setup(
        "rv64gcv-O2",
        "riscv64-none-elf-gcc",
        ("-march=rv64gcv", "-mabi=lp64d", "-O2"),
        "rv64gcv_zicntr",
        vector=True,
    ),
    Setup(
        "rv64gcv-O3",
        "riscv64-none-elf-gcc",
        ("-march=rv64gcv", "-mabi=lp64d", "-O3"),
        "rv64gcv_zicntr",
        vector=True,
    ),
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


# Tools and the machine


def command(
    arguments: list[str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a tool and capture its output, turning a missing tool into a `Failure`.

    Parameters
    ----------
    arguments : list[str]
        The tool and its arguments, never joined into a shell string, because paths may contain spaces.
    cwd : Path | None
        Where to run it.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The finished process, whatever its exit code.

    Raises
    ------
    Failure
        When the tool isn't on `PATH`.
    """

    try:
        return subprocess.run(arguments, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise Failure(
            f"{arguments[0]} is not on PATH. Run this inside the dev shell, from `nix develop`."
        ) from error


def first_line(arguments: list[str]) -> str:
    """The first line a tool prints, on stdout or stderr, for recording its version."""

    process = command(arguments)
    return (process.stdout or process.stderr).splitlines()[0]


def cpu_model() -> str:
    """The CPU's model name, or an empty string where `/proc/cpuinfo` doesn't have one."""

    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def repository(source: Path) -> tuple[str, bool]:
    """The commit the source is at, and whether the tree has changes on top of it."""

    commit = command(["git", "rev-parse", "HEAD"], cwd=source).stdout.strip()
    dirty = bool(command(["git", "status", "--porcelain"], cwd=source).stdout.strip())
    return commit, dirty


# Units


def parse(text: str, setup: str, wall_s: float) -> Result:
    """Read the program's output.

    Parameters
    ----------
    text : str
        Everything the run printed.
    setup : str
        The setup's name.
    wall_s : float
        Seconds the run took.

    Returns
    -------
    Result
        The parsed lines.

    Raises
    ------
    Failure
        When the output has no shape lines or no total line.
    """

    shapes = [
        Shape(*(int(group) for group in match.groups()[:6]), abssum=float(match[7]))
        for match in SHAPE_LINE.finditer(text)
    ]
    total = TOTAL_LINE.search(text)
    if not shapes or total is None:
        raise Failure(f"{setup}: the output has no shape or total lines.")
    return Result(setup, wall_s, shapes, int(total[1]), int(total[2]))


def compare(result: Result, host: Result) -> None:
    """Check that a build's sums agree with the host's.

    Parameters
    ----------
    result : Result
        The build to check.
    host : Result
        The host build's result.

    Raises
    ------
    Failure
        When the shapes differ, or a sum differs by more than `TOLERANCE`, naming the first shape that does.
    """

    def sizes(item: Result) -> list[tuple[int, int, int, int]]:
        return [(s.rows, s.in_, s.out, s.calls) for s in item.shapes]

    if sizes(result) != sizes(host):
        raise Failure(f"{result.setup}: the shapes don't match the host's.")

    for shape, expected in zip(result.shapes, host.shapes, strict=True):
        difference = abs(shape.abssum - expected.abssum) / abs(expected.abssum)
        # Written as a negation, so a NaN sum fails instead of slipping through
        if not difference <= TOLERANCE:
            raise Failure(
                f"{result.setup}: shape rows={shape.rows} in={shape.in_} out={shape.out} has abssum {shape.abssum!r}, "
                f"the host has {expected.abssum!r}, {difference:.2g} apart relative."
            )


def run_unit(
    setup: Setup, source: Path, folder: Path, machine: str, host: Result | None
) -> Result:
    """Build and run one setup, writing its files, with `result.json` last.

    Parameters
    ----------
    setup : Setup
        What to build and how to run it.
    source : Path
        The C folder.
    folder : Path
        The unit's folder.
    machine : str
        This computer's label.
    host : Result | None
        The host's result to check against, or `None` for the host unit itself.

    Returns
    -------
    Result
        What the unit measured.

    Raises
    ------
    Failure
        When the build, the run or a check fails. The unit then has no `result.json`.
    """

    # A forced rerun that fails must not leave the previous result marking the unit as done
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "result.json").unlink(missing_ok=True)
    binary = folder / setup.binary
    started = datetime.now().astimezone().isoformat(timespec="seconds")

    # Build

    flags = list(setup.flags)
    if setup.vector:
        flags.append("-fopt-info-vec-optimized")
    compile_command = [
        setup.compiler,
        *flags,
        "-o",
        str(binary),
        *(str(source / name) for name in SOURCES),
    ]
    built = command(compile_command)
    (folder / "compile.log").write_text(built.stderr)
    if built.returncode != 0:
        raise Failure(f"{setup.name}: the build failed. See {folder / 'compile.log'}.")

    # A vector build whose kernel stayed scalar would make the low estimate wrong
    if setup.vector and not KERNEL_VECTORIZED.search(built.stderr):
        raise Failure(
            f"{setup.name}: GCC didn't vectorize the kernel's loop. See {folder / 'compile.log'}."
        )

    # Run

    if setup.isa is None:
        run_command = [str(binary)]
    else:
        pk = os.environ.get("RISCV_PK")
        if not pk:
            raise Failure(
                "RISCV_PK is not set. Run this inside the dev shell, from `nix develop`."
            )
        run_command = ["spike", f"--isa={setup.isa}", pk, str(binary)]

    begin = time.perf_counter()
    try:
        ran = subprocess.run(
            run_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except FileNotFoundError as error:
        raise Failure(
            f"{run_command[0]} is not on PATH. Run this inside the dev shell, from `nix develop`."
        ) from error
    wall_s = time.perf_counter() - begin
    (folder / setup.raw_output).write_text(ran.stdout)

    # Record how the unit was built and run, even when a check below fails

    commit, dirty = repository(source)
    options = {
        "setup": setup.name,
        "compile_command": compile_command,
        "run_command": run_command,
        "isa": setup.isa,
        "compiler_version": first_line([setup.compiler, "--version"]),
        "spike_version": None if setup.isa is None else first_line(["spike", "--help"]),
        "pk": None if setup.isa is None else run_command[2],
        "machine": machine,
        "hostname": socket.gethostname(),
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "kernel": platform.release(),
        "commit": commit,
        "dirty": dirty,
        "started": started,
        "wall_s": wall_s,
    }
    (folder / "options.json").write_text(json.dumps(options, indent=2) + "\n")

    if ran.returncode != 0:
        raise Failure(
            f"{setup.name}: the run exited with {ran.returncode}. See {folder / setup.raw_output}."
        )

    # Check

    result = parse(ran.stdout, setup.name, wall_s)
    if host is not None:
        compare(result, host)

    (folder / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


# Tables


def load_results(folder: Path) -> list[Result]:
    """Read every RISC-V unit's result in one machine's folder, ordered by setup name."""

    return [
        Result.load(path)
        for path in sorted(folder.glob("*/result.json"))
        if path.parent.name != HOST.name
    ]


def group_passes(series: Path) -> dict[str, int]:
    """Add up the passes of each series group.

    Parameters
    ----------
    series : Path
        `benchmark/series.csv`.

    Returns
    -------
    dict[str, int]
        Passes for `U working`, `U screen`, `U held`, `M working` and `M held`, in that order.

    Raises
    ------
    Failure
        When the table is missing.
    """

    if not series.exists():
        raise Failure(f"{series} is missing. Pass --series, or run select-series.")

    groups: dict[str, int] = dict.fromkeys(
        ("U working", "U screen", "U held", "M working", "M held"), 0
    )
    with series.open(newline="") as file:
        for row in csv.DictReader(file):
            groups[f"{row['kind']} {row['group']}"] += int(row["passes"])
            if row["screen_passes"]:
                groups["U screen"] += int(row["screen_passes"])
    return groups


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write a table with Unix line endings, so reruns give the same bytes."""

    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_results(folder: Path, machine: str, results: list[Result]) -> None:
    """Write `results.csv`: one row per setup and shape, then a `total` row per setup with blank sizes."""

    rows: list[list[object]] = []
    for result in results:
        for s in result.shapes:
            rows.append(
                [
                    machine,
                    result.setup,
                    s.rows,
                    s.in_,
                    s.out,
                    s.calls,
                    s.macs,
                    s.instret,
                    s.instret / s.macs,
                    "",
                    "",
                ]
            )
        calls = sum(s.calls for s in result.shapes)
        rows.append(
            [
                machine,
                result.setup,
                "",
                "",
                "",
                calls,
                result.macs,
                result.instret,
                result.instret / result.macs,
                result.wall_s,
                result.spike_ips,
            ]
        )
    header = [
        "machine",
        "setup",
        "rows",
        "in",
        "out",
        "calls",
        "macs",
        "instret",
        "instret_per_mac",
        "wall_s",
        "spike_ips",
    ]
    write_csv(folder / "results.csv", header, rows)


def projection(
    results: list[Result], groups: dict[str, int]
) -> list[tuple[str, str, int, int, float]]:
    """Project Spike core-hours per setup and series group, a floor that covers matrix multiplies only.

    Parameters
    ----------
    results : list[Result]
        The RISC-V units' results.
    groups : dict[str, int]
        Passes per group, from `group_passes`.

    Returns
    -------
    list[tuple[str, str, int, int, float]]
        Setup, group, passes, instructions and core-hours.
    """

    return [
        (
            result.setup,
            group,
            passes,
            passes * result.instret,
            passes * result.instret / result.spike_ips / 3600,
        )
        for result in results
        for group, passes in groups.items()
    ]


def write_projection(
    folder: Path, machine: str, rows: list[tuple[str, str, int, int, float]]
) -> None:
    """Write `projection.csv` from `projection`'s rows."""

    header = ["machine", "setup", "group", "passes", "instret", "core_hours"]
    write_csv(folder / "projection.csv", header, [[machine, *row] for row in rows])
