"""`benchmark profile`: where a build's instructions go, its cycle range and its memory use, on a few windows."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import shutil
import signal
from pathlib import Path
from typing import Annotated

# External
import typer

# Internal
from score.benchmark.progress import Log
from score.benchmark.run import (
    COUNTER_EXTENSION,
    EXIT_INTERRUPTED,
    EXIT_REFUSED,
    ISA_SEPARATOR,
    PK_ROLE,
    _folder_name,
)
from score.benchmark.scope import read_series
from score.benchmark.tables import TABLES_FOLDER
from score.layout import BENCHMARK_FOLDER
from score.profile.cycles import load_weights
from score.profile.disassembly import read_mnemonics
from score.profile.errors import ProfileError
from score.profile.symbols import read_functions
from score.profile.tables import WindowNumbers, analyse_window, write_tables
from score.profile.units import ProfileSettings, is_done, run_units, units_of
from score.series_check import _find_series, write_rows
from score.spike_check import (
    CLEAN_BINARY,
    ISA,
    PK_VARIABLE,
    _sha256,
    _top_level,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The rv64 build folder when none is named, under the git top level
BUILD_FOLDER = Path("tspulse/build/rv64")

# The profile build's program, beside the clean one
PROFILE_BINARY = "tspulse-profile"

# The folder under the label's, in the data folder and in the tracked tables, that holds the profile
WORK_SUBFOLDER = "profile"

# The series when none is named
DEFAULT_SERIES = "560_YAHOO_id_10_Synthetic_tr_500_1st_893"
# The windows when none are named: 1 stands in for window 0, which has no window before it, 429 holds the series' only anomaly, and 908 is the last
DEFAULT_WINDOWS = "1,429,908"

# The separator between windows in `--windows`
WINDOW_SEPARATOR = ","

# Role the row file's hash is recorded under
ROWS_ROLE = "rows"

# The roles the clean and the profile builds' hashes are recorded under
CLEAN_ROLE = "clean"
PROFILE_ROLE = "profile"

# Where the clean build's pieces sit relative to its build folder: the generated weight table beside the build folders, the `.ci` files in `stack`, and the sources two folders up
GENERATED_TABLE = Path("..") / "generated" / "weights_table.c"
STACK_FOLDER = "stack"
SOURCE_FOLDER = Path("..") / ".."

# The vector extension, as a letter of the base ISA, as an extension of its own, and as a prefix of the `zv*` subsets
VECTOR_LETTER = "v"
VECTOR_SUBSET_PREFIX = "zv"
BASE_ISA_PREFIX = "rv64"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _parse_windows(text: str) -> list[int]:
    """Read `--windows`.

    Parameters
    ----------
    text : str
        Comma-separated window numbers.

    Returns
    -------
    list[int]
        The windows, each once, in the order given.

    Raises
    ------
    typer.BadParameter
        When a part is not a number, or is below 1: window 0 has no window before it, which the differential runs need.
    """

    windows: list[int] = []
    for part in text.split(WINDOW_SEPARATOR):
        if not part.strip().isdigit() or int(part) < 1:
            raise typer.BadParameter(
                f"{part!r} is not a window of at least 1; window 0 can't be profiled, since it has no window before it.",
                param_hint="--windows",
            )
        if int(part) not in windows:
            windows.append(int(part))
    return windows


def _check_isa(isa: str) -> None:
    """Refuse an ISA the profile can't handle.

    Parameters
    ----------
    isa : str
        Spike's ISA string.

    Raises
    ------
    typer.BadParameter
        When it is not rv64, lacks `zicntr`, without which `rdinstret` traps, or has the vector extension, whose builds have no weight table.
    """

    parts = isa.split(ISA_SEPARATOR)
    if not parts[0].startswith(BASE_ISA_PREFIX):
        raise typer.BadParameter(
            f"{isa!r} is not rv64; profile covers rv64 builds only.",
            param_hint="--isa",
        )
    if COUNTER_EXTENSION not in parts:
        raise typer.BadParameter(
            f"{isa!r} lacks {COUNTER_EXTENSION}, without which `rdinstret` traps.",
            param_hint="--isa",
        )
    base = parts[0].removeprefix(BASE_ISA_PREFIX)
    if (
        VECTOR_LETTER in base
        or VECTOR_LETTER in parts[1:]
        or any(part.startswith(VECTOR_SUBSET_PREFIX) for part in parts[1:])
    ):
        raise typer.BadParameter(
            f"{isa!r} has the vector extension; profile covers scalar rv64 builds only.",
            param_hint="--isa",
        )


def profile(
    label: Annotated[
        str,
        typer.Option(
            help="The name the tables go under: benchmark/results/LABEL/profile/."
        ),
    ],
    build: Annotated[
        Path | None,
        typer.Option(
            help="The rv64 build folder, holding `tspulse` and `tspulse-profile`. Defaults to tspulse/build/rv64 in this worktree."
        ),
    ] = None,
    series: Annotated[
        str, typer.Option(help="The series, by CSV name without `.csv`.")
    ] = DEFAULT_SERIES,
    windows: Annotated[
        str,
        typer.Option(
            help="The windows to measure, comma-separated, each at least 1: window 0 can't be profiled, since it has no window before it."
        ),
    ] = DEFAULT_WINDOWS,
    jobs: Annotated[
        int, typer.Option(min=1, help="Spike runs at once. Each is heavy.")
    ] = 1,
    isa: Annotated[
        str,
        typer.Option(
            help="Spike's ISA string. Needs `zicntr`, and vector builds are refused."
        ),
    ] = ISA,
    work: Annotated[
        Path | None,
        typer.Option(
            help="The folder for the row file and the runs' files. Defaults to profile/ under the label's folder in the data folder."
        ),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
) -> None:
    """Profile a build on a few windows under Spike, skipping runs already done, and write six tables.

    Each window takes three Spike runs: the clean build over the window and its predecessor, over the predecessor alone, and the profile build.
    The tables are kernels.csv, modules.csv, mix.csv, cycles.csv, memory.csv and traffic.csv.

    Parameters
    ----------
    label : str
        The label the tables go under.
    build : Path or None
        The rv64 build folder.
    series : str
        The series' CSV name without `.csv`.
    windows : str
        Comma-separated windows, each at least 1.
    jobs : int
        Spike runs at once.
    isa : str
        Spike's ISA string.
    work : Path or None
        The work folder.
    data : Path or None
        The folder with the data.

    Raises
    ------
    typer.Exit
        With code 1 when a run fails or a table can't be made, with `EXIT_REFUSED` when something needed is missing, before any run, and with `EXIT_INTERRUPTED` after Ctrl-C or SIGTERM.
    """

    # SIGTERM otherwise ends the parent without unwinding, and its Spikes keep running. As Ctrl-C it kills them first
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    # Step 1: the options
    _folder_name(label, "--label")
    measured = _parse_windows(windows)
    _check_isa(isa)

    # Step 2: the data folder, Spike, pk and the binaries
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(EXIT_REFUSED)
    pk = os.environ.get(PK_VARIABLE)
    if not pk or shutil.which("spike") is None:
        typer.echo(
            f"Spike or ${PK_VARIABLE} is missing. Run this inside the dev shell, from `nix develop`.",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED)
    top = _top_level()
    build_folder = (build or top / BUILD_FOLDER).resolve()
    binaries = {
        CLEAN_ROLE: build_folder / CLEAN_BINARY,
        PROFILE_ROLE: build_folder / PROFILE_BINARY,
        PK_ROLE: Path(pk),
    }
    for path in binaries.values():
        if not path.is_file():
            typer.echo(f"Not found: {path}", err=True)
            raise typer.Exit(EXIT_REFUSED)

    # The tables read the `.ci` files and the weight table after the runs, and an old build folder may lack them, which would waste the Spike time
    stack_folder = build_folder / STACK_FOLDER
    weights_table = build_folder / GENERATED_TABLE
    if not any(stack_folder.rglob("*.ci")):
        typer.echo(
            f"No .ci file under {stack_folder}: run `make TARGET=rv64` in tspulse/.",
            err=True,
        )
        raise typer.Exit(EXIT_REFUSED)
    if not weights_table.is_file():
        typer.echo(f"Not found: {weights_table}", err=True)
        raise typer.Exit(EXIT_REFUSED)

    # Step 3: the series, and that every window exists in it
    known = {one.stem: one for one in read_series(top)}
    if series not in known:
        raise typer.BadParameter(
            f"Not a working series: {series}", param_hint="--series"
        )
    last = known[series].windows - 1
    if max(measured) > last:
        raise typer.BadParameter(
            f"{series} has windows 0 to {last}.", param_hint="--windows"
        )

    # Step 4: the work folder and the row file, reused when it exists
    work_folder = (work or data / BENCHMARK_FOLDER / label / WORK_SUBFOLDER).resolve()
    work_folder.mkdir(parents=True, exist_ok=True)
    log = Log(work_folder / "profile.log")
    rows = work_folder / f"{series}.rows"
    if not rows.is_file():
        csv_path = _find_series(data, series)
        if csv_path is None:
            typer.echo(f"No series {series} under {data}.", err=True)
            raise typer.Exit(EXIT_REFUSED)
        try:
            write_rows(csv_path, rows)
        except (OSError, ValueError, KeyError) as error:
            typer.echo(f"Cannot read {csv_path}: {error}", err=True)
            raise typer.Exit(EXIT_REFUSED) from error
    settings = ProfileSettings(
        clean=binaries[CLEAN_ROLE],
        profile=binaries[PROFILE_ROLE],
        pk=binaries[PK_ROLE],
        isa=isa,
        rows=rows,
        work=work_folder,
        channels=known[series].channels,
        sha256={
            **{role: _sha256(path) for role, path in binaries.items()},
            ROWS_ROLE: _sha256(rows),
        },
    )

    # Step 5: the runs
    units = units_of(measured)
    pending = [unit for unit in units if not is_done(unit, settings)]
    try:
        failed = run_units(pending, len(units) - len(pending), jobs, settings, log)
    except KeyboardInterrupt:
        log.line(
            "Interrupted: every program started was stopped; run the same command again to resume."
        )
        raise typer.Exit(EXIT_INTERRUPTED) from None
    if failed:
        log.line(f"{failed} runs failed, so no tables were written.")
        raise typer.Exit(1)

    # Step 6: the tables, which read only the runs' files
    tables = top / TABLES_FOLDER / label / WORK_SUBFOLDER
    try:
        functions = read_functions(binaries[CLEAN_ROLE])
        mnemonics = read_mnemonics(binaries[CLEAN_ROLE])
        weights = load_weights()
        numbers: list[WindowNumbers] = [
            analyse_window(window, settings, functions, mnemonics, weights)
            for window in measured
        ]
        write_tables(
            tables,
            numbers,
            binaries[CLEAN_ROLE],
            functions,
            weights_table,
            stack_folder,
            build_folder / SOURCE_FOLDER,
        )
    except (OSError, ProfileError) as error:
        log.line(f"cannot write the tables: {error}")
        raise typer.Exit(1) from error
    log.line(f"tables written to {tables}")
