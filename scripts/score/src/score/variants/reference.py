"""Check a variant's host dump build against the reference windows and keep the outcome in `reference.json`."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

# External
import typer

# Internal
from score import noise, rows
from score.benchmark.run import EXIT_REFUSED
from score.build_record import BuildRecord, BuildRecordError, check_binaries
from score.capture import flat
from score.check import CHECK_FOLDER, FLAT_FOLDER, _check_window
from score.layout import DATASETS_FOLDER, REFERENCE_FOLDER
from score.reference import WINDOWS
from score.series_check import write_rows
from score.spike_check import DUMP_BINARY
from score.variants.state import (
    REFERENCE_FILE,
    ReferenceCheck,
    ReferenceWindow,
    write_json,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The pooled noise table, in the reference folder
NOISE_TABLE = "noise.csv"


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def check_reference(
    host_folder: Path,
    host: BuildRecord,
    folder: Path,
    top: Path,
    data: Path,
    emit: Callable[[str], None],
) -> int:
    """Run the host dump build on the reference windows, compare each with its reference, and write `reference.json`.

    A window that differs is an outcome, not a failure: it is kept in `reference.json` for the variant's verdict.

    Parameters
    ----------
    host_folder : Path
        The variant's host build folder, holding `tspulse-dump`.
    host : BuildRecord
        The host build's record, whose `sha256` map goes into `reference.json`.
    folder : Path
        The label's folder.
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    emit : Callable[[str], None]
        Prints and logs one line.

    Returns
    -------
    int
        How many windows differ from the reference.

    Raises
    ------
    typer.Exit
        With code 1 when the noise table or a reference series can't be read, and with `EXIT_REFUSED` when `tspulse-dump` doesn't match its build record.
    """

    # Step 1: the dump build, checked against its record so the reason reaches the label's log, and the pooled noise
    try:
        check_binaries(host, host_folder, (DUMP_BINARY,))
    except BuildRecordError as error:
        emit(f"{DUMP_BINARY} can't be used: {error}")
        raise typer.Exit(EXIT_REFUSED) from error
    binary = host_folder.resolve() / DUMP_BINARY
    reference_folder = data / REFERENCE_FOLDER
    try:
        ratios = noise.read_pooled(reference_folder / NOISE_TABLE)
    except (OSError, ValueError) as error:
        emit(f"cannot read the pooled noise: {error}")
        raise typer.Exit(1) from error
    work = top / CHECK_FOLDER
    work.mkdir(parents=True, exist_ok=True)

    # Step 2: one run per window, the way `check-model` loops over them. Each series' row file is written the first time one of its windows comes up
    windows: list[ReferenceWindow] = []
    written: set[str] = set()
    for reference_series, index in WINDOWS:
        series_folder = reference_series.stem if reference_series else FLAT_FOLDER
        name = f"{series_folder}/w{index}"
        rows_path = work / f"{series_folder}.rows"
        if series_folder not in written:
            if reference_series:
                csv_path = data / DATASETS_FOLDER / reference_series
                try:
                    write_rows(csv_path, rows_path)
                except (OSError, ValueError, KeyError) as error:
                    emit(f"cannot read {csv_path}: {error}")
                    raise typer.Exit(1) from error
            else:
                window = flat()
                rows.write(rows_path, window.raw, window.mean, window.std)
            written.add(series_folder)
        result = _check_window(
            binary,
            name,
            rows_path,
            index,
            reference_folder / series_folder / f"w{index}.tdump",
            work,
            ratios,
            None,
        )
        emit(f"  {result.summary()}")
        if result.error:
            emit(f"  error: {result.error}")
        windows.append(
            ReferenceWindow(
                window=name,
                ok=result.ok,
                first_failure=result.failed[0] if result.failed else None,
                error=result.error,
            )
        )

    # Step 3: the outcome, with the hashes it was made with so a later call can tell whether it still holds
    write_json(
        folder / REFERENCE_FILE,
        asdict(ReferenceCheck(windows=windows, sha256=host.sha256)),
    )
    return sum(not window.ok for window in windows)
