"""The `check-model` command."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

# External
import numpy as np
import typer

# Internal
from score import noise, rows, series_check, tdump
from score.capture import flat
from score.layout import DATASETS_FOLDER, REFERENCE_FOLDER
from score.reference import WINDOWS
from score.series_check import VUS_TOLERANCE, write_rows


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Records the C port never writes, matched by prefix after the copy: the second `get_fft` on the unmasked input and the softmax mapping the baseline skips
EXPECTED_MISSING_EVERY_COPY = (
    "backbone.groundtruth_scaler_for_fft",
    "backbone.original_past_values_fft",
    "output.original_past_values_fft",
    "backbone.original_fft_softmax",
    "output.original_fft_softmax",
    "decoder_with_head.fft_softmax_mapping",
    "decoder_with_head.fft_softmax_preds",
    "output.fft_softmax_preds",
)

# The pass with nothing hidden only feeds the forecast head, so the time and FFT heads are skipped
EXPECTED_MISSING_PLAIN = (
    "decoder_with_head.time_head",
    "decoder_with_head.fft_head",
    "decoder_with_head.reconstruction_outputs",
    "decoder_with_head.reconstruction_fft_outputs",
    "decoder_with_head.reconstructed_ts_from_fft",
    "output.reconstruction_outputs",
    "output.fft_reconstruction_outputs",
    "output.reconstructed_ts_from_fft",
)

# The hidden passes only feed the stitched time and FFT errors, so the forecast head is skipped
EXPECTED_MISSING_HIDDEN = (
    "decoder_with_head.forecast_mapping",
    "decoder_with_head.forecast_output",
    "output.forecast_output",
)

# float64 records, which the C writer can't hold
EXPECTED_MISSING_NO_COPY = ("input.raw", "input.mean", "input.std")

FLAT_FOLDER = "flat"
PLAIN_COPY = "plain"
CHECK_FOLDER = Path("tspulse/build/check-model")


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass
class WindowResult:
    """How one reference window compared with the C port's dump.

    Attributes
    ----------
    label : str
        The window's name, such as `flat/w0`.
    passed : int
        Records within the allowed difference.
    failed : list[str]
        Names of records that differ too much, are not finite or have another shape.
    missing : list[str]
        Names of reference records the dump lacks.
    unexpected_missing : list[str]
        Names in `missing` that are not on the expected list.
    unexpected_present : list[str]
        Names on the expected list that the dump has anyway.
    extra : list[str]
        Names in the dump that the reference lacks.
    worst : float
        The largest `diff / allowed` over the compared records.
    worst_name : str
        The record where `worst` was found, empty when nothing was compared.
    error : str
        Why the program or the dump could not be used, empty when it could.
    """

    label: str
    passed: int = 0
    failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected_missing: list[str] = field(default_factory=list)
    unexpected_present: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    worst: float = 0.0
    worst_name: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether the window passes: nothing failed, extra, unexpectedly missing or unexpectedly present, something compared, and no error."""

        return not (
            self.error
            or self.failed
            or self.extra
            or self.unexpected_missing
            or self.unexpected_present
            or self.passed == 0
        )

    def summary(self) -> str:
        """The one-line summary printed for the window."""

        worst = f"worst {self.worst:.3g} of allowed at {self.worst_name}"
        return (
            f"{self.label}: {self.passed} pass, {len(self.failed)} fail, "
            f"{len(self.missing)} missing ({len(self.unexpected_missing)} unexpected), "
            f"{len(self.extra)} extra, {len(self.unexpected_present)} present but expected missing, "
            f"{worst if self.worst_name else 'nothing compared'}"
        )


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _expected_missing(name: str) -> bool:
    """Check whether the C port is expected to leave a reference record out.

    Parameters
    ----------
    name : str
        A reference record name, with its copy prefix.

    Returns
    -------
    bool
        True when the name, after its copy, starts with a prefix on the expected-missing list for that copy.
    """

    if name in EXPECTED_MISSING_NO_COPY:
        return True
    match = noise.COPY_PREFIX.match(name)
    if match is None:
        return False
    skipped = EXPECTED_MISSING_EVERY_COPY + (
        EXPECTED_MISSING_PLAIN
        if match.group(1) == PLAIN_COPY
        else EXPECTED_MISSING_HIDDEN
    )
    return name[match.end() :].startswith(skipped)


def _max_diff(ref: np.ndarray, dump: np.ndarray) -> float:
    """Compute the largest absolute difference between two arrays of the same shape.

    Parameters
    ----------
    ref : np.ndarray
        The reference values.
    dump : np.ndarray
        The values to check.

    Returns
    -------
    float
        The largest `|dump − ref|`, computed in float64 and ignoring positions where either side is NaN. 0 when nothing is left.
    """

    diff = np.abs(dump.astype(np.float64) - ref.astype(np.float64))
    finite = diff[~np.isnan(diff)]
    return float(finite.max()) if finite.size else 0.0


def _compare(
    result: WindowResult,
    reference: dict[str, np.ndarray],
    dump: dict[str, np.ndarray],
    ratios: dict[str, float],
    only: str | None,
) -> None:
    """Fill a result by comparing a dump against its reference with the noise rule.

    Parameters
    ----------
    result : WindowResult
        The result to fill in.
    reference : dict[str, np.ndarray]
        The reference records.
    dump : dict[str, np.ndarray]
        The C port's records.
    ratios : dict[str, float]
        The pooled noise ratio per step.
    only : str or None
        Only compare records whose name starts with this.
    """

    names = [name for name in reference if only is None or name.startswith(only)]
    for name in names:
        # Step 1: a record the dump lacks is missing, and is expected to be or not
        expected_missing = _expected_missing(name)
        if name not in dump:
            result.missing.append(name)
            if not expected_missing:
                result.unexpected_missing.append(name)
            continue
        if expected_missing:
            result.unexpected_present.append(name)

        # Step 2: a shape mismatch fails without looking at the values
        expected, got = reference[name], dump[name]
        if got.shape != expected.shape:
            result.failed.append(name)
            continue

        # Step 3: the rule `compare-reference` applies. A NaN or Inf in the dump fails even when `_max_diff`, which skips NaNs, is within the allowed difference
        diff = _max_diff(expected, got)
        ref_max = float(np.max(np.abs(expected))) if expected.size else 0.0
        allowed = noise.allowed(ratios[noise.step(name)], ref_max)
        ratio = diff / allowed if allowed else (0.0 if diff == 0.0 else float("inf"))
        if ratio > result.worst or not result.worst_name:
            result.worst, result.worst_name = ratio, name
        if bool(np.isfinite(got).all()) and diff <= allowed:
            result.passed += 1
        else:
            result.failed.append(name)

    # Step 4: the dump's records that the reference doesn't have
    result.extra = [
        name
        for name in dump
        if name not in reference and (only is None or name.startswith(only))
    ]


def _check_window(
    binary: Path,
    label: str,
    rows_path: Path,
    index: int,
    reference_path: Path,
    work: Path,
    ratios: dict[str, float],
    only: str | None,
) -> WindowResult:
    """Run the dump build on one window and compare what it wrote.

    Parameters
    ----------
    binary : Path
        The dump build of the C program.
    label : str
        The window's name, such as `flat/w0`.
    rows_path : Path
        The `.rows` file holding the window's series.
    index : int
        The window's index in that series.
    reference_path : Path
        The reference `.tdump`.
    work : Path
        The folder for the errors CSV and the dump.
    ratios : dict[str, float]
        The pooled noise ratio per step.
    only : str or None
        Only compare records whose name starts with this.

    Returns
    -------
    WindowResult
        How the dump compares with the reference.
    """

    result = WindowResult(label)
    stem = label.replace("/", "_")
    errors_path = work / f"{stem}.errors.csv"
    dump_path = work / f"{stem}.tdump"

    # Step 1: the program on just this window, dumping it
    command = [
        str(binary),
        str(rows_path),
        str(errors_path),
        "--windows",
        f"{index}:{index + 1}",
        "--dump-window",
        str(index),
        "--dump",
        str(dump_path),
    ]
    try:
        run = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        result.error = f"cannot run {binary}: {error}"
        return result
    if run.returncode:
        result.error = f"exit {run.returncode}: {run.stderr.strip()}"
        return result

    # Step 2: read both dumps and compare
    try:
        _compare(
            result, tdump.read(reference_path), tdump.read(dump_path), ratios, only
        )
    except (OSError, ValueError) as error:
        result.error = str(error)
    return result


def _work_folder() -> Path:
    """Find the default work folder inside the current worktree.

    Returns
    -------
    Path
        `tspulse/build/check-model` under the git top level, so it doesn't depend on the current folder.

    Raises
    ------
    typer.Exit
        With code 1 when the current folder is not inside a git worktree.
    """

    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if top.returncode:
        typer.echo(f"Not inside a git worktree: {top.stderr.strip()}", err=True)
        raise typer.Exit(1)
    return Path(top.stdout.strip()) / CHECK_FOLDER


def _check_model(
    binary: Annotated[
        Path, typer.Argument(help="The dump build of the C program, `tspulse-dump`.")
    ],
    data: Annotated[
        Path | None,
        typer.Option(envvar="TSPULSE_DATA", help="The folder with the data."),
    ] = None,
    work: Annotated[
        Path | None,
        typer.Option(
            help="The folder for the row files, errors and dumps. Defaults to tspulse/build/check-model in this worktree."
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(help="Only compare records whose name starts with this."),
    ] = None,
    series: Annotated[
        str | None,
        typer.Option(
            help=f"After the 7 windows, also check that the C detects anomalies on this whole series, by CSV name without `.csv`, as well as the Python baseline: VUS-PR within {VUS_TOLERANCE:g} per score. The 7 windows stay the precise check."
        ),
    ] = None,
    clean: Annotated[
        Path | None,
        typer.Option(
            help="With --series, the clean build of the C program, `tspulse`."
        ),
    ] = None,
) -> None:
    """Run the C program on the 7 reference windows and compare what it dumps with the reference, then optionally a whole series.

    Parameters
    ----------
    binary : Path
        The dump build of the C program.
    data : Path or None
        The folder with the data.
    work : Path or None
        The folder for the row files, errors and dumps.
    only : str or None
        Only compare records whose name starts with this.
    series : str or None
        The name of a series to check as a whole.
    clean : Path or None
        The clean build of the C program.

    Raises
    ------
    typer.Exit
        With code 1 when the data folder or a file is missing, an option is wrong, any window fails, or the series' detection differs.
    """

    # Step 1: the data folder, the programs and the pooled noise
    if data is None:
        typer.echo("No data folder: set $TSPULSE_DATA or pass --data.", err=True)
        raise typer.Exit(1)
    if series is not None and clean is None:
        typer.echo("--series needs --clean.", err=True)
        raise typer.Exit(1)
    if series is None and clean is not None:
        typer.echo("--clean needs --series.", err=True)
        raise typer.Exit(1)
    for program in (binary, clean):
        if program is not None and not program.is_file():
            typer.echo(f"Not found: {program}", err=True)
            raise typer.Exit(1)
    reference_folder = data / REFERENCE_FOLDER
    try:
        ratios = noise.read_pooled(reference_folder / "noise.csv")
    except (OSError, ValueError) as error:
        typer.echo(error, err=True)
        raise typer.Exit(1) from error
    work = work or _work_folder()
    work.mkdir(parents=True, exist_ok=True)

    # The series' files are checked first, so a typo does not cost the 7-window stage
    prepared = series_check.prepare(data, series) if series is not None else None

    # Step 2: one run per reference window. Each series' row file holds the whole series and is written the first time one of its windows comes up, so the program's skip of earlier windows is exercised too
    results = []
    written: set[str] = set()
    for reference_series, index in WINDOWS:
        folder = reference_series.stem if reference_series else FLAT_FOLDER
        label = f"{folder}/w{index}"
        rows_path = work / f"{folder}.rows"
        if folder not in written:
            if reference_series:
                csv_path = data / DATASETS_FOLDER / reference_series
                try:
                    write_rows(csv_path, rows_path)
                except (OSError, ValueError, KeyError) as error:
                    typer.echo(f"Cannot read {csv_path}: {error}", err=True)
                    raise typer.Exit(1) from error
            else:
                window = flat()
                rows.write(rows_path, window.raw, window.mean, window.std)
            written.add(folder)
        result = _check_window(
            binary.resolve(),
            label,
            rows_path,
            index,
            reference_folder / folder / f"w{index}.tdump",
            work,
            ratios,
            only,
        )
        results.append(result)
        typer.echo(result.summary())
        if result.error:
            typer.echo(f"  error: {result.error}")
        for name in result.failed[:1]:
            typer.echo(f"  first failure: {name}")
        for name in result.extra[:5]:
            typer.echo(f"  extra: {name}")

    # Step 3: the series, when asked for
    ok = all(result.ok for result in results)
    if series is not None and clean is not None and prepared is not None:
        ok = series_check.stage(prepared, series, work, clean) and ok

    # Step 4: the verdict
    if not ok:
        raise typer.Exit(1)


# ░█▄█░█▀█░▀█▀░█▀█
# ░█░█░█▀█░░█░░█░█
# ░▀░▀░▀░▀░▀▀▀░▀░▀


def main() -> None:
    """Run `check-model`."""

    typer.run(_check_model)
