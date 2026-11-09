"""The whole-series stage of `check-model`: does the C port detect anomalies as well as the Python baseline?

The C's per-window errors go through the pipeline's own post-processing into per-row scores, and those through the benchmark's metrics.
The series passes when every score's VUS-PR matches the baseline's within `VUS_TOLERANCE`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

# External
import numpy as np
import typer

# Internal
from score import rows
from score.capture import load
from score.layout import DATASETS_FOLDER, MODEL_FOLDER
from score.metrics import SeriesMetrics, score_series
from score.pipeline import HEADS, LABEL, Head, load_model, load_series
from score.rescore import scores_from_errors
from score.unit import SCORES


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# A VUS-PR difference below the third decimal, the precision VUS-PR results are reported at
VUS_TOLERANCE = 1e-3

# The archives' folders under the datasets folder, in the order a series is looked for
SERIES_FOLDERS = ("TSB-AD-U", "TSB-AD-M")
BASELINE_FOLDER = Path("results") / "python-baseline"

# The C program's CSV header, with the heads in `HEADS` order
ERRORS_HEADER = ",".join(("window", *HEADS))


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class PreparedSeries:
    """A series found with its baseline and the model, before any run starts.

    Attributes
    ----------
    csv_path : Path
        The series' CSV.
    baseline : dict[str, np.ndarray]
        The Python baseline's arrays: `errors_<head>` and `score_<score>`.
    model_path : Path
        The folder holding the checkpoint, read by the pipeline's post-processing for its context length.
    """

    csv_path: Path
    baseline: dict[str, np.ndarray]
    model_path: Path


@dataclass(frozen=True)
class ScoreComparison:
    """How one of the C port's scores detects anomalies, next to the baseline's same score.

    Attributes
    ----------
    name : str
        The score: a head or `ensemble`.
    baseline : SeriesMetrics
        The baseline's metrics.
    port : SeriesMetrics
        The C port's metrics.
    row_diff : float
        The largest per-row difference between the two scores, before the metrics rescale them.
    """

    name: str
    baseline: SeriesMetrics
    port: SeriesMetrics
    row_diff: float

    @property
    def vus_pr_diff(self) -> float:
        """The absolute VUS-PR difference, which decides the pass."""

        return abs(self.port.vus_pr - self.baseline.vus_pr)

    @property
    def ok(self) -> bool:
        """Whether the VUS-PR difference is within the tolerance. A NaN fails, because no comparison with it is true."""

        return self.vus_pr_diff <= VUS_TOLERANCE

    @staticmethod
    def header() -> str:
        """The printed table's header, with the column widths of `summary`."""

        return (
            f"  {'score':<9} {'baseline':>10} {'C':>10} {'|diff|':>9} {'result':<6} "
            f"{'VUS-ROC Δ':>9} {'AUC-PR Δ':>9} {'row Δ':>9}"
        )

    def summary(self) -> str:
        """The score's line of the printed table."""

        verdict = "pass" if self.ok else "FAIL"
        return (
            f"  {self.name:<9} {self.baseline.vus_pr:>10.6f} {self.port.vus_pr:>10.6f} "
            f"{self.vus_pr_diff:>9.2e} {verdict:<6} "
            f"{abs(self.port.vus_roc - self.baseline.vus_roc):>9.2e} "
            f"{abs(self.port.auc_pr - self.baseline.auc_pr):>9.2e} {self.row_diff:>9.2e}"
        )


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _find_series(data: Path, stem: str) -> Path | None:
    """Find a series' CSV in the dataset folders.

    Parameters
    ----------
    data : Path
        The folder with the data.
    stem : str
        The CSV's name without `.csv`.

    Returns
    -------
    Path or None
        The first `<data>/datasets/<folder>/<stem>.csv` that exists, or None.
    """

    for folder in SERIES_FOLDERS:
        path = data / DATASETS_FOLDER / folder / f"{stem}.csv"
        if path.is_file():
            return path
    return None


def write_rows(csv_path: Path, rows_path: Path) -> None:
    """Write a series' CSV as a `.rows` file, standardized over the whole series as the pipeline does.

    Parameters
    ----------
    csv_path : Path
        The series' CSV.
    rows_path : Path
        The `.rows` file to write.
    """

    series = load(csv_path)
    rows.write(rows_path, series.raw, series.mean, series.std)


def prepare(data: Path, stem: str) -> PreparedSeries:
    """Find a series, its baseline and the model, before any run starts.

    Parameters
    ----------
    data : Path
        The folder with the data.
    stem : str
        The series' CSV name without `.csv`.

    Returns
    -------
    PreparedSeries
        The CSV, the baseline's arrays and the model folder.

    Raises
    ------
    typer.Exit
        With code 1 when the series, its baseline or the model is missing, or the baseline lacks an array the check needs.
    """

    # Step 1: the files
    csv_path = _find_series(data, stem)
    baseline_path = data / BASELINE_FOLDER / f"{stem}.npz"
    model_path = data / MODEL_FOLDER
    if csv_path is None or not baseline_path.is_file():
        typer.echo(f"No series {stem} with a baseline under {data}.", err=True)
        raise typer.Exit(1)
    if not model_path.is_dir():
        typer.echo(f"Not found: {model_path}", err=True)
        raise typer.Exit(1)

    # Step 2: the baseline's errors and scores, every one of which the check needs
    keys = [f"errors_{head}" for head in HEADS] + [f"score_{name}" for name in SCORES]
    try:
        with np.load(baseline_path) as archive:
            baseline = {key: archive[key] for key in keys}
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        typer.echo(f"Cannot read {baseline_path}: {error}", err=True)
        raise typer.Exit(1) from error
    return PreparedSeries(csv_path, baseline, model_path)


def _run_clean(
    binary: Path, rows_path: Path, windows: int, work: Path
) -> dict[Head, np.ndarray]:
    """Run the clean build over every window of a series and read its errors.

    Parameters
    ----------
    binary : Path
        The clean build of the C program.
    rows_path : Path
        The `.rows` file holding the series.
    windows : int
        How many windows the series has, which the CSV must hold in order.
    work : Path
        The folder for the errors CSV.

    Returns
    -------
    dict[Head, np.ndarray]
        Per head, one float32 error per window.

    Raises
    ------
    RuntimeError
        When the program cannot run, fails, or writes a CSV that does not hold every window in order.
    """

    # Step 1: an empty output file first, because the same command must work under Spike, whose proxy kernel can't create files
    errors_path = work / f"{rows_path.stem}.errors.csv"
    errors_path.touch()

    # Step 2: the program on every window, its default
    try:
        run = subprocess.run(
            [str(binary), str(rows_path), str(errors_path)],
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"cannot run {binary}: {error}") from error
    if run.returncode:
        raise RuntimeError(f"exit {run.returncode}: {run.stderr.strip()}")

    # Step 3: its CSV, which must line up window by window with the baseline
    try:
        with errors_path.open() as handle:
            header = handle.readline().strip()
        got = np.loadtxt(errors_path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error
    if (
        header != ERRORS_HEADER
        or got.shape != (windows, len(HEADS) + 1)
        or not np.array_equal(got[:, 0], np.arange(windows))
    ):
        raise RuntimeError(f"{errors_path} does not hold windows 0 to {windows - 1}")

    # The program prints each float32 error with 9 significant digits, which read back as float32 exactly
    return {
        head: got[:, column + 1].astype(np.float32) for column, head in enumerate(HEADS)
    }


def _compare_scores(
    prepared: PreparedSeries, errors: dict[Head, np.ndarray]
) -> list[ScoreComparison]:
    """Score the C port's errors as the baseline was scored, and compare every score's metrics.

    Parameters
    ----------
    prepared : PreparedSeries
        The series, its baseline and the model folder.
    errors : dict[Head, np.ndarray]
        The C port's errors per head.

    Returns
    -------
    list[ScoreComparison]
        One comparison per score in `SCORES` order.

    Raises
    ------
    OSError
        When the model cannot be loaded.
    ValueError
        When the baseline's scores do not have one value per row of the series.
    RuntimeError
        When the pipeline does not merge exactly the three heads.
    """

    # Step 1: labels and values as the baseline run loaded them
    frame = load_series(prepared.csv_path)
    labels = frame[LABEL].astype(int).to_numpy()
    values = frame.drop(columns=LABEL).values.astype(float)

    # Step 2: the C's errors through the pipeline's post-processing
    model = load_model(prepared.model_path, values.shape[1])
    heads = scores_from_errors(model, frame, errors)
    port = heads.scores | {"ensemble": heads.ensemble}

    # Step 3: both sides' metrics through the same function, so only the scores differ. A baseline from another version of the series would not line up row by row
    for name in SCORES:
        stored = prepared.baseline[f"score_{name}"]
        if stored.shape != port[name].shape:
            raise ValueError(
                f"the baseline's score_{name} has {stored.shape[0]} rows, the series {port[name].shape[0]}"
            )
    comparisons = []
    for name in SCORES:
        stored = prepared.baseline[f"score_{name}"]
        comparisons.append(
            ScoreComparison(
                name=name,
                baseline=score_series(stored, labels, values),
                port=score_series(port[name], labels, values),
                row_diff=float(np.max(np.abs(port[name] - stored))),
            )
        )
    return comparisons


def stage(prepared: PreparedSeries, stem: str, work: Path, clean: Path) -> bool:
    """Check a whole series' detection and print the outcome.

    Parameters
    ----------
    prepared : PreparedSeries
        What `prepare` returned.
    stem : str
        The series' CSV name without `.csv`.
    work : Path
        The folder for the row file and errors.
    clean : Path
        The clean build of the C program.

    Returns
    -------
    bool
        Whether every score's VUS-PR is within the tolerance of the baseline's. False also when the C run fails, writes a NaN or Inf error, or the scoring cannot be done.

    Raises
    ------
    typer.Exit
        With code 1 when the CSV cannot be turned into a row file.
    """

    # Step 1: the row file
    rows_path = work / f"{stem}.rows"
    try:
        write_rows(prepared.csv_path, rows_path)
    except (OSError, ValueError, KeyError) as error:
        typer.echo(f"Cannot read {prepared.csv_path}: {error}", err=True)
        raise typer.Exit(1) from error

    # Step 2: the clean build over every window, timed alone
    windows = len(prepared.baseline[f"errors_{HEADS[0]}"])
    began = time.perf_counter()
    try:
        errors = _run_clean(clean.resolve(), rows_path, windows, work)
    except RuntimeError as error:
        typer.echo(f"{stem}: error: {error}")
        return False
    elapsed = time.perf_counter() - began

    # Step 3: a NaN or Inf error would be scored as some value by the post-processing, so it fails here
    bad = {
        head: int(np.count_nonzero(~np.isfinite(values)))
        for head, values in errors.items()
    }
    if any(bad.values()):
        counts = ", ".join(f"{head} {count}" for head, count in bad.items())
        typer.echo(
            f"{stem}: the C wrote NaN or Inf errors (windows per head: {counts}), {elapsed:.1f} s"
        )
        return False

    # Step 4: the scores and their metrics, timed alone. A model that won't load or a stale baseline is reported like a failed C run
    began = time.perf_counter()
    try:
        comparisons = _compare_scores(prepared, errors)
    except (OSError, ValueError, RuntimeError) as error:
        typer.echo(f"{stem}: error: {error}")
        return False
    scoring = time.perf_counter() - began

    # Step 5: the table. VUS-ROC, AUC-PR and the per-row difference are information only
    typer.echo(
        f"{stem}: {windows} windows, C {elapsed:.1f} s, scoring {scoring:.1f} s, VUS-PR tolerance {VUS_TOLERANCE:g}"
    )
    typer.echo(ScoreComparison.header())
    for comparison in comparisons:
        typer.echo(comparison.summary())
    return all(comparison.ok for comparison in comparisons)
