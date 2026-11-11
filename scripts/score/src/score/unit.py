"""Score one series with every head, and save its errors, scores and metrics."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

# External
import numpy as np
import torch

# Internal
from score.kind import Kind
from score.layout import DATASETS_FOLDER, MODEL_FOLDER
from score.metrics import score_series
from score.pipeline import HEADS, LABEL, load_model, load_series, score_heads


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The tspulse-r1 commit that `download-weights` fetches
WEIGHTS_REVISION = "2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2"

SCORES = (*HEADS, "ensemble")


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Unit:
    """One series to score, with the details `benchmark/series.csv` gives it.

    Attributes
    ----------
    file_name : str
        The series' CSV name.
    kind : Kind
        The archive it comes from.
    source : str
        The original dataset, from the name.
    rows : int
        How many rows it has.
    channels : int
        How many channels it has.
    windows : int
        How many 512-row windows the model scores.
    passes : int
        How many single-channel model passes the ensemble takes over the whole series, which orders the units by cost.
    ibm_ensemble : float
        IBM's published ensemble VUS-PR, printed next to ours when the unit is done.
    """

    file_name: str
    kind: Kind
    source: str
    rows: int
    channels: int
    windows: int
    passes: int
    ibm_ensemble: float


@dataclass(frozen=True)
class Settings:
    """How every unit of one run is scored and where it goes.

    Attributes
    ----------
    data : Path
        The data folder, `$TSPULSE_DATA`.
    out : Path
        The results folder.
    batch_size : int
        How many windows the model handles at once.
    threads : int
        How many threads PyTorch uses in each worker.
    git_sha : str
        The project commit the run is made from, recorded with each result.
    """

    data: Path
    out: Path
    batch_size: int
    threads: int
    git_sha: str


@dataclass(frozen=True)
class Finished:
    """What the command prints when a unit is done.

    Attributes
    ----------
    file_name : str
        The series' CSV name.
    seconds : float
        How long the whole unit took, loading and metrics included.
    vus_pr : float
        Our ensemble VUS-PR.
    ibm_vus_pr : float
        IBM's ensemble VUS-PR for the same series.
    """

    file_name: str
    seconds: float
    vus_pr: float
    ibm_vus_pr: float


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def result_path(out: Path, file_name: str) -> Path:
    """Name the JSON whose presence marks a unit as done.

    Parameters
    ----------
    out : Path
        The results folder.
    file_name : str
        The series' CSV name.

    Returns
    -------
    Path
        `<out>/<stem>.json`.
    """

    return out / f"{Path(file_name).stem}.json"


def run_unit(unit: Unit, settings: Settings) -> Finished:
    """Score one series and write its `.npz` and `.json`.

    Runs in its own process, so the peak memory it records belongs to this series alone.

    Parameters
    ----------
    unit : Unit
        The series.
    settings : Settings
        The run's settings.

    Returns
    -------
    Finished
        The line to print.
    """

    start = time.perf_counter()
    torch.set_num_threads(settings.threads)

    frame = load_series(
        settings.data / DATASETS_FOLDER / f"TSB-AD-{unit.kind}" / unit.file_name
    )
    labels = frame[LABEL].astype(int).to_numpy()
    values = frame.drop(columns=LABEL).values.astype(float)
    model = load_model(settings.data / MODEL_FOLDER, values.shape[1])

    model_start = time.perf_counter()
    heads = score_heads(model, frame, settings.batch_size)
    model_seconds = time.perf_counter() - model_start

    scores = heads.scores | {"ensemble": heads.ensemble}
    metrics = {name: score_series(scores[name], labels, values) for name in SCORES}

    stem = Path(unit.file_name).stem
    settings.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        settings.out / f"{stem}.npz",
        labels=labels,
        **{f"errors_{head}": heads.errors[head] for head in HEADS},
        **{f"score_{name}": scores[name] for name in SCORES},
    )

    seconds = time.perf_counter() - start
    record = asdict(unit) | {
        "sliding_window": metrics["ensemble"].sliding_window,
        "metrics": {
            name: {"VUS-PR": m.vus_pr, "VUS-ROC": m.vus_roc, "AUC-PR": m.auc_pr}
            for name, m in metrics.items()
        },
        "seconds": seconds,
        "model_seconds": model_seconds,
        # Linux reports kibibytes
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "batch_size": settings.batch_size,
        "threads": settings.threads,
        "torch": torch.__version__,
        "granite_tsfm": version("granite-tsfm"),
        "weights_revision": WEIGHTS_REVISION,
        "git_sha": settings.git_sha,
    }

    # Written last and renamed into place, so a unit stopped halfway is never taken as done
    path = result_path(settings.out, unit.file_name)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(record, indent=2))
    os.replace(partial, path)

    return Finished(
        unit.file_name, seconds, metrics["ensemble"].vus_pr, unit.ibm_ensemble
    )
