"""Score a series' scopes from the accuracy units' errors.

A scope's windows come from every unit overlapping its range, stitched in order and checked to be contiguous.
The span is scored as a series of its own, as the Python baseline scores a whole series.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
from dataclasses import dataclass
from pathlib import Path

# External
import numpy as np
import pandas as pd
from tsfm_public.models.tspulse import TSPulseForReconstruction

# Internal
from score.benchmark.results import ERRORS_FILE, read_errors
from score.benchmark.scope import Scope, Series
from score.benchmark.units import AccuracyUnit
from score.capture import CONTEXT
from score.metrics import SeriesMetrics, score_series
from score.pipeline import HEADS, LABEL, Head
from score.rescore import scores_from_errors
from score.unit import SCORES


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class ScopeScore:
    """One series' scope, scored.

    Attributes
    ----------
    series : Series
        The series.
    scope : Scope
        The scope.
    metrics : dict[str, SeriesMetrics]
        The metrics of each score in `SCORES` order: the three heads and the ensemble.
    """

    series: Series
    scope: Scope
    metrics: dict[str, SeriesMetrics]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def overlapping_units(
    units: list[AccuracyUnit], start: int, stop: int
) -> list[AccuracyUnit]:
    """Pick the units holding windows of a range.

    Parameters
    ----------
    units : list[AccuracyUnit]
        One series' units.
    start : int
        The range's first window.
    stop : int
        One past the range's last window.

    Returns
    -------
    list[AccuracyUnit]
        The units sharing a window with `[start, stop)`, in window order.
    """

    return sorted(
        (unit for unit in units if unit.start < stop and unit.stop > start),
        key=lambda unit: unit.start,
    )


def stitch(
    label_folder: Path, units: list[AccuracyUnit], start: int, stop: int
) -> dict[Head, np.ndarray]:
    """Join the units' errors into one array per head over `[start, stop)`.

    Parameters
    ----------
    label_folder : Path
        The label's folder.
    units : list[AccuracyUnit]
        The units overlapping the range, in window order, all done.
    start : int
        The range's first window.
    stop : int
        One past the range's last window.

    Returns
    -------
    dict[Head, np.ndarray]
        Per head, one float32 error per window of the range.

    Raises
    ------
    ValueError
        When the units leave a gap, don't reach the range's ends, hold different heads, or lack one of the three heads the pipeline merges.
    OSError
        When an errors file can't be read.
    """

    # Step 1: the units' windows must run on without a gap and cover the range
    if not units or units[0].start > start or units[-1].stop < stop:
        raise ValueError(f"the units do not cover windows {start} to {stop - 1}")
    for before, after in zip(units, units[1:], strict=False):
        if before.stop != after.start:
            raise ValueError(f"windows {before.stop} to {after.start - 1} are missing")

    # Step 2: each unit's columns by header name, which must be the same heads in every unit and the three the pipeline needs
    parts = [
        read_errors(label_folder / unit.folder / ERRORS_FILE, unit.start, unit.stop)
        for unit in units
    ]
    if any(part.keys() != parts[0].keys() for part in parts):
        raise ValueError("the units' errors hold different heads")
    if set(parts[0]) != set(HEADS):
        raise ValueError(
            f"the errors hold heads {sorted(parts[0])}, the pipeline needs {sorted(HEADS)}"
        )

    # Step 3: join and cut to the range
    offset = units[0].start
    return {
        head: np.concatenate([part[head] for part in parts])[
            start - offset : stop - offset
        ]
        for head in HEADS
    }


def score_scope(
    series: Series,
    scope: Scope,
    frame: pd.DataFrame,
    model: TSPulseForReconstruction,
    errors: dict[Head, np.ndarray],
    start: int,
    stop: int,
    destination: Path,
) -> ScopeScore:
    """Score the span of windows `[start, stop)` as a series of its own and save its arrays.

    Parameters
    ----------
    series : Series
        The series.
    scope : Scope
        The scope.
    frame : pd.DataFrame
        The whole series as `load_series` returns it.
    model : TSPulseForReconstruction
        The loaded model, read only for its context length.
    errors : dict[Head, np.ndarray]
        The errors of windows `[start, stop)`.
    start : int
        The span's first window.
    stop : int
        One past the span's last window.
    destination : Path
        The `.npz` to write.

    Returns
    -------
    ScopeScore
        The metrics of the three heads and the ensemble.

    Raises
    ------
    ValueError
        When the span's rows hold no anomaly label, as in a window-capped span that stops before the first anomaly.
    OSError
        When the arrays can't be written.
    """

    # Step 1: the span's rows. Window `w` covers rows `w..w+512`, so windows `[start, stop)` use rows `[start, stop + 512)`
    span = frame.iloc[start : stop + CONTEXT].reset_index(drop=True)
    labels = span[LABEL].astype(int).to_numpy()
    values = span.drop(columns=LABEL).values.astype(float)
    if not labels.any():
        raise ValueError(
            f"rows {start} to {stop + CONTEXT - 1} hold no anomaly label, so VUS is undefined"
        )

    # Step 2: the errors through the pipeline's post-processing, then each score's metrics from the span's own values, so its VUS window comes from the span
    heads = scores_from_errors(model, span, errors)
    scores = heads.scores | {"ensemble": heads.ensemble}
    metrics = {name: score_series(scores[name], labels, values) for name in SCORES}

    # Step 3: the arrays, so metrics can be recomputed without the model. Replaced whole, so a reader never sees half a file
    arrays = {f"errors_{head}": errors[head] for head in HEADS}
    arrays |= {f"score_{name}": scores[name] for name in SCORES}
    arrays |= {"labels": labels, "windows": np.array([start, stop])}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, destination)
    return ScopeScore(series, scope, metrics)
