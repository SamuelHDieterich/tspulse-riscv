"""Load a series and the model, and run IBM's TSPulse anomaly pipeline."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# External
import numpy as np
import pandas as pd
import torch
from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

LABEL = "Label"
HEADS = ("time", "fft", "forecast")

# These are the values IBM's TSB-AD benchmark uses with the pipeline.
# They change how the model is run and how errors become scores, never its weights.

# How many rows at the end of each 512-row window get graded. They are split into 8-row pieces, each hidden once, and the errors are averaged into one grade.
# The default of 32 hides 4 pieces. IBM's 96 hides 12, which triples the work for `time` and `fft`.
AGGREGATION_LENGTH = 96

# Averages each score with its neighbours, so the line is less jittery. `forecast` skips this. Same as the default.
SMOOTHING_LENGTH = 8

# Builds the ensemble by keeping the highest score on each row, so one alarmed head is enough. Same as the default.
AGGREGATION_FUNCTION = "max"

# The "too small to care about" line for errors, which scales with how much the series moves from row to row.
# The default is 0.01. IBM's 0.0 turns the line off.
LEAST_SIGNIFICANT_SCALE = 0.0

# Extra weight for errors above that line. With the default of 0.2, a head with no such errors can't score above 0.2.
# IBM's 1.0 adds no weight, so the worst row of each head simply scales to 1 before smoothing.
LEAST_SIGNIFICANT_SCORE = 1.0

# How many windows the model handles at once, and `time` and `fft` make 12 copies of each. Bigger uses more memory, and rounding moves the scores slightly between sizes. Same as the default.
BATCH_SIZE = 128


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Heads:
    """Everything one pipeline call with all heads produces, before and after post-processing.

    Attributes
    ----------
    errors : dict[str, np.ndarray]
        Per head, one model error per window, as the model returns it.
    scores : dict[str, np.ndarray]
        Per head, one score per row, after boundary adjustment and smoothing.
    ensemble : np.ndarray
        The row-wise maximum of `scores`, which is what the pipeline returns.
    """

    errors: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    ensemble: np.ndarray


class _CapturingPipeline(TimeSeriesAnomalyDetectionPipeline):
    """The anomaly pipeline, recording each head's errors and scores before they are merged.

    It takes the same arguments as `TimeSeriesAnomalyDetectionPipeline`, and both attributes are filled by the call.

    Attributes
    ----------
    errors : dict[str, np.ndarray]
        Per head, one model error per window, keyed in the order `postprocess` stacks the heads.
    stacked : np.ndarray | None
        The heads' scores just before they are merged, with shape (heads, rows, 1), or None before the first call.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.errors: dict[str, np.ndarray] = {}
        self.stacked: np.ndarray | None = None
        merge = self.aggr_function

        # `postprocess` stacks the heads' scores in the order of the error keys, then calls this once
        def record(score: np.ndarray, axis: int) -> np.ndarray:
            self.stacked = score
            return merge(score, axis=axis)

        self.aggr_function = record

    def postprocess(
        self, model_outputs: Any, **postprocess_parameters: Any
    ) -> pd.DataFrame:
        """Record each head's errors, then post-process as the parent pipeline does.

        Parameters
        ----------
        model_outputs : Any
            Per head, a list with the model errors of every batch, as the parent's `run_single` collects them.
        **postprocess_parameters : Any
            Passed to the parent unchanged.

        Returns
        -------
        pd.DataFrame
            The parent's result, whose `anomaly_score` column is the ensemble.
        """

        self.errors = {
            key: torch.cat(list(batches)).numpy()
            for key, batches in model_outputs.items()
        }
        return super().postprocess(model_outputs, **postprocess_parameters)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def load_series(path: Path) -> pd.DataFrame:
    """Read a series, dropping incomplete rows.

    Parameters
    ----------
    path : Path
        A numeric CSV with one column per channel and a `Label` column.

    Returns
    -------
    pd.DataFrame
        The series, indexed by row position.
    """

    return pd.read_csv(path).dropna().reset_index(drop=True)


def load_model(path: Path, channels: int) -> TSPulseForReconstruction:
    """Load the TSPulse checkpoint for inference.

    Parameters
    ----------
    path : Path
        The folder holding the checkpoint.
    channels : int
        How many channels the series has.

    Returns
    -------
    TSPulseForReconstruction
        The model in evaluation mode.
    """

    # A mask hides part of the input so the model has to rebuild it, and a big miss marks an anomaly.
    # Training used random masks (var_hybrid). The pipeline has to choose what to hide, so it needs "user", or its masks are silently replaced with random ones.

    # The checkpoint says 1 channel and fails on more. The same weights are reused for every channel, so the real count changes nothing the model learned.
    return TSPulseForReconstruction.from_pretrained(
        path, mask_type="user", num_input_channels=channels
    ).eval()


def score(
    model: TSPulseForReconstruction, series: pd.DataFrame, modes: Sequence[str]
) -> np.ndarray:
    """Run the anomaly pipeline over a series.

    Parameters
    ----------
    model : TSPulseForReconstruction
        The loaded model.
    series : pd.DataFrame
        The series, with `Label` as its only non-channel column.
    modes : Sequence[str]
        The heads to combine into one score.

    Returns
    -------
    np.ndarray
        One anomaly score per row. Rows outside the first and last scored windows repeat those windows' scores.
    """

    channels = series.columns.drop(LABEL).tolist()

    # batch_size only takes effect here. Passed to the pipeline call instead, it is silently dropped.
    pipeline = TimeSeriesAnomalyDetectionPipeline(
        model,
        target_columns=channels,
        prediction_mode=list(modes),
        aggregation_length=AGGREGATION_LENGTH,
        smoothing_length=SMOOTHING_LENGTH,
        aggr_function=AGGREGATION_FUNCTION,
        least_significant_scale=LEAST_SIGNIFICANT_SCALE,
        least_significant_score=LEAST_SIGNIFICANT_SCORE,
        batch_size=BATCH_SIZE,
    )
    return pipeline(series[channels])["anomaly_score"].to_numpy()


def score_heads(
    model: TSPulseForReconstruction, series: pd.DataFrame, batch_size: int = BATCH_SIZE
) -> Heads:
    """Run the anomaly pipeline once with every head, keeping each head's errors and scores.

    Parameters
    ----------
    model : TSPulseForReconstruction
        The loaded model.
    series : pd.DataFrame
        The series, with `Label` as its only non-channel column.
    batch_size : int
        How many windows the model handles at once.

    Returns
    -------
    Heads
        The raw errors and the scores of `time`, `fft` and `forecast`, and their ensemble.
    """

    channels = series.columns.drop(LABEL).tolist()
    pipeline = _CapturingPipeline(
        model,
        target_columns=channels,
        prediction_mode=list(HEADS),
        aggregation_length=AGGREGATION_LENGTH,
        smoothing_length=SMOOTHING_LENGTH,
        aggr_function=AGGREGATION_FUNCTION,
        least_significant_scale=LEAST_SIGNIFICANT_SCALE,
        least_significant_score=LEAST_SIGNIFICANT_SCORE,
        batch_size=batch_size,
    )
    ensemble = pipeline(series[channels])["anomaly_score"].to_numpy()

    keys = list(pipeline.errors)
    if pipeline.stacked is None or sorted(keys) != sorted(HEADS):
        raise RuntimeError(
            f"Expected the pipeline to merge the heads {HEADS}, got {keys}"
        )

    # The stack has shape (heads, rows, 1): the channels are already averaged
    scores = {key: pipeline.stacked[i].ravel() for i, key in enumerate(keys)}
    return Heads(
        errors={head: pipeline.errors[head] for head in HEADS},
        scores={head: scores[head] for head in HEADS},
        ensemble=ensemble,
    )
