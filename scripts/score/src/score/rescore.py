"""Turn per-window errors into per-row scores with the anomaly pipeline's own post-processing, without running the model."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from typing import Any

# External
import numpy as np
import pandas as pd
import torch
from tsfm_public.models.tspulse import TSPulseForReconstruction

# Internal
from score.pipeline import (
    BATCH_SIZE,
    HEADS,
    LABEL,
    CapturingPipeline,
    Head,
    Heads,
    build_pipeline,
    split_heads,
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class _ReplayPipeline(CapturingPipeline):
    """The capturing pipeline, post-processing given errors where it would run the model.

    It takes the same arguments as `CapturingPipeline`, plus the errors to replay.

    Attributes
    ----------
    replayed : dict[Head, np.ndarray]
        Per head, one error per window, in place of the model's.
    """

    def __init__(
        self, *args: Any, replayed: dict[Head, np.ndarray], **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.replayed = replayed

    def run_single(
        self,
        inputs: pd.DataFrame,
        preprocess_params: dict[str, Any],
        forward_params: dict[str, Any],
        postprocess_params: dict[str, Any],
    ) -> pd.DataFrame:
        """Preprocess the series, then post-process the replayed errors as if the model had returned them.

        Parameters
        ----------
        inputs : pd.DataFrame
            The series' channels.
        preprocess_params : dict[str, Any]
            What the pipeline's `_sanitize_parameters` gave for `preprocess`.
        forward_params : dict[str, Any]
            What it gave for the model call, unused because the model never runs.
        postprocess_params : dict[str, Any]
            What it gave for `postprocess`.

        Returns
        -------
        pd.DataFrame
            The parent's result, whose `anomaly_score` column is the ensemble.
        """

        # `preprocess` keeps the series and its standardized copy, which `postprocess` reads for the least-significant line. Without it `postprocess` fails on a missing key
        self.preprocess(inputs, **preprocess_params)

        # One batch per head, in the shape the model's batches are concatenated into
        batches = {head: [torch.from_numpy(self.replayed[head])] for head in HEADS}
        return self.postprocess(batches, **postprocess_params)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def scores_from_errors(
    model: TSPulseForReconstruction,
    series: pd.DataFrame,
    errors: dict[Head, np.ndarray],
) -> Heads:
    """Turn each head's per-window errors into per-row scores and their ensemble, as `score_heads` does after the model.

    Parameters
    ----------
    model : TSPulseForReconstruction
        The loaded model. Only its configuration is read, for the context length the boundary adjustment pads with.
    series : pd.DataFrame
        The series, with `Label` as its only non-channel column.
    errors : dict[Head, np.ndarray]
        Per head, one float32 error per window, as the model would return it.

    Returns
    -------
    Heads
        The given errors, each head's scores and the ensemble.

    Raises
    ------
    RuntimeError
        If the pipeline does not merge exactly the three heads.
    """

    # Replay the given errors in place of the model's
    channels = series.columns.drop(LABEL).tolist()
    pipeline = build_pipeline(
        _ReplayPipeline, model, channels, BATCH_SIZE, replayed=errors
    )
    ensemble = pipeline(series[channels])["anomaly_score"].to_numpy()
    return split_heads(pipeline, ensemble)
