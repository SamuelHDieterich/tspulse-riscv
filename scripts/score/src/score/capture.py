"""Capture every inner output of TSPulse on one window, and how far float32 drifts from float64 there."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

# External
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tsfm_public.models.tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
    TimeSeriesAnomalyDetectionPipeline,
)

# Internal
from score.pipeline import (
    AGGREGATION_FUNCTION,
    AGGREGATION_LENGTH,
    BATCH_SIZE,
    HEADS,
    LABEL,
    LEAST_SIGNIFICANT_SCALE,
    LEAST_SIGNIFICANT_SCORE,
    SMOOTHING_LENGTH,
    load_series,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

CONTEXT = 512

# The stitched pass hides each of the last 12 patches once, so every window becomes 12 rows of its batch
HIDDEN_COPIES = 12

# Loss modules only score training targets, and their outputs carry no batch dimension
SKIPPED_MODULES = {"loss", "ces_loss"}

# A placeholder `torch.zeros(1)`, unused at inference, which the batch filter can't tell from a batch of 1
SKIPPED_NAME = "fft_mask"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Window:
    """One window as the pipeline feeds it, with the values it was made from.

    Attributes
    ----------
    raw : np.ndarray
        float64 (513, C), the series rows before standardizing.
    mean : np.ndarray
        float64 (C,), the per-channel average subtracted.
    std : np.ndarray
        float64 (C,), the per-channel spread divided by.
    values : np.ndarray
        float32 (513, C), the 512 past rows and the forecast target, standardized.
    """

    raw: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    values: np.ndarray

    @property
    def records(self) -> list[tuple[str, np.ndarray]]:
        """The `input.*` records, in dump order."""

        return [
            ("input.raw", self.raw),
            ("input.mean", self.mean),
            ("input.std", self.std),
            ("input.window", self.values),
        ]


@dataclass(frozen=True)
class Series:
    """A whole series, standardized the way the pipeline does it.

    Attributes
    ----------
    raw : np.ndarray
        float64 (rows, C), after dropping incomplete rows.
    scaled : np.ndarray
        float64 (rows, C), standardized over the whole series.
    mean : np.ndarray
        float64 (C,).
    std : np.ndarray
        float64 (C,), the population spread.
    """

    raw: np.ndarray
    scaled: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @property
    def windows(self) -> int:
        """How many windows the pipeline scores: each needs 512 past rows and one target."""

        return len(self.raw) - CONTEXT

    def window(self, index: int) -> Window:
        """Window `index` covers rows `index` to `index + 511`, and its forecast target is row `index + 512`."""

        if not 0 <= index < self.windows:
            raise IndexError(f"Window {index} is outside 0 … {self.windows - 1}")
        rows = slice(index, index + CONTEXT + 1)
        return Window(
            raw=self.raw[rows],
            mean=self.mean,
            std=self.std,
            values=self.scaled[rows].astype(np.float32),
        )


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def load(path: Path) -> Series:
    """Load a series and standardize every channel over the whole file, as the pipeline does.

    Parameters
    ----------
    path : Path
        A numeric CSV with one column per channel and a `Label` column.

    Returns
    -------
    Series
        The series and the values used to standardize it.
    """

    frame = load_series(path)
    raw = frame[frame.columns.drop(LABEL)].to_numpy(dtype=np.float64)

    # The pipeline's own scaler, so the average and spread match it to the last bit
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    return Series(raw=raw, scaled=scaled, mean=scaler.mean_, std=scaler.scale_)


def flat() -> Window:
    """A made-up window of zeros, given average 0 and spread 1 so it stays all zeros once standardized."""

    zeros = np.zeros((CONTEXT + 1, 1))
    return Window(
        raw=zeros,
        mean=np.zeros(1),
        std=np.ones(1),
        values=zeros.astype(np.float32),
    )


def _parts(name: str, value: object) -> Iterator[tuple[str, torch.Tensor]]:
    """Split a module output into named tensors: `.0`, `.1` … for tuples and `.<field>` for `ModelOutput`."""

    if isinstance(value, torch.Tensor):
        yield name, value
    # ModelOutput is both a mapping and a tuple, so the mapping test comes first to keep field names
    elif isinstance(value, Mapping):
        for key, part in value.items():
            yield from _parts(f"{name}.{key}", part)
    elif isinstance(value, tuple | list):
        for index, part in enumerate(value):
            yield from _parts(f"{name}.{index}", part)


def run(
    model: TSPulseForReconstruction,
    past_values: np.ndarray,
    future_values: np.ndarray,
    row: int,
    threads: int,
) -> dict[str, np.ndarray]:
    """Run the anomaly pipeline's scoring on a batch and keep one window's inner outputs and errors.

    The batch's dtype picks the arithmetic, so the caller converts the model to match.

    Parameters
    ----------
    model : TSPulseForReconstruction
        The loaded model, in the same dtype as the inputs.
    past_values : np.ndarray
        (B, 512, C), standardized.
    future_values : np.ndarray
        (B, 1, C), standardized.
    row : int
        The batch row of the window to keep.
    threads : int
        PyTorch's thread count, which changes how sums are split and so the last bits.

    Returns
    -------
    dict[str, np.ndarray]
        The window's records, in the order they were produced, named as `score.tdump` describes.

    Raises
    ------
    RuntimeError
        If the pipeline calls the model a third time, which no known path does.
    """

    # Step 1: state shared by the hooks below. `calls` counts the pipeline's model calls, `fired` how often each module has run within the current call
    batch = past_values.shape[0]
    records: dict[str, np.ndarray] = {}
    calls = 0
    fired: dict[str, int] = {}

    # Step 2: store one tensor as the window's records, one per copy
    def keep(name: str, tensor: torch.Tensor) -> None:
        # Phase 1 has the batch as given, phase 2 has 12 copies of each row
        size = batch if calls == 1 else batch * HIDDEN_COPIES
        if SKIPPED_NAME in name or tensor.ndim == 0 or tensor.shape[0] != size:
            return

        # The format holds only floats
        if tensor.is_complex():
            tensor, name = tensor.real, f"{name}.real"
        elif tensor.dtype == torch.bool:
            tensor = tensor.float()

        # Slicing first, then cloning, so a batch of 128 only ever stores one window
        if calls == 1:
            copies = {"plain": row}
        else:
            copies = {
                f"hidden{i:02d}": row * HIDDEN_COPIES + i for i in range(HIDDEN_COPIES)
            }
        for copy, index in copies.items():
            key = f"{copy}/{name}"
            if key in records:
                raise RuntimeError(f"{key} captured twice")
            records[key] = tensor[index].detach().cpu().clone().numpy()

    # Step 3: a hook before each model call. The pipeline calls the model once with the batch as given, for the forecast, then once with the 12 hidden copies
    def count_call(module: torch.nn.Module, args: tuple) -> None:
        nonlocal calls
        calls += 1
        fired.clear()
        if calls > 2:
            raise RuntimeError("The pipeline called the model a third time")

    # Step 4: a hook after each module, which numbers repeat runs within a call and stores every tensor in the output
    def module_hook(name: str):
        def hook(module: torch.nn.Module, args: tuple, output: object) -> None:
            fired[name] = fired.get(name, 0) + 1
            suffixed = name if fired[name] == 1 else f"{name}#{fired[name]}"
            for part, tensor in _parts(suffixed, output):
                keep(part, tensor)

        return hook

    # Step 5: a hook after the whole model, for its output fields. Losses are training-only
    def output_hook(module: torch.nn.Module, args: tuple, output: Mapping) -> None:
        for field, value in output.items():
            if "loss" not in field and isinstance(value, torch.Tensor):
                keep(f"output.{field}", value)

    # Step 6: attach the hooks. The handles let them be removed afterwards, so the model is left as it came
    handles = [
        model.register_forward_pre_hook(count_call),
        model.register_forward_hook(output_hook),
    ]
    handles += [
        module.register_forward_hook(module_hook(name))
        for name, module in model.named_modules()
        if name and name not in SKIPPED_MODULES
    ]

    # Step 7: the anomaly pipeline, set up as `score-series` sets it up
    channels = past_values.shape[2]
    pipeline = TimeSeriesAnomalyDetectionPipeline(
        model,
        target_columns=[f"channel{c}" for c in range(channels)],
        prediction_mode=list(HEADS),
        aggregation_length=AGGREGATION_LENGTH,
        smoothing_length=SMOOTHING_LENGTH,
        aggr_function=AGGREGATION_FUNCTION,
        least_significant_scale=LEAST_SIGNIFICANT_SCALE,
        least_significant_score=LEAST_SIGNIFICANT_SCORE,
        batch_size=BATCH_SIZE,
    )

    # Step 8: score the batch, which fires every hook. The thread count and the hooks are restored even if it fails
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            errors = pipeline._model_processor.compute_score(
                {
                    "past_values": torch.from_numpy(past_values),
                    "future_values": torch.from_numpy(future_values),
                },
                expand_score=False,
            )
    finally:
        torch.set_num_threads(previous_threads)
        for handle in handles:
            handle.remove()

    # Step 9: each head's error for the kept window
    for head in HEADS:
        records[f"error.{head}"] = errors[head][row : row + 1].numpy().copy()
    return records


def capture(
    model: TSPulseForReconstruction, window: Window
) -> tuple[dict[str, np.ndarray], list[tuple[str, float, float]]]:
    """Capture one window in float32 at batch 1 and one thread, and measure its float64 drift.

    Parameters
    ----------
    model : TSPulseForReconstruction
        The loaded model, in float32. It is float32 again on return.
    window : Window
        The window to capture.

    Returns
    -------
    dict[str, np.ndarray]
        Every record of the window's dump, in dump order.
    list[tuple[str, float, float]]
        Per record, its name, `max |float32 − float64|` and `max |float32|`.
    """

    # Step 1: the float32 capture, which is the reference. `None` adds the batch dimension of 1
    past_values = window.values[None, :CONTEXT]
    future_values = window.values[None, CONTEXT:]
    single = run(model, past_values, future_values, row=0, threads=1)

    # Step 2: the same window in float64, the nearer-exact answer the float32 drift is measured against.
    # The same float32 inputs, upcast, so only the model's arithmetic differs.
    # float32 → float64 → float32 is exact, so the weights come back unchanged.
    model.double()
    try:
        double = run(
            model,
            past_values.astype(np.float64),
            future_values.astype(np.float64),
            row=0,
            threads=1,
        )
    finally:
        model.float()

    # Step 3: per record, how far float32 drifted from float64. The inputs are given, not computed, so they have no drift. A record with no float64 twin gets no row
    records = dict(window.records) | single
    noise = []
    for name, values in records.items():
        if name.startswith("input."):
            drift = 0.0
        elif name in double:
            drift = float(np.max(np.abs(values.astype(np.float64) - double[name])))
        else:
            continue
        noise.append((name, drift, float(np.max(np.abs(values)))))
    return records, noise
