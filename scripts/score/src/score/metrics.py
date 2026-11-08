"""IBM's VUS metrics for TSB-AD, restyled with the arithmetic unchanged.

Source: granite-tsfm `e4c03aa226bb`, `notebooks/hfdemo/tspulse/anomaly_detection/utility/metrics.py`, Apache-2.0, itself adapted from TSB-UAD (https://github.com/TheDatumOrg/TSB-UAD).
Only the `opt` path is kept, which is the default IBM's `run_experiment.py` uses.
`auc_pr` and `score_series` are additions: the first is scikit-learn's average precision, the second is the scaling and window search `run_experiment.py` and `utility/model.py` do before calling `get_metrics`.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass

# External
import numpy as np
from scipy.signal import argrelextrema
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# How many score thresholds each curve is drawn through
THRESHOLDS = 250

# The VUS window when the autocorrelation has no usable peak
FALLBACK_WINDOW = 125


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class SeriesMetrics:
    """One score's metrics on one series.

    Attributes
    ----------
    vus_pr : float
        The volume under the range precision-recall surface.
    vus_roc : float
        The volume under the range ROC surface.
    auc_pr : float
        The area under the plain precision-recall curve, with no tolerance window.
    sliding_window : int
        The widest tolerance window the volumes were taken over.
    """

    vus_pr: float
    vus_roc: float
    auc_pr: float
    sliding_window: int


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀

# The range curves behind VUS, as IBM computes them


def _label_ranges(label: np.ndarray) -> list[tuple[int, int]]:
    """List the first and last row of each labelled run.

    Parameters
    ----------
    label : np.ndarray
        One 0 or 1 label per row.

    Returns
    -------
    list[tuple[int, int]]
        One `(first, last)` pair per run of 1s, both inclusive, in row order.
    """

    starts = np.where(np.diff(label) == 1)[0] + 1
    (ends,) = np.where(np.diff(label) == -1)
    if len(ends):
        if not len(starts) or ends[0] < starts[0]:
            # The labels open inside an anomaly
            starts = np.concatenate([[0], starts])
    if len(starts):
        if not len(ends) or ends[-1] < starts[-1]:
            # The labels close inside an anomaly
            ends = np.concatenate([ends, [len(label) - 1]])
    return list(zip(starts, ends, strict=False))


def _merge_ranges(
    label: np.ndarray, ranges: list[tuple[int, int]], window: int
) -> list[tuple[int, int]]:
    """Widen each range by half a window on both sides, merging ranges that then touch or overlap.

    Parameters
    ----------
    label : np.ndarray
        One label per row, read only for its length.
    ranges : list[tuple[int, int]]
        The labelled runs, from `_label_ranges`. Must not be empty.
    window : int
        The tolerance window, half of which is added on each side.

    Returns
    -------
    list[tuple[int, int]]
        The widened ranges, both ends inclusive and clipped to the series.
    """

    start = max(ranges[0][0] - window // 2, 0)
    merged = []
    for i in range(len(ranges) - 1):
        if ranges[i][1] + window // 2 < ranges[i + 1][0] - window // 2:
            merged.append((start, ranges[i][1] + window // 2))
            start = ranges[i + 1][0] - window // 2
    merged.append(
        (start, min(ranges[len(ranges) - 1][1] + window // 2, len(label) - 1))
    )
    return merged


def _soften_edges(
    label: np.ndarray, ranges: list[tuple[int, int]], window: int = 5
) -> np.ndarray:
    """Give the rows within half a window of each range a partial label that fades with distance.

    Parameters
    ----------
    label : np.ndarray
        One 0 or 1 label per row.
    ranges : list[tuple[int, int]]
        The labelled runs, from `_label_ranges`.
    window : int
        The tolerance window `ℓ`.

    Returns
    -------
    np.ndarray
        The softened labels, between 0 and 1.

    Notes
    -----
    A row `d` rows before the start or after the end of a range, with `0 < d ≤ ℓ // 2`, gets `√(1 − d / ℓ)` from that range.
    Rows close to two ranges add both shares, and every soft label is capped at 1:

    `ỹ_t = min(1, y_t + Σ_ranges √(1 − d_t / ℓ))`
    """

    soft = label.copy().astype(float)
    length = len(soft)

    for k in range(len(ranges)):
        start = ranges[k][0]
        end = ranges[k][1]

        after = np.arange(end + 1, min(end + window // 2 + 1, length))
        soft[after] += np.sqrt(1 - (after - end) / (window))

        before = np.arange(max(start - window // 2, 0), start)
        soft[before] += np.sqrt(1 - (start - before) / (window))

    soft = np.minimum(np.ones(length), soft)
    return soft


def _range_auc_volume(
    labels_original: np.ndarray,
    score: np.ndarray,
    window_size: int,
    thresholds: int = THRESHOLDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Draw the range ROC and PR curves for every window from 0 to `window_size`, and average their areas.

    Parameters
    ----------
    labels_original : np.ndarray
        One 0 or 1 label per row.
    score : np.ndarray
        One anomaly score per row.
    window_size : int
        The widest tolerance window.
    thresholds : int
        How many score thresholds each curve is drawn through.

    Returns
    -------
    tpr_3d : np.ndarray
        True-positive rates, one row per window and one column per threshold, with the curve's end points.
    fpr_3d : np.ndarray
        False-positive rates, in the same layout as `tpr_3d`.
    prec_3d : np.ndarray
        Precisions, one row per window and one column per threshold, with the starting point.
    window_3d : np.ndarray
        The windows, 0 to `window_size`.
    vus_roc : float
        The mean area under the range ROC curves, VUS-ROC.
    vus_pr : float
        The mean area under the range PR curves, VUS-PR.

    Notes
    -----
    A plain ROC or PR curve grades each row on its own, so a detection a few rows off an anomaly counts as a miss.
    The range curves forgive that: rows near a labelled range get a partial label, and recall is scaled by how many ranges were found at all.
    Doing this for every tolerance window from 0 to `window_size` and averaging the areas gives a volume under a surface, so no single window has to be chosen.

    The comments in the body give each step's equation, as the code computes it, with this notation:

    - `T` rows, indexed by `t`, with labels `y_t` in {0, 1} and scores `s_t`.
    - `P = Σ_t y_t`, the labelled rows.
    - `K` thresholds `θ_k`, indexed by `k` from 1 to `K`, and `p_t = [s_t ≥ θ_k]`, 1 when row `t` is flagged at that threshold.
    - `N_k = Σ_t p_t`, the rows flagged at threshold `k`.
    - `ℓ` a tolerance window, from 0 to `L = window_size`.
    - `ỹ_t`, the soft label of row `t` for window `ℓ`, from `_soften_edges`.
    - `R`, one range of the labelled ranges widened by `ℓ // 2`, from `_merge_ranges`.

    References
    ----------
    .. [1] J. Paparrizos, P. Boniol, T. Palpanas, R. S. Tsay, A. Elmore and M. J. Franklin, "Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection", PVLDB 15(11), 2022. https://doi.org/10.14778/3551793.3551830
    .. [2] N. Tatbul, T. J. Lee, S. Zdonik, M. Alam and J. Gottschlich, "Precision and Recall for Time Series", NeurIPS 2018, the source of the existence reward. https://arxiv.org/abs/1803.03639
    .. [3] ROC curves: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    .. [4] Precision and recall: https://en.wikipedia.org/wiki/Precision_and_recall
    .. [5] Average precision, the area under the precision-recall curve: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
    .. [6] The trapezoidal rule, which Step 5d applies to the ROC curve: https://en.wikipedia.org/wiki/Trapezoidal_rule
    """

    # Step 1: the windows to sweep, the labelled ranges, and those ranges widened by the largest window, which hold every row a partial label can reach
    window_3d = np.arange(0, window_size + 1, 1)
    positives = np.sum(labels_original)
    ranges = _label_ranges(labels_original)
    widest = _merge_ranges(labels_original, ranges, window_size)

    # Step 2: the thresholds are the scores at evenly spaced ranks, from the highest score down to the lowest
    #   θ_k = s_(i_k),  i_k = ⌊(k − 1)(T − 1) / (K − 1)⌋
    # where s_(i) is the i-th score in descending order, counted from 0
    score_sorted = -np.sort(-score)

    # Step 3: room for one curve per window, plus each curve's area
    tpr_3d = np.zeros((window_size + 1, thresholds + 2))
    fpr_3d = np.zeros((window_size + 1, thresholds + 2))
    prec_3d = np.zeros((window_size + 1, thresholds + 1))

    auc_3d = np.zeros(window_size + 1)
    ap_3d = np.zeros(window_size + 1)

    # Always zero: kept so the true-positive sum adds the same terms in the same order as IBM's
    tp = np.zeros(thresholds)
    n_pred = np.zeros(thresholds)

    # Step 4: how many rows each threshold flags. This doesn't depend on the window, so it is counted once.
    #   N_k = Σ_t [s_t ≥ θ_k]
    for k, i in enumerate(np.linspace(0, len(score) - 1, thresholds).astype(int)):
        threshold = score_sorted[i]
        pred = score >= threshold
        n_pred[k] = np.sum(pred)

    # Step 5: one ROC and one PR curve per window
    for window in window_3d:
        # Step 5a: rows within half a window of a range get a partial label, and each range is widened by the same half window
        labels_extended = _soften_edges(labels_original, ranges, window)
        merged = _merge_ranges(labels_extended, ranges, window)

        # Point 0 starts the ROC curve at (0, 0) and the PR curve at precision 1
        #   (TPR_0, FPR_0) = (0, 0)      precision_0 = 1
        tf_list = np.zeros((thresholds + 2, 2))
        precision_list = np.ones(thresholds + 1)
        j = 0

        # Step 5b: one point on each curve per threshold
        for i in np.linspace(0, len(score) - 1, thresholds).astype(int):
            threshold = score_sorted[i]
            pred = score >= threshold
            labels = labels_extended.copy()
            existence = 0

            # A partial label counts only where the row is flagged, and a widened range is found if any of its rows is flagged.
            # With the next loop, each row gets the weight
            #   w_t = 1         if y_t = 1
            #   w_t = ỹ_t · p_t  otherwise
            for seg in merged:
                labels[seg[0] : seg[1] + 1] = (
                    labels_extended[seg[0] : seg[1] + 1] * pred[seg[0] : seg[1] + 1]
                )
                if (pred[seg[0] : (seg[1] + 1)] > 0).any():
                    existence += 1
            # The labelled rows themselves always count in full, flagged or not
            for seg in ranges:
                labels[seg[0] : seg[1] + 1] = 1

            # Soft true positives: the label weight of the flagged rows. False positives are the flagged rows minus that weight, so a flagged row with a partial label counts partly as each.
            #   TP = Σ_t w_t · p_t      FP = N_k − TP      W = Σ_t w_t
            # `widest` holds every row with a nonzero weight, so summing over it is summing over all rows
            true_positives = 0
            n_labels = 0
            for seg in widest:
                true_positives += np.dot(
                    labels[seg[0] : seg[1] + 1], pred[seg[0] : seg[1] + 1]
                )
                n_labels += np.sum(labels[seg[0] : seg[1] + 1])

            true_positives += tp[j]
            false_positives = n_pred[j] - true_positives

            # The share of widened ranges found at all, which scales recall down when whole anomalies are missed [2]
            #   E = |{R : p_t = 1 for some t in R}| / |{R}|
            existence_ratio = existence / len(merged)

            # The positives are the mean of the hard and the soft label counts, and the negatives are every other row
            #   P' = (P + W) / 2
            #   recall = min(TP / P', 1)      TPR_k = recall · E
            #   FPR_k = FP / (T − P')         precision_k = TP / N_k
            positives_new = (positives + n_labels) / 2
            recall = min(true_positives / positives_new, 1)

            tpr = recall * existence_ratio
            negatives_new = len(labels) - positives_new
            fpr = false_positives / negatives_new

            precision = true_positives / n_pred[j]

            j += 1
            tf_list[j] = [tpr, fpr]
            precision_list[j] = precision

        # Step 5c: without this point the ROC curve would stop short of (1, 1)
        #   (TPR_(K+1), FPR_(K+1)) = (1, 1)
        tf_list[j + 1] = [1, 1]

        tpr_3d[window] = tf_list[:, 0]
        fpr_3d[window] = tf_list[:, 1]
        prec_3d[window] = precision_list

        # Step 5d: the ROC area by the trapezoidal rule over the false-positive rate [6]
        #   AUC_ℓ = Σ_(k=0..K) (FPR_(k+1) − FPR_k) · (TPR_(k+1) + TPR_k) / 2
        width = tf_list[1:, 1] - tf_list[:-1, 1]
        height = (tf_list[1:, 0] + tf_list[:-1, 0]) / 2
        auc_range = np.dot(width, height)
        auc_3d[window] = auc_range

        # Step 5e: the PR area as average precision: each step in recall times the precision reached there, leaving out the (1, 1) point [5]
        #   AP_ℓ = Σ_(k=1..K) (TPR_k − TPR_(k−1)) · precision_k
        width_pr = tf_list[1:-1, 0] - tf_list[:-2, 0]
        height_pr = precision_list[1:]

        ap_range = np.dot(width_pr, height_pr)
        ap_3d[window] = ap_range

    # Step 6: averaging the areas over the windows turns them into volumes [1]
    #   VUS-ROC = Σ_(ℓ=0..L) AUC_ℓ / (L + 1)      VUS-PR = Σ_(ℓ=0..L) AP_ℓ / (L + 1)
    return (
        tpr_3d,
        fpr_3d,
        prec_3d,
        window_3d,
        sum(auc_3d) / len(window_3d),
        sum(ap_3d) / len(window_3d),
    )


# The metrics, and the scaling IBM applies before them


def get_metrics(
    score: np.ndarray,
    labels: np.ndarray,
    sliding_window: int = 100,
    thresholds: int = THRESHOLDS,
) -> dict[str, float]:
    """Compute VUS-PR and VUS-ROC.

    Parameters
    ----------
    score : np.ndarray
        One anomaly score per row.
    labels : np.ndarray
        One 0 or 1 label per row.
    sliding_window : int
        The widest tolerance window, from `find_length_rank`.
    thresholds : int
        How many score thresholds each curve is drawn through.

    Returns
    -------
    dict[str, float]
        `VUS-PR` and `VUS-ROC`.
    """

    *_, vus_roc, vus_pr = _range_auc_volume(
        labels.astype(int), score, sliding_window, thresholds
    )
    return {"VUS-PR": vus_pr, "VUS-ROC": vus_roc}


def find_length_rank(data: np.ndarray, rank: int = 1) -> int:
    """Estimate a series' period from its autocorrelation, which sets the VUS window.

    Parameters
    ----------
    data : np.ndarray
        The series, rows by channels.
    rank : int
        Which autocorrelation peak to take: 1 is the highest.

    Returns
    -------
    int
        The window. Multivariate data always gets 0, and a series with no usable peak gets 125.

    Raises
    ------
    ValueError
        If `rank` is not 0, 1, 2 or 3.
    """

    data = data.squeeze()
    if len(data.shape) > 1:
        return 0
    # IBM returns a one-row window for rank 0, without looking at the series
    if rank == 0:
        return 1
    data = data[: min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # No peaks at all, as in a constant series
    if not len(local_max):
        return FALLBACK_WINDOW

    # Positions in `local_max`, from the highest peak to the lowest. Since `local_max` is in lag order, a larger position means a longer lag.
    sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]
    highest = sorted_local_max[0]

    match rank:
        # The highest peak
        case 1:
            max_local_max = highest
        # The highest peak at a longer lag than the highest, or the highest when there is none
        case 2:
            max_local_max = next(
                (i for i in sorted_local_max[1:] if i > highest), highest
            )
        # IBM's third peak. It reuses the second peak's position in `local_max` as a position in `sorted_local_max`, mixing the two orders. Kept as IBM wrote it.
        case 3:
            second = next((i for i in sorted_local_max[1:] if i > highest), 1)
            max_local_max = next(
                (i for i in sorted_local_max[second:] if i > sorted_local_max[second]),
                highest,
            )
        case _:
            raise ValueError(f"rank must be 0, 1, 2 or 3, got {rank}")

    # IBM keeps only peaks between 3 and 300 positions past `base`, and falls back otherwise
    if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
        return FALLBACK_WINDOW
    return local_max[max_local_max] + base


def auc_pr(score: np.ndarray, labels: np.ndarray) -> float:
    """Compute the area under the plain precision-recall curve, with no tolerance window.

    Parameters
    ----------
    score : np.ndarray
        One anomaly score per row.
    labels : np.ndarray
        One 0 or 1 label per row.

    Returns
    -------
    float
        Average precision.
    """

    return float(average_precision_score(labels, score))


def score_series(
    score: np.ndarray, labels: np.ndarray, data: np.ndarray
) -> SeriesMetrics:
    """Scale a raw pipeline score as IBM's benchmark does, and compute its metrics.

    Parameters
    ----------
    score : np.ndarray
        The pipeline's anomaly score, one per row.
    labels : np.ndarray
        One 0 or 1 label per row.
    data : np.ndarray
        The series, rows by channels, which sets the VUS window.

    Returns
    -------
    SeriesMetrics
        VUS-PR, VUS-ROC, AUC-PR and the window used.
    """

    # Copied so the inputs to `get_metrics` match IBM's, though neither step changes the order of the scores
    norm_value = np.nanmax(np.asarray(score), axis=0, keepdims=True) + 1e-5
    scaled = score / norm_value
    scaled = (
        MinMaxScaler(feature_range=(0, 1)).fit_transform(scaled.reshape(-1, 1)).ravel()
    )

    window = find_length_rank(data, rank=1)
    vus = get_metrics(scaled, labels, sliding_window=window)
    return SeriesMetrics(
        vus_pr=float(vus["VUS-PR"]),
        vus_roc=float(vus["VUS-ROC"]),
        auc_pr=auc_pr(scaled, labels),
        sliding_window=int(window),
    )
