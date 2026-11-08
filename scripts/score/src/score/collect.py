"""Turn the per-series JSON results into the tracked tables, and compare them with IBM's."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from pathlib import Path

# External
import pandas as pd

# Internal
from score.kind import Kind
from score.unit import SCORES


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

METRICS = ("VUS-PR", "VUS-ROC", "AUC-PR")

# IBM computes only these two
IBM_METRICS = ("VUS-PR", "VUS-ROC")

# IBM's files name the forecast head after its own mode name
IBM_FILE_HEAD = {
    "time": "time",
    "fft": "fft",
    "forecast": "future",
    "ensemble": "ensemble",
}

RUN_COLUMNS = (
    "file_name",
    "kind",
    "source",
    "rows",
    "channels",
    "windows",
    "passes",
    "sliding_window",
    "seconds",
    "model_seconds",
    "peak_rss_kib",
    "batch_size",
    "threads",
    "torch",
    "granite_tsfm",
    "weights_revision",
    "git_sha",
)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def ibm_tuning(lists: Path) -> pd.DataFrame:
    """Read IBM's per-series Tuning results for every head.

    Parameters
    ----------
    lists : Path
        The folder with IBM's files, `$TSPULSE_DATA/lists/ibm`.

    Returns
    -------
    pd.DataFrame
        Columns `file_name`, `head`, `metric`, `ibm_value`.
    """

    frames = []
    for kind in Kind:
        for head, file_head in IBM_FILE_HEAD.items():
            table = pd.read_csv(lists / f"TSB-AD-{kind}-Tuning-{file_head}.csv")
            # The M files call this column `file_list`
            table = table.rename(columns={table.columns[0]: "file_name"})
            long = table.melt(
                id_vars="file_name",
                value_vars=list(IBM_METRICS),
                var_name="metric",
                value_name="ibm_value",
            )
            frames.append(long.assign(head=head))
    return pd.concat(frames, ignore_index=True)


def collect(out: Path, lists: Path, tables: Path) -> pd.DataFrame:
    """Write `python.csv` and `python-runs.csv` from the JSON results.

    Parameters
    ----------
    out : Path
        The folder with the per-series results.
    lists : Path
        The folder with IBM's files.
    tables : Path
        The folder to write the tables into.

    Returns
    -------
    pd.DataFrame
        The tidy scores table, as written.
    """

    records = [json.loads(path.read_text()) for path in sorted(out.glob("*.json"))]
    if not records:
        raise FileNotFoundError(f"No results in {out}")

    rows = [
        {
            "file_name": r["file_name"],
            "kind": r["kind"],
            "source": r["source"],
            "head": head,
            "metric": metric,
            "value": r["metrics"][head][metric],
        }
        for r in records
        for head in SCORES
        for metric in METRICS
    ]
    scores = pd.DataFrame(rows).merge(
        ibm_tuning(lists), on=["file_name", "head", "metric"], how="left"
    )
    runs = pd.DataFrame(
        [{column: r[column] for column in RUN_COLUMNS} for r in records]
    )

    tables.mkdir(parents=True, exist_ok=True)
    scores.to_csv(tables / "python.csv", index=False)
    runs.to_csv(tables / "python-runs.csv", index=False)
    return scores


def compare(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarize each head and metric against IBM's, as the match rule reads them.

    Parameters
    ----------
    scores : pd.DataFrame
        The tidy scores table.

    Returns
    -------
    pd.DataFrame
        Per kind, head and metric: plain means of ours and IBM's, means per source, and the largest per-series difference.
    """

    paired = scores.dropna(subset=["ibm_value"]).assign(
        difference=lambda t: (t["value"] - t["ibm_value"]).abs()
    )
    keys = ["kind", "head", "metric"]
    plain = paired.groupby(keys).agg(
        series=("value", "size"),
        ours=("value", "mean"),
        ibm=("ibm_value", "mean"),
        largest=("difference", "max"),
    )
    per_source = (
        paired.groupby([*keys, "source"])[["value", "ibm_value"]]
        .mean()
        .groupby(keys)
        .mean()
    )
    summary = plain.assign(
        mean_difference=plain["ours"] - plain["ibm"],
        ours_per_source=per_source["value"],
        ibm_per_source=per_source["ibm_value"],
    )
    return summary.reset_index()
