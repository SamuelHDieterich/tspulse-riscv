"""Which TSB-AD series the benchmark uses, and how the table of them is built."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import random
import re
from dataclasses import astuple, dataclass, fields
from itertools import groupby
from pathlib import Path

# External
import pandas as pd


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# Fixed so the sample can be regenerated
SEED = 2026

# The ensemble runs 13 model passes per window
PASSES_PER_WINDOW = 13

# The model's context length, so a series gives this many fewer windows than rows
WINDOW_ROWS = 512

# About a third of TSB-AD-U-Eva, which keeps the final runs affordable
HELD_U = 120

# Multivariate runs only the baseline and final picks, so it stays small
WORKING_M = 12
HELD_M = 20

# Leaves out the wide, long series that would take most of the multivariate budget
HELD_M_MAX_WINDOWS = 300_000

# Caps the quick screen of an approximate variant, at about a tenth of the full cost
SCREEN_WINDOWS = 5_000

# Measured with 8 one-thread PyTorch workers on the i5-13450HX, only for the printed estimate
PASSES_PER_SECOND = 4_000

NAME = re.compile(r"\d+_(.+?)_id_\d+_(.+?)_tr_\d+_1st_\d+\.csv")

# The file suffixes of IBM's heads, in the table's column order. `future` holds the forecast head.
HEADS = ("time", "fft", "future", "ensemble")


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Failure(Exception):
    """A step failed in a way the user can act on, and the message says how."""


@dataclass(frozen=True)
class Series:
    """One TSB-AD series, as far as selection needs to know it.

    Attributes
    ----------
    file_name : str
        The CSV's name.
    kind : str
        `U` for univariate, `M` for multivariate.
    source : str
        The original dataset, from the name.
    domain : str
        The application domain, from the name.
    rows : int
        How many rows it has.
    channels : int
        How many columns it has, without `Label`.
    first_label : int
        The first row labelled as an anomaly.
    """

    file_name: str
    kind: str
    source: str
    domain: str
    rows: int
    channels: int
    first_label: int

    @property
    def windows(self) -> int:
        """How many windows the pipeline slides over it."""

        return self.rows - WINDOW_ROWS

    @property
    def passes(self) -> int:
        """How many single-channel model passes a whole run takes."""

        return self.windows * self.channels * PASSES_PER_WINDOW


@dataclass(frozen=True)
class Row:
    """One line of the table. The field order is the column order."""

    file_name: str
    kind: str
    group: str
    order: int
    source: str
    domain: str
    rows: int
    channels: int
    windows: int
    passes: int
    screen_start: int | None
    screen_stop: int | None
    screen_passes: int | None
    ibm_time: str
    ibm_fft: str
    ibm_forecast: str
    ibm_ensemble: str


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_series(root: Path, kind: str, name: str) -> Series:
    """Read what selection needs from one series, without touching its data columns.

    Parameters
    ----------
    root : Path
        The data folder.
    kind : str
        `U` or `M`.
    name : str
        The CSV's name.

    Returns
    -------
    Series
        The series.

    Raises
    ------
    Failure
        When the file is missing or its name doesn't parse.
    """

    match = NAME.fullmatch(name)
    if match is None:
        raise Failure(f"{name} doesn't follow the TSB-AD naming scheme.")

    path = root / "datasets" / f"TSB-AD-{kind}" / name
    if not path.exists():
        raise Failure(f"{path} is missing. Run download-datasets.")

    columns = pd.read_csv(path, nrows=0).columns
    labels = pd.read_csv(path, usecols=["Label"], dtype={"Label": int})["Label"]
    return Series(
        file_name=name,
        kind=kind,
        source=match[1],
        domain=match[2],
        rows=len(labels),
        channels=len(columns) - 1,
        first_label=int(labels.argmax()),
    )


def read_split(root: Path, kind: str, split: str) -> list[Series]:
    """Read every series in one of TSB-AD's lists.

    Parameters
    ----------
    root : Path
        The data folder.
    kind : str
        `U` or `M`.
    split : str
        `Tuning` or `Eva`.

    Returns
    -------
    list[Series]
        The series, in the list's order.

    Raises
    ------
    Failure
        When the list or one of its files is missing, or a name doesn't parse.
    """

    path = root / "lists" / "tsb-ad" / f"TSB-AD-{kind}-{split}.csv"
    if not path.exists():
        raise Failure(f"{path} is missing. Run download-lists.")
    names = pd.read_csv(path, dtype=str)["file_name"]
    return [read_series(root, kind, name) for name in names]


def read_scores(root: Path, kind: str, split: str) -> list[dict[str, str]]:
    """Read IBM's VUS-PR per series, as text so nothing reformats it.

    Parameters
    ----------
    root : Path
        The data folder.
    kind : str
        `U` or `M`.
    split : str
        `Tuning` or `Eva`.

    Returns
    -------
    list[dict[str, str]]
        One mapping from file name to score for each of `HEADS`.

    Raises
    ------
    Failure
        When one of IBM's files is missing.
    """

    scores = []
    for head in HEADS:
        path = root / "lists" / "ibm" / f"TSB-AD-{kind}-{split}-{head}.csv"
        if not path.exists():
            raise Failure(f"{path} is missing. Run download-lists.")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        # The first column is `file_name` in the U files and `file_list` in the M files
        scores.append(dict(zip(frame.iloc[:, 0], frame["VUS-PR"], strict=True)))
    return scores


def balanced(series: list[Series]) -> list[Series]:
    """Order series so that any prefix is close to proportional per source.

    Each source's series are shuffled and spread evenly over the range 0 to 1, so sources take turns in proportion to their size.
    The result depends only on `SEED` and the file names.

    Parameters
    ----------
    series : list[Series]
        The series, in any order.

    Returns
    -------
    list[Series]
        The same series, ordered.
    """

    rng = random.Random(SEED)
    by_source = sorted(series, key=lambda item: (item.source, item.file_name))

    spread = []
    for _, members in groupby(by_source, key=lambda item: item.source):
        group = list(members)
        rng.shuffle(group)
        spread += [((i + 0.5) / len(group), item) for i, item in enumerate(group)]

    # Every series draws exactly one tie-break, in this order, so the draws depend only on the seed and the names
    keyed = [(key, rng.random(), item.file_name, item) for key, item in spread]
    keyed.sort(key=lambda entry: entry[:3])
    return [item for *_, item in keyed]


def screen_span(series: Series) -> tuple[int, int, int]:
    """Choose the rows a quick screen runs on.

    Parameters
    ----------
    series : Series
        The series.

    Returns
    -------
    tuple[int, int, int]
        The first row, the row after the last, and the passes that takes.
    """

    if series.windows <= SCREEN_WINDOWS:
        start, stop = 0, series.rows
    else:
        # A quarter of the span comes before the first label, so the model sees normal data first
        span = WINDOW_ROWS + SCREEN_WINDOWS
        start = min(max(series.first_label - span // 4, 0), series.rows - span)
        stop = start + span

    return start, stop, (stop - start - WINDOW_ROWS) * PASSES_PER_WINDOW


def to_rows(
    series: list[Series],
    group: str,
    scores: list[dict[str, str]],
    screened: bool = False,
) -> list[Row]:
    """Turn one ordered group into table rows.

    Parameters
    ----------
    series : list[Series]
        The group, in its final order.
    group : str
        `working` or `held`.
    scores : list[dict[str, str]]
        IBM's scores, from `read_scores`.
    screened : bool
        Whether this group has screen spans.

    Returns
    -------
    list[Row]
        One row per series.

    Raises
    ------
    Failure
        When a series has no score from IBM.
    """

    rows = []
    for order, item in enumerate(series):
        try:
            time, fft, forecast, ensemble = (head[item.file_name] for head in scores)
        except KeyError as error:
            raise Failure(f"{item.file_name} has no score in IBM's files.") from error

        start, stop, passes = screen_span(item) if screened else (None, None, None)
        rows.append(
            Row(
                file_name=item.file_name,
                kind=item.kind,
                group=group,
                order=order,
                source=item.source,
                domain=item.domain,
                rows=item.rows,
                channels=item.channels,
                windows=item.windows,
                passes=item.passes,
                screen_start=start,
                screen_stop=stop,
                screen_passes=passes,
                ibm_time=time,
                ibm_fft=fft,
                ibm_forecast=forecast,
                ibm_ensemble=ensemble,
            )
        )
    return rows


def select(root: Path) -> list[Row]:
    """Choose every group.

    Parameters
    ----------
    root : Path
        The data folder.

    Returns
    -------
    list[Row]
        The table, ordered by kind, then group, then `order`.

    Raises
    ------
    Failure
        When a list, a series or a score is missing, or a name doesn't parse.
    """

    u_working = sorted(read_split(root, "U", "Tuning"), key=lambda s: s.file_name)
    u_held = balanced(read_split(root, "U", "Eva"))[:HELD_U]

    m_tuning = read_split(root, "M", "Tuning")
    m_working = sorted(m_tuning, key=lambda s: (s.passes, s.file_name))[:WORKING_M]

    m_eva = read_split(root, "M", "Eva")
    m_held = balanced(
        [s for s in m_eva if s.windows * s.channels <= HELD_M_MAX_WINDOWS]
    )[:HELD_M]

    return (
        to_rows(u_working, "working", read_scores(root, "U", "Tuning"), screened=True)
        + to_rows(u_held, "held", read_scores(root, "U", "Eva"))
        + to_rows(m_working, "working", read_scores(root, "M", "Tuning"))
        + to_rows(m_held, "held", read_scores(root, "M", "Eva"))
    )


def write_table(rows: list[Row], output: Path) -> None:
    """Write the table, creating its folder.

    Parameters
    ----------
    rows : list[Row]
        The table.
    output : Path
        The CSV to write.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(field.name for field in fields(Row))
        writer.writerows(astuple(row) for row in rows)
