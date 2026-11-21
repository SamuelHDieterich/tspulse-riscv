"""The tracked tables of a label: one row per series, scope, head and metric, and one per counted window.

With `--against`, each row also holds the other label's value for the same series, and the difference or ratio.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import io
from dataclasses import dataclass, fields, replace
from pathlib import Path

# Internal
from score.benchmark.scope import Scope, _read_table
from score.benchmark.scoring import ScopeScore
from score.kind import Kind
from score.spike_results import _replace


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The Python baseline's table, under the git top level
PYTHON_TABLE = Path("benchmark/baseline/python.csv")

# Where each label's tables go, under the git top level
TABLES_FOLDER = Path("benchmark/results")
ACCURACY_TABLE = "accuracy.csv"
INSTRUCTIONS_TABLE = "instructions.csv"

# The metrics a table holds, as the baseline names them, and the `SeriesMetrics` field each reads
METRICS = {"VUS-PR": "vus_pr", "VUS-ROC": "vus_roc", "AUC-PR": "auc_pr"}

# IBM published these two only
IBM_METRICS = ("VUS-PR", "VUS-ROC")

# IBM's per-head Eval scores, under the data folder: `TSB-AD-<kind>-Eva-<file>.csv`, with each head's file and each kind's key column
IBM_EVAL_FOLDER = Path("lists/ibm")
IBM_EVAL_FILE = "TSB-AD-{kind}-Eva-{name}.csv"
IBM_HEAD_FILES = {
    "time": "time",
    "fft": "fft",
    "forecast": "future",
    "ensemble": "ensemble",
}
IBM_KEY_COLUMNS = {Kind.U: "file_name", Kind.M: "file_list"}

# The columns that only a table written with `--against` has
AGAINST_COLUMNS = {
    ACCURACY_TABLE: ("against_value", "difference"),
    INSTRUCTIONS_TABLE: ("against_instructions", "ratio"),
}


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class AccuracyRow:
    """One metric of one score of one series' scope.

    Attributes
    ----------
    file_name : str
        The series' CSV name.
    kind : str
        The archive it comes from.
    source : str
        The original dataset.
    scope : str
        The scope scored.
    head : str
        The score: a head or `ensemble`.
    metric : str
        The metric, named as the baseline names it.
    value : float
        The C's value, scored from the host build's errors.
    python_value : float or None
        The Python baseline's value, for `whole-python` rows.
    ibm_value : float or None
        IBM's published value, for `whole-python` and `held` VUS rows (its Eval tables for `held`).
    against_value : float or None
        The `--against` label's value for the same row.
    difference : float or None
        `value − against_value`.
    """

    file_name: str
    kind: str
    source: str
    scope: str
    head: str
    metric: str
    value: float
    python_value: float | None
    ibm_value: float | None
    against_value: float | None = None
    difference: float | None = None


@dataclass(frozen=True)
class InstructionRow:
    """One counted window's instructions.

    Attributes
    ----------
    series : str
        The series' CSV name without `.csv`.
    kind : str
        The archive it comes from.
    channels : int
        How many channels it has.
    window : int
        The window counted.
    program_instructions : int
        The instructions of the window's one model run.
    per_channel : float
        `program_instructions / channels`.
    identical : bool or str
        Whether the host's and Spike's errors were byte-identical, or `not checked`.
    spike_seconds : float
        The Spike run's wall time.
    machine : str
        The computer that ran it.
    against_instructions : int or None
        The `--against` label's instructions for the same window.
    ratio : float or None
        `program_instructions / against_instructions`.
    """

    series: str
    kind: str
    channels: int
    window: int
    program_instructions: int
    per_channel: float
    identical: bool | str
    spike_seconds: float
    machine: str
    against_instructions: int | None = None
    ratio: float | None = None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _optional_float(text: str) -> float | None:
    """Read a table cell that may be empty.

    Parameters
    ----------
    text : str
        The cell.

    Returns
    -------
    float or None
        Its value, None when empty.
    """

    return float(text) if text else None


def read_baseline(top: Path) -> dict[tuple[str, str, str], tuple[float, float | None]]:
    """Read the Python baseline's values and IBM's.

    Parameters
    ----------
    top : Path
        The worktree's top level.

    Returns
    -------
    dict[tuple[str, str, str], tuple[float, float | None]]
        `(file_name, head, metric)` to the Python value and IBM's value, None where IBM published none.

    Raises
    ------
    typer.Exit
        With code 1 when the table can't be read.
    """

    return {
        (row["file_name"], row["head"], row["metric"]): (
            float(row["value"]),
            _optional_float(row["ibm_value"]),
        )
        for row in _read_table(top / PYTHON_TABLE)
    }


def read_ibm_eval(data: Path) -> dict[tuple[str, str, str], float]:
    """Read IBM's Eval scores of every head, for both archives.

    Parameters
    ----------
    data : Path
        The folder with the data.

    Returns
    -------
    dict[tuple[str, str, str], float]
        `(file_name, head, metric)` to IBM's VUS-PR or VUS-ROC.

    Raises
    ------
    typer.Exit
        With code 1 when a file can't be read.
    """

    scores = {}
    for kind, key in IBM_KEY_COLUMNS.items():
        for head, name in IBM_HEAD_FILES.items():
            path = data / IBM_EVAL_FOLDER / IBM_EVAL_FILE.format(kind=kind, name=name)
            for row in _read_table(path):
                for metric in IBM_METRICS:
                    scores[(row[key], head, metric)] = float(row[metric])
    return scores


def accuracy_rows(
    scored: list[ScopeScore],
    baseline: dict[tuple[str, str, str], tuple[float, float | None]],
    ibm_eval: dict[tuple[str, str, str], float],
    compare_python: bool,
) -> list[AccuracyRow]:
    """Turn scored scopes into table rows, with the baseline beside the `whole-python` ones and IBM's Eval scores beside the `held` ones.

    Parameters
    ----------
    scored : list[ScopeScore]
        The scored scopes.
    baseline : dict
        What `read_baseline` returned.
    ibm_eval : dict
        What `read_ibm_eval` returned, or empty when no held-back scope was scored.
    compare_python : bool
        Whether to fill the baseline's values. They are for the whole series, so a window-capped run leaves them empty. IBM's Eval scores follow the same rule.

    Returns
    -------
    list[AccuracyRow]
        One row per series, scope, score and metric.
    """

    rows = []
    for item in scored:
        file_name = f"{item.series.stem}.csv"
        for head, metrics in item.metrics.items():
            for metric, field in METRICS.items():
                # The baseline ran each series whole, so only that scope sits beside it, and only when it ran whole here too
                python_value = ibm_value = None
                if compare_python and item.scope == Scope.WHOLE_PYTHON:
                    python_value, ibm_value = baseline.get(
                        (file_name, head, metric), (None, None)
                    )
                    if metric not in IBM_METRICS:
                        ibm_value = None
                # No Python run touched the held-back series, so only IBM's score sits beside them
                elif compare_python and item.scope == Scope.HELD:
                    ibm_value = ibm_eval.get((file_name, head, metric))
                rows.append(
                    AccuracyRow(
                        file_name=file_name,
                        kind=str(item.series.kind),
                        source=item.series.source,
                        scope=str(item.scope),
                        head=head,
                        metric=metric,
                        value=getattr(metrics, field),
                        python_value=python_value,
                        ibm_value=ibm_value,
                    )
                )
    return rows


def read_against_accuracy(path: Path) -> dict[tuple[str, str, str, str], float]:
    """Read another label's accuracy table.

    Parameters
    ----------
    path : Path
        Its `accuracy.csv`.

    Returns
    -------
    dict[tuple[str, str, str, str], float]
        `(file_name, scope, head, metric)` to the value.

    Raises
    ------
    typer.Exit
        With code 1 when the table can't be read.
    """

    return {
        (row["file_name"], row["scope"], row["head"], row["metric"]): float(
            row["value"]
        )
        for row in _read_table(path)
    }


def read_against_instructions(path: Path) -> dict[tuple[str, int], int]:
    """Read another label's instruction table.

    Parameters
    ----------
    path : Path
        Its `instructions.csv`.

    Returns
    -------
    dict[tuple[str, int], int]
        `(series, window)` to the program instructions.

    Raises
    ------
    typer.Exit
        With code 1 when the table can't be read.
    """

    return {
        (row["series"], int(row["window"])): int(row["program_instructions"])
        for row in _read_table(path)
    }


def compare_accuracy(
    rows: list[AccuracyRow], against: dict[tuple[str, str, str, str], float]
) -> list[AccuracyRow]:
    """Add the other label's value and the difference to each row it has.

    Parameters
    ----------
    rows : list[AccuracyRow]
        This label's rows.
    against : dict
        What `read_against_accuracy` returned.

    Returns
    -------
    list[AccuracyRow]
        The rows, with `against_value` and `difference` where the other label scored the same row.
    """

    compared = []
    for row in rows:
        other = against.get((row.file_name, row.scope, row.head, row.metric))
        if other is None:
            compared.append(row)
        else:
            compared.append(
                replace(row, against_value=other, difference=row.value - other)
            )
    return compared


def compare_instructions(
    rows: list[InstructionRow], against: dict[tuple[str, int], int]
) -> list[InstructionRow]:
    """Add the other label's instructions and the ratio to each row it has.

    Parameters
    ----------
    rows : list[InstructionRow]
        This label's rows.
    against : dict
        What `read_against_instructions` returned.

    Returns
    -------
    list[InstructionRow]
        The rows, with `against_instructions` and `ratio` where the other label counted the same window.
    """

    compared = []
    for row in rows:
        other = against.get((row.series, row.window))
        if other is None:
            compared.append(row)
        else:
            compared.append(
                replace(
                    row,
                    against_instructions=other,
                    ratio=row.program_instructions / other,
                )
            )
    return compared


def write_table(
    path: Path,
    row_type: type[AccuracyRow] | type[InstructionRow],
    rows: list[AccuracyRow] | list[InstructionRow],
    against: bool,
) -> None:
    """Write a table, replaced whole, with the `--against` columns only when asked.

    Parameters
    ----------
    path : Path
        The table, named `accuracy.csv` or `instructions.csv`.
    row_type : type
        The rows' class, which names the columns even when there are no rows.
    rows : list[AccuracyRow] or list[InstructionRow]
        Its rows.
    against : bool
        Whether the table holds the `--against` columns.

    Raises
    ------
    OSError
        When the table can't be written.
    """

    columns = [
        field.name
        for field in fields(row_type)
        if against or field.name not in AGAINST_COLUMNS[path.name]
    ]
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        cells = [getattr(row, name) for name in columns]
        writer.writerow(["" if cell is None else cell for cell in cells])
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace(path, text.getvalue())
