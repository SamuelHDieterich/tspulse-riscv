"""A held-back label's summary: the Spike check, the instructions and the VUS-PR of each kind and head, paired against the stack's."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

# Internal
from score.benchmark.scope import Scope, Series
from score.benchmark.tables import ACCURACY_TABLE, INSTRUCTIONS_TABLE
from score.kind import Kind
from score.spike_check import HELD_GROUP
from score.variants.building import ResolvedVariant
from score.variants.held_record import (
    HeadScore,
    HeldInstructions,
    HeldSummary,
    Pairing,
    SpikeCheck,
    write_held_summary,
)
from score.variants.listing import VariantError
from score.variants.state import missing_stages, tables_folder
from score.variants.summary import (
    HEAD_COLUMN,
    KIND_COLUMN,
    METRIC_COLUMN,
    MISSING_NOT_RUN,
    PER_CHANNEL_COLUMN,
    SCOPE_COLUMN,
    SERIES_COLUMN,
    STATUS_DONE,
    VALUE_COLUMN,
    VUS_HEADS,
    VUS_METRIC,
    WINDOW_COLUMN,
    _build,
    _mean,
    _numbers,
    _read_table,
)
from score.variants.summary_record import summary_path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The scope the VUS-PR is read in
VUS_SCOPE = Scope.HELD

# The kinds the summary keeps apart, as TSB-AD and IBM report them
KINDS = (Kind.U, Kind.M)

# The columns of the tables the summary reads, beside the ones `summary.py` names
FILE_NAME_COLUMN = "file_name"
SOURCE_COLUMN = "source"
IBM_COLUMN = "ibm_value"
IDENTICAL_COLUMN = "identical"

# How `instructions.csv` spells the two outcomes of the host against Spike comparison; any other text means it wasn't made
IDENTICAL_YES = "True"
IDENTICAL_NO = "False"

# A series' CSV name ends with this, which the summary drops
CSV_EXTENSION = ".csv"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class _SeriesScore:
    """One series' VUS-PR of one head.

    Attributes
    ----------
    source : str
        The series' original dataset.
    value : float
        The C's VUS-PR.
    ibm : float or None
        IBM's Eval VUS-PR, None when the table has none.
    """

    source: str
    value: float
    ibm: float | None


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def held_status_text(summary: HeldSummary) -> str:
    """Say in words whether a held-back label is done, or what it lacks.

    Parameters
    ----------
    summary : HeldSummary
        The label's summary.

    Returns
    -------
    str
        `done` when nothing is missing, else the missing parts joined by `; `.
    """

    return "; ".join(summary.missing) or STATUS_DONE


def not_run_held_summary(variant: ResolvedVariant) -> HeldSummary:
    """Stand in for the summary of a held-back pick that has none yet.

    Parameters
    ----------
    variant : ResolvedVariant
        The list entry and its name.

    Returns
    -------
    HeldSummary
        Only the pick's name and settings, and `not run` as what it lacks.
    """

    return HeldSummary(
        name=variant.name,
        settings=dict(variant.entry.settings),
        missing=[MISSING_NOT_RUN],
    )


def _series_scores(
    rows: list[dict[str, str]], kind: Kind, head: str
) -> dict[str, _SeriesScore]:
    """Pick the held-back VUS-PR rows of one kind and head from an accuracy table.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The accuracy table's rows.
    kind : Kind
        The archive.
    head : str
        The head.

    Returns
    -------
    dict[str, _SeriesScore]
        Each series with a VUS-PR, by its CSV name without `.csv`.
    """

    scores = {}
    for row in rows:
        if (
            row[SCOPE_COLUMN] != VUS_SCOPE
            or row[METRIC_COLUMN] != VUS_METRIC
            or row[HEAD_COLUMN] != head
            or row[KIND_COLUMN] != kind
        ):
            continue
        value = _numbers([row], VALUE_COLUMN)
        if not value:
            continue
        ibm = _numbers([row], IBM_COLUMN)
        scores[row[FILE_NAME_COLUMN].removesuffix(CSV_EXTENSION)] = _SeriesScore(
            source=row[SOURCE_COLUMN], value=value[0], ibm=ibm[0] if ibm else None
        )
    return scores


def _source_mean(scores: dict[str, _SeriesScore]) -> float | None:
    """Take the mean of each source's mean VUS-PR.

    Parameters
    ----------
    scores : dict[str, _SeriesScore]
        The series' scores.

    Returns
    -------
    float or None
        The mean over sources of the mean over each source's series, None when there are no series.
    """

    by_source: dict[str, list[float]] = {}
    for score in scores.values():
        by_source.setdefault(score.source, []).append(score.value)
    return _mean([fmean(values) for values in by_source.values()])


def _pair(
    scores: dict[str, _SeriesScore], stack: dict[str, _SeriesScore]
) -> Pairing | None:
    """Pair a pick's VUS-PR with the stack's, series by series.

    Parameters
    ----------
    scores : dict[str, _SeriesScore]
        The pick's series' scores.
    stack : dict[str, _SeriesScore]
        The stack's.

    Returns
    -------
    Pairing or None
        The differences `pick − stack` over the series both scored, None when there are none.
    """

    differences = {
        name: scores[name].value - stack[name].value for name in scores if name in stack
    }
    if not differences:
        return None
    count = len(differences)
    worst_series = max(differences, key=lambda name: abs(differences[name]))

    # Mean: https://en.wikipedia.org/wiki/Arithmetic_mean
    # Sample standard deviation, with n − 1 in the denominator: https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
    # Standard error of the mean, sd / sqrt(n): https://en.wikipedia.org/wiki/Standard_error
    deviation = stdev(differences.values()) if count > 1 else None
    return Pairing(
        series=count,
        mean=fmean(differences.values()),
        worst_series=worst_series,
        worst=differences[worst_series],
        std=deviation,
        error=None if deviation is None else deviation / math.sqrt(count),
    )


def _head_score(
    rows: list[dict[str, str]],
    stack_rows: list[dict[str, str]] | None,
    kind: Kind,
    head: str,
) -> HeadScore:
    """Score one head of one kind.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The pick's accuracy table.
    stack_rows : list[dict[str, str]] or None
        The stack's accuracy table, None for the stack itself and until it exists.
    kind : Kind
        The archive.
    head : str
        The head.

    Returns
    -------
    HeadScore
        The means, with IBM's over the same series, and the pairing when the stack's table is given.
    """

    scores = _series_scores(rows, kind, head)
    ibm = [score.ibm for score in scores.values() if score.ibm is not None]
    paired = (
        None
        if stack_rows is None
        else _pair(scores, _series_scores(stack_rows, kind, head))
    )
    return HeadScore(
        series=len(scores),
        mean=_mean([score.value for score in scores.values()]),
        source_mean=_source_mean(scores),
        ibm_mean=_mean(ibm),
        paired=paired,
    )


def _pairing_incomplete(
    rows: list[dict[str, str]],
    stack_rows: list[dict[str, str]] | None,
    kind: Kind,
    head: str,
    score: HeadScore,
) -> bool:
    """Check whether a head's pairing leaves out series that both labels scored.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The pick's accuracy table.
    stack_rows : list[dict[str, str]] or None
        The stack's accuracy table.
    kind : Kind
        The archive.
    head : str
        The head.
    score : HeadScore
        The pick's score of this head, with its pairing.

    Returns
    -------
    bool
        True when the pick scored the head and the pairing covers fewer series than both labels have a VUS-PR for. A head the pick has no series for, or a series the stack has no value for, is not a gap the stack's run could fill.
    """

    if score.series == 0 or stack_rows is None:
        return False
    both = _series_scores(rows, kind, head).keys() & _series_scores(
        stack_rows, kind, head
    )
    paired = 0 if score.paired is None else score.paired.series
    return paired < len(both)


def _spike_check(rows: list[dict[str, str]]) -> SpikeCheck:
    """Count the counted windows whose host and Spike errors were compared, and those that differed.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The instruction table's rows.

    Returns
    -------
    SpikeCheck
        The compared and differing counts, which leave out windows marked as not checked.
    """

    outcomes = [row[IDENTICAL_COLUMN] for row in rows]
    return SpikeCheck(
        units_compared=sum(one in (IDENTICAL_YES, IDENTICAL_NO) for one in outcomes),
        units_differing=outcomes.count(IDENTICAL_NO),
    )


def _kind_mean(rows: list[dict[str, str]], kind: Kind) -> float | None:
    """Take the mean instructions per channel of one kind's counted windows.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The instruction table's rows.
    kind : Kind
        The archive.

    Returns
    -------
    float or None
        The mean `per_channel`, None when the kind has no rows.
    """

    return _mean(
        _numbers([row for row in rows if row[KIND_COLUMN] == kind], PER_CHANNEL_COLUMN)
    )


def _ratio(value: float | None, stack: float | None) -> float | None:
    """Divide one mean by the stack's.

    Parameters
    ----------
    value : float or None
        The pick's mean.
    stack : float or None
        The stack's mean.

    Returns
    -------
    float or None
        The quotient, None when either is missing or the stack's is zero.
    """

    if value is None or not stack:
        return None
    return value / stack


def _window_key(row: dict[str, str]) -> tuple[str, str]:
    """Name the counted window an instruction row is for.

    Parameters
    ----------
    row : dict[str, str]
        A row of the instruction table.

    Returns
    -------
    tuple[str, str]
        The series and the window, as the table spells them.
    """

    return row[SERIES_COLUMN], row[WINDOW_COLUMN]


def _instructions(
    rows: list[dict[str, str]], stack_rows: list[dict[str, str]] | None
) -> HeldInstructions:
    """Take the instruction means of a label and their ratio to the stack's.

    Parameters
    ----------
    rows : list[dict[str, str]]
        The label's instruction table.
    stack_rows : list[dict[str, str]] or None
        The stack's instruction table, None for the stack itself, whose ratios are left out, and until it exists.

    Returns
    -------
    HeldInstructions
        The means per kind, and each divided by the stack's mean of that kind.
    """

    u_mean = _kind_mean(rows, Kind.U)
    m_mean = _kind_mean(rows, Kind.M)
    if stack_rows is None:
        return HeldInstructions(u_mean=u_mean, m_mean=m_mean)

    # Both means are over the windows the two labels counted, so a stack with other windows doesn't skew the ratio
    shared = {_window_key(row) for row in rows} & {
        _window_key(row) for row in stack_rows
    }
    ours = [row for row in rows if _window_key(row) in shared]
    theirs = [row for row in stack_rows if _window_key(row) in shared]
    return HeldInstructions(
        u_mean=u_mean,
        m_mean=m_mean,
        u_ratio=_ratio(_kind_mean(ours, Kind.U), _kind_mean(theirs, Kind.U)),
        m_ratio=_ratio(_kind_mean(ours, Kind.M), _kind_mean(theirs, Kind.M)),
    )


def _stack_table(
    top: Path, stack_label: str | None, label: str, name: str
) -> list[dict[str, str]] | None:
    """Read one of the stack's tables when a pick is to be compared with it.

    Parameters
    ----------
    top : Path
        The git top level.
    stack_label : str or None
        The stack's label, None when there is no stack to compare with.
    label : str
        The pick's label, which is not compared with itself.
    name : str
        The table's file name.

    Returns
    -------
    list[dict[str, str]] or None
        The table's rows, None for the stack itself and when the stack has no such table yet.

    Raises
    ------
    OSError
        When the table exists but can't be read.
    """

    if stack_label is None or stack_label == label:
        return None
    path = tables_folder(top, stack_label) / name
    return _read_table(path) if path.is_file() else None


def held_summary(
    top: Path,
    data: Path,
    label: str,
    variant: ResolvedVariant,
    group_series: list[Series],
    stack_label: str | None,
) -> HeldSummary:
    """Compute a held-back pick's summary from its own tables and the stack's.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The pick's label.
    variant : ResolvedVariant
        The list entry and its name.
    group_series : list[Series]
        Every held-back series.
    stack_label : str or None
        The stack's label, which the pick is paired with unless it is the stack itself.

    Returns
    -------
    HeldSummary
        The pick's summary. A pick that is not the stack lacks the stack's label until the stack's tables exist.

    Raises
    ------
    ValueError
        When a table or the build record can't be used.
    OSError
        When a file can't be read.
    """

    # Step 1: the pick's own tables
    tables = tables_folder(top, label)
    rows = _read_table(tables / ACCURACY_TABLE)
    instruction_rows = _read_table(tables / INSTRUCTIONS_TABLE)

    # Step 2: the stack's, which pair with the pick's. Without them the pairing and the ratio are left out, and the summary says so
    stack_rows = _stack_table(top, stack_label, label, ACCURACY_TABLE)
    stack_instructions = _stack_table(top, stack_label, label, INSTRUCTIONS_TABLE)
    vus_pr = {
        str(kind): {
            head: _head_score(rows, stack_rows, kind, head) for head in VUS_HEADS
        }
        for kind in KINDS
    }

    # Step 3: what the label lacks. A pairing over fewer series than the pick scored, or against a stack that lacks something itself, isn't final, so it names the stack
    missing = missing_stages(top, data, label, group_series, HELD_GROUP)
    if stack_label not in (None, label):
        partial = any(
            _pairing_incomplete(rows, stack_rows, kind, head, vus_pr[str(kind)][head])
            for kind in KINDS
            for head in VUS_HEADS
        )
        if (
            stack_rows is None
            or partial
            or missing_stages(top, data, stack_label, group_series, HELD_GROUP)
        ):
            missing.append(stack_label)

    # Step 4: the summary
    return HeldSummary(
        name=variant.name,
        settings=dict(variant.entry.settings),
        build=_build(tables),
        missing=missing,
        spike_check=_spike_check(instruction_rows),
        instructions=_instructions(instruction_rows, stack_instructions),
        vus_pr=vus_pr,
    )


def write_held_variant_summary(
    top: Path,
    data: Path,
    label: str,
    variant: ResolvedVariant,
    group_series: list[Series],
    stack_label: str | None,
) -> HeldSummary:
    """Write a held-back pick's `summary.json`.

    Parameters
    ----------
    top : Path
        The git top level.
    data : Path
        The folder with the data.
    label : str
        The pick's label.
    variant : ResolvedVariant
        The list entry and its name.
    group_series : list[Series]
        Every held-back series.
    stack_label : str or None
        The stack's label.

    Returns
    -------
    HeldSummary
        The summary written.

    Raises
    ------
    VariantError
        When a file the summary needs is missing or can't be read.
    """

    try:
        summary = held_summary(top, data, label, variant, group_series, stack_label)
        write_held_summary(summary_path(top, label), summary)
    except (OSError, ValueError, KeyError) as error:
        raise VariantError(f"cannot write {label}'s summary: {error}") from error
    return summary
