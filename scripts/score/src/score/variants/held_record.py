"""The types a held-back label's summary is made of, and how they are written to and read from JSON."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Internal
from score.variants.summary_record import (
    JSON_INDENT,
    BuildSource,
    _encode,
    _object,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The `HeldSummary` and `HeadScore` fields that are rebuilt from nested objects
BUILD_FIELD = "build"
SPIKE_CHECK_FIELD = "spike_check"
INSTRUCTIONS_FIELD = "instructions"
VUS_FIELD = "vus_pr"
PAIRED_FIELD = "paired"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class SpikeCheck:
    """How many counted windows were run on the host and under Spike, and whether the two agreed.

    Attributes
    ----------
    units_compared : int
        The counted windows whose host and Spike errors were compared.
    units_differing : int
        Those whose errors were not byte-identical.
    """

    units_compared: int
    units_differing: int


@dataclass(frozen=True)
class HeldInstructions:
    """Mean instruction counts per channel and per counted window, by kind of series.

    Attributes
    ----------
    u_mean : float or None
        The mean over the univariate counted windows, None when there are none.
    m_mean : float or None
        The mean over the multivariate counted windows, None when there are none.
    u_ratio : float or None
        `u_mean` divided by the stack's, None for the stack itself and until the stack's label has instruction counts.
    m_ratio : float or None
        `m_mean` divided by the stack's, None as `u_ratio` is.
    """

    u_mean: float | None = None
    m_mean: float | None = None
    u_ratio: float | None = None
    m_ratio: float | None = None


@dataclass(frozen=True)
class Pairing:
    """The differences from the stack's VUS-PR, one per series both labels scored.

    Attributes
    ----------
    series : int
        How many series the pairing covers.
    mean : float
        The mean difference.
    worst_series : str
        The series whose difference is largest by absolute value, by CSV name without `.csv`.
    worst : float
        That series' difference.
    std : float or None
        The sample standard deviation of the differences, None for a single series.
    error : float or None
        The standard error of the mean, None for a single series.
    """

    series: int
    mean: float
    worst_series: str
    worst: float
    std: float | None = None
    error: float | None = None


@dataclass(frozen=True)
class HeadScore:
    """One head's VUS-PR over the held-back series of one kind.

    Attributes
    ----------
    series : int
        How many series have a VUS-PR.
    mean : float or None
        The plain mean over series, None when none has one.
    source_mean : float or None
        The mean of each source's mean, which weighs every source of the archive equally.
    ibm_mean : float or None
        IBM's Eval mean over the same series, None when none of them has an IBM value.
    paired : Pairing or None
        The differences from the stack, None for the stack and until its label has scores.
    """

    series: int = 0
    mean: float | None = None
    source_mean: float | None = None
    ibm_mean: float | None = None
    paired: Pairing | None = None


@dataclass(frozen=True)
class HeldSummary:
    """What a held-back pick is and how it scored.

    Attributes
    ----------
    name : str
        The variant's build name.
    settings : dict[str, str]
        Each setting and its value, in the list's order.
    build : BuildSource or None
        Which code the build was made from, None when the label holds no build record.
    missing : list[str]
        What the label lacks, empty when it lacks nothing.
    spike_check : SpikeCheck or None
        The host against Spike comparison, None for a pick not run.
    instructions : HeldInstructions or None
        The instruction means and their ratio to the stack's, None for a pick not run.
    vus_pr : dict[str, dict[str, HeadScore]]
        Each kind's, then each head's, VUS-PR, empty for a pick not run.
    """

    name: str
    settings: dict[str, str] = field(default_factory=dict)
    build: BuildSource | None = None
    missing: list[str] = field(default_factory=list)
    spike_check: SpikeCheck | None = None
    instructions: HeldInstructions | None = None
    vus_pr: dict[str, dict[str, HeadScore]] = field(default_factory=dict)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _head_score_from_json(stored: Any, where: str) -> HeadScore:
    """Rebuild one head's score from the object its JSON holds.

    Parameters
    ----------
    stored : Any
        What the JSON holds.
    where : str
        What the object stands for, for the error.

    Returns
    -------
    HeadScore
        The score, with the fields the object leaves out at None.

    Raises
    ------
    TypeError
        When the object doesn't fit `HeadScore`.
    """

    parts = dict(_object(stored, where))
    if PAIRED_FIELD in parts:
        parts[PAIRED_FIELD] = Pairing(**_object(parts[PAIRED_FIELD], PAIRED_FIELD))
    return HeadScore(**parts)


def held_summary_from_json(stored: Any, where: str) -> HeldSummary:
    """Rebuild a held-back summary from the object its JSON holds.

    Parameters
    ----------
    stored : Any
        What the JSON holds.
    where : str
        Where it sits, for the error.

    Returns
    -------
    HeldSummary
        The summary, with the fields the object leaves out at None or empty.

    Raises
    ------
    ValueError
        When the object doesn't fit `HeldSummary`.
    """

    try:
        parts = dict(_object(stored, "a summary"))

        # Step 1: the parts that are one object each
        for key, cls in (
            (BUILD_FIELD, BuildSource),
            (SPIKE_CHECK_FIELD, SpikeCheck),
            (INSTRUCTIONS_FIELD, HeldInstructions),
        ):
            if key in parts:
                parts[key] = cls(**_object(parts[key], key))

        # Step 2: one score per kind and head
        if VUS_FIELD in parts:
            kinds = _object(parts[VUS_FIELD], VUS_FIELD)
            parts[VUS_FIELD] = {
                kind: {
                    head: _head_score_from_json(one, f"{kind} {head}")
                    for head, one in _object(heads, kind).items()
                }
                for kind, heads in kinds.items()
            }
        return HeldSummary(**parts)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from error


def write_held_summary(path: Path, summary: HeldSummary) -> None:
    """Write a held-back label's summary as JSON.

    Parameters
    ----------
    path : Path
        The file, whose folder is made.
    summary : HeldSummary
        The summary.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_encode(summary), indent=JSON_INDENT) + "\n")


def write_held_summaries(path: Path, summaries: list[HeldSummary]) -> None:
    """Write several held-back summaries as one JSON list.

    Parameters
    ----------
    path : Path
        The file, whose folder is made.
    summaries : list[HeldSummary]
        The summaries, in the order they are written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    contents = [_encode(summary) for summary in summaries]
    path.write_text(json.dumps(contents, indent=JSON_INDENT) + "\n")


def read_held_summary(path: Path) -> HeldSummary | None:
    """Read a held-back label's summary.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    HeldSummary or None
        The summary, or None when there is no file.

    Raises
    ------
    ValueError
        When the file isn't JSON or doesn't fit `HeldSummary`.
    OSError
        When the file exists but can't be read.
    """

    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error
    return held_summary_from_json(stored, str(path))
