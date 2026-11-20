"""The types a variant's summary is made of, and how they are written to and read from JSON."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# Internal
from score.variants.state import SUMMARY_FILE, tables_folder


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The field metadata that names a field's key in the JSON, for a key that isn't a Python name
JSON_KEY = "json_key"

# The JSON's indentation
JSON_INDENT = 2

# The `Summary` fields that need more than `Part(**object)` to be rebuilt
REFERENCE_FIELD = "reference"
VERDICT_FIELD = "verdict"
VUS_FIELD = "vus_pr"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Verdict(StrEnum):
    """The verdicts a summary gives.

    Attributes
    ----------
    IDENTICAL : str
        Every compared window matches the baseline's.
    WITHIN : str
        Not identical, but every reference window passed.
    BEYOND : str
        Some reference window failed.
    REFERENCE : str
        The baseline itself, which the others are measured against.
    """

    IDENTICAL = "identical"
    WITHIN = "within-tolerance"
    BEYOND = "beyond-tolerance"
    REFERENCE = "reference"


@dataclass(frozen=True)
class BuildSource:
    """Which code a build was made from, as its `build.json` says.

    Attributes
    ----------
    commit : str or None
        The commit the build was made from, None when the record has none.
    tree : str or None
        That commit's tree, which still names the code after a history rewrite, None when the record has none.
    dirty : bool or None
        Whether the working tree had changes at build time, None when the record doesn't say.
    """

    commit: str | None = None
    tree: str | None = None
    dirty: bool | None = None


@dataclass(frozen=True)
class ExactnessCounts:
    """How many windows the variant's errors were compared on.

    Attributes
    ----------
    windows_compared : int
        The windows whose error row both the variant and the baseline hold.
    windows_differing : int
        Those whose row differs as text.
    """

    windows_compared: int
    windows_differing: int


@dataclass(frozen=True)
class ReferenceCounts:
    """How many of the reference windows passed.

    Attributes
    ----------
    passed : int
        The windows that passed, `pass` in the JSON.
    failed : int
        The windows that failed, `fail` in the JSON.
    """

    passed: int = field(metadata={JSON_KEY: "pass"})
    failed: int = field(metadata={JSON_KEY: "fail"})


@dataclass(frozen=True)
class Instructions:
    """Mean instruction counts per channel.

    Attributes
    ----------
    u_mean : float or None
        The mean over the univariate counted windows, None when there are none.
    m_mean : float or None
        The mean over the multivariate counted windows, None when there are none.
    ratio : float or None
        The mean ratio of the variant's univariate counts to the baseline's, None for the baseline.
    """

    u_mean: float | None = None
    m_mean: float | None = None
    ratio: float | None = None


@dataclass(frozen=True)
class Cycles:
    """The cycle range of the mean window, over all classes.

    Attributes
    ----------
    low : int
        The low end.
    high : int
        The high end.
    """

    low: int
    high: int


@dataclass(frozen=True)
class MemoryBytes:
    """What the program takes of memory.

    Attributes
    ----------
    weights : int
        The read-only weights.
    text : int
        The `.text` section.
    static : int
        The sum of the static buffers.
    stack : int
        The stack's worst path.
    heap_per_channel : int
        The heap formulas' sum at one channel.
    """

    weights: int
    text: int
    static: int
    stack: int
    heap_per_channel: int


@dataclass(frozen=True)
class HeadVus:
    """One head's VUS-PR over the series.

    Attributes
    ----------
    value : float or None
        The mean VUS-PR, None when no series has one.
    difference : float or None
        The mean difference from the baseline, None for the baseline and when no series has one.
    worst : float or None
        The most negative difference, the series that got worst, None as `difference` is.
    """

    value: float | None = None
    difference: float | None = None
    worst: float | None = None


@dataclass(frozen=True)
class Summary:
    """What a variant is, whether it is exact, and what it costs and scores.

    Attributes
    ----------
    name : str
        The variant's build name.
    settings : dict[str, str]
        Each setting and its value, in the list's order. Empty for the baseline.
    build : BuildSource or None
        Which code the build was made from, None for the baseline.
    missing : list[str]
        What the label lacks, empty when it lacks nothing.
    verdict : Verdict or None
        The judgement against the baseline, None for a variant not run.
    exactness : ExactnessCounts or None
        The window comparison with the baseline, None for the baseline and a variant not run.
    reference : ReferenceCounts or None
        The reference windows' outcome, None for the baseline and a variant not run.
    instructions : Instructions or None
        The instruction means, None for a variant not run.
    cycles : Cycles or None
        The cycle range, None for a variant not run.
    memory_bytes : MemoryBytes or None
        The sizes, None for a variant not run.
    traffic_bytes : int or None
        The mean window's total traffic, None for a variant not run.
    profile_from : str or None
        The name of the stack whose profile the cycles, traffic and memory come from, None for a variant that was profiled itself.
    vus_pr : dict[str, HeadVus]
        Each head's VUS-PR, empty for a variant not run.
    """

    name: str
    settings: dict[str, str] = field(default_factory=dict)
    build: BuildSource | None = None
    missing: list[str] = field(default_factory=list)
    verdict: Verdict | None = None
    exactness: ExactnessCounts | None = None
    reference: ReferenceCounts | None = None
    instructions: Instructions | None = None
    cycles: Cycles | None = None
    memory_bytes: MemoryBytes | None = None
    traffic_bytes: int | None = None
    profile_from: str | None = None
    vus_pr: dict[str, HeadVus] = field(default_factory=dict)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _is_empty(value: Any) -> bool:
    """Say whether a value is left out of the JSON.

    Parameters
    ----------
    value : Any
        An encoded value.

    Returns
    -------
    bool
        True for None and for an empty object or list. A zero, a false and an empty string are values.
    """

    return value is None or (isinstance(value, (dict, list)) and not value)


def _encode(value: Any) -> Any:
    """Turn a summary, or a part of one, into what `json` writes.

    Parameters
    ----------
    value : Any
        A dataclass, a dict, a list or a scalar.

    Returns
    -------
    Any
        The same shape, with each dataclass an object and every None or empty field or entry left out.
    """

    if is_dataclass(value) and not isinstance(value, type):
        pairs = [
            (one.metadata.get(JSON_KEY, one.name), getattr(value, one.name))
            for one in fields(value)
        ]
    elif isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, list):
        return [_encode(item) for item in value]
    else:
        return value
    encoded = ((key, _encode(item)) for key, item in pairs)
    return {key: item for key, item in encoded if not _is_empty(item)}


def summary_to_json(summary: Summary) -> dict[str, Any]:
    """Turn a summary into the object its JSON holds.

    Parameters
    ----------
    summary : Summary
        The summary.

    Returns
    -------
    dict[str, Any]
        Its fields, with every None or empty one left out.
    """

    return _encode(summary)


def _object(stored: Any, what: str) -> dict[str, Any]:
    """Check that a JSON value is an object.

    Parameters
    ----------
    stored : Any
        What the JSON holds.
    what : str
        What the object stands for, for the error.

    Returns
    -------
    dict[str, Any]
        The object.

    Raises
    ------
    TypeError
        When the value isn't an object.
    """

    if not isinstance(stored, dict):
        raise TypeError(f"{what} needs an object")
    return stored


def summary_from_json(stored: Any, where: str) -> Summary:
    """Rebuild a summary from the object its JSON holds.

    Parameters
    ----------
    stored : Any
        What the JSON holds.
    where : str
        Where it sits, for the error.

    Returns
    -------
    Summary
        The summary, with the fields the object leaves out at None or empty.

    Raises
    ------
    ValueError
        When the object doesn't fit `Summary`.
    """

    try:
        parts = dict(_object(stored, "a summary"))

        # Step 1: the parts that are one object each
        for key, cls in (
            ("build", BuildSource),
            ("exactness", ExactnessCounts),
            ("instructions", Instructions),
            ("cycles", Cycles),
            ("memory_bytes", MemoryBytes),
        ):
            if key in parts:
                parts[key] = cls(**_object(parts[key], key))

        # Step 2: the reference counts, whose JSON keys differ from their field names
        if REFERENCE_FIELD in parts:
            by_key = {
                one.metadata[JSON_KEY]: one.name for one in fields(ReferenceCounts)
            }
            counts = _object(parts[REFERENCE_FIELD], REFERENCE_FIELD)
            parts[REFERENCE_FIELD] = ReferenceCounts(
                **{by_key.get(key, key): count for key, count in counts.items()}
            )

        # Step 3: the verdict, and one VUS-PR record per head
        if VERDICT_FIELD in parts:
            parts[VERDICT_FIELD] = Verdict(parts[VERDICT_FIELD])
        if VUS_FIELD in parts:
            heads = _object(parts[VUS_FIELD], VUS_FIELD)
            parts[VUS_FIELD] = {
                head: HeadVus(**_object(one, head)) for head, one in heads.items()
            }
        return Summary(**parts)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from error


def write_summary(path: Path, summary: Summary) -> None:
    """Write a label's summary as JSON.

    Parameters
    ----------
    path : Path
        The file, whose folder is made.
    summary : Summary
        The summary.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_to_json(summary), indent=JSON_INDENT) + "\n")


def write_summaries(path: Path, summaries: list[Summary]) -> None:
    """Write several summaries as one JSON list.

    Parameters
    ----------
    path : Path
        The file, whose folder is made.
    summaries : list[Summary]
        The summaries, in the order they are written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    contents = [summary_to_json(summary) for summary in summaries]
    path.write_text(json.dumps(contents, indent=JSON_INDENT) + "\n")


def read_summary(path: Path) -> Summary | None:
    """Read a label's summary.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    Summary or None
        The summary, or None when there is no file.

    Raises
    ------
    ValueError
        When the file isn't JSON or doesn't fit `Summary`.
    OSError
        When the file exists but can't be read.
    """

    if not path.is_file():
        return None
    try:
        stored = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: {error}") from error
    return summary_from_json(stored, str(path))


def summary_path(top: Path, label: str) -> Path:
    """Where a label's summary sits.

    Parameters
    ----------
    top : Path
        The git top level.
    label : str
        The label.

    Returns
    -------
    Path
        `summary.json` in the label's tracked results folder.
    """

    return tables_folder(top, label) / SUMMARY_FILE
