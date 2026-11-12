"""The units a benchmark run is split into, and how a call chooses them.

An accuracy unit runs a chunk of a series' windows and keeps their errors.
An instruction unit runs one window under Spike and keeps how many instructions its model run took.
A unit's cost is its channels times its windows, which both stages' run time follows about linearly.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

# External
import typer

# Internal
from score.benchmark.scope import Scope, Series, scope_range
from score.kind import Kind


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Channel-windows per accuracy unit on the host, about 17 min on one thread, so resuming loses little and the jobs stay balanced
HOST_CHUNK_CHANNEL_WINDOWS = 5_000

# Channel-windows per accuracy unit under Spike, about an hour at 35 s per channel-window
SPIKE_CHUNK_CHANNEL_WINDOWS = 100

# The folders each stage's units sit in, under the label's folder and the work folder
ACCURACY_FOLDER = "accuracy"
INSTRUCTIONS_FOLDER = "instructions"

# The folder under the label's holding `collect`'s scored arrays, which depend on the label's binaries
SCORES_FOLDER = "scores"

# How many `_`-separated fields of a series' stem name it in a log line, such as `090_SVDB`
SHORT_NAME_FIELDS = 2

# The separator in `--shard K/N`
SHARD_SEPARATOR = "/"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class CountWindows(StrEnum):
    """Which windows of each series the instruction stage counts.

    Attributes
    ----------
    ENDS_AND_MIDDLE : str
        A univariate series' first, middle and last screen window, and a multivariate series' middle window.
    MIDDLE : str
        One window per series: the screen span's middle for a univariate series, the middle window for a multivariate one.
    """

    ENDS_AND_MIDDLE = "ends-and-middle"
    MIDDLE = "middle"


@dataclass(frozen=True)
class AccuracyUnit:
    """A chunk of a series' windows, whose errors the accuracy stage keeps.

    Attributes
    ----------
    series : str
        The series' CSV name without `.csv`.
    kind : Kind
        The archive it comes from.
    channels : int
        How many channels it has.
    start : int
        The first window.
    stop : int
        One past the last window.
    """

    series: str
    kind: Kind
    channels: int
    start: int
    stop: int

    @property
    def cost(self) -> int:
        """How many channel-windows the unit runs.

        Returns
        -------
        int
            Channels times windows.
        """

        return self.channels * (self.stop - self.start)

    @property
    def folder(self) -> Path:
        """The unit's folder, under the label's folder and the work folder.

        Returns
        -------
        Path
            `accuracy/<stem>/w<start>-<stop>`, with the half-open range of `--windows`.
        """

        return Path(ACCURACY_FOLDER) / self.series / f"w{self.start}-{self.stop}"

    @property
    def name(self) -> str:
        """The unit as a log line names it.

        Returns
        -------
        str
            Such as `090_SVDB w5000-7500`.
        """

        return f"{_short_name(self.series)} w{self.start}-{self.stop}"


@dataclass(frozen=True)
class InstructionUnit:
    """One window run under Spike, whose instructions the instruction stage counts.

    Attributes
    ----------
    series : str
        The series' CSV name without `.csv`.
    kind : Kind
        The archive it comes from.
    channels : int
        How many channels it has.
    window : int
        The window.
    """

    series: str
    kind: Kind
    channels: int
    window: int

    @property
    def cost(self) -> int:
        """How many channel-windows the unit runs.

        Returns
        -------
        int
            The channels, since it runs one window.
        """

        return self.channels

    @property
    def folder(self) -> Path:
        """The unit's folder, under the label's folder and the work folder.

        Returns
        -------
        Path
            `instructions/<stem>/w<window>`.
        """

        return Path(INSTRUCTIONS_FOLDER) / self.series / f"w{self.window}"

    @property
    def name(self) -> str:
        """The unit as a log line names it.

        Returns
        -------
        str
            Such as `090_SVDB w24744`.
        """

        return f"{_short_name(self.series)} w{self.window}"


@dataclass(frozen=True)
class Shard:
    """One of N parts a unit list is split into, so several computers can share a run.

    Attributes
    ----------
    index : int
        Which part, from 0.
    count : int
        How many parts.
    """

    index: int
    count: int


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _short_name(stem: str) -> str:
    """Shorten a series' stem to its number and archive.

    Parameters
    ----------
    stem : str
        Such as `090_SVDB_id_7_Medical_tr_12157_1st_12257`.

    Returns
    -------
    str
        Such as `090_SVDB`.
    """

    return "_".join(stem.split("_")[:SHORT_NAME_FIELDS])


def parse_shard(text: str | None) -> Shard | None:
    """Read `--shard K/N`.

    Parameters
    ----------
    text : str or None
        The option's value, or None when it wasn't given.

    Returns
    -------
    Shard or None
        The shard, or None for the whole list.

    Raises
    ------
    typer.BadParameter
        When the value isn't two integers `K/N` with `0 <= K < N`.
    """

    if text is None:
        return None
    index, separator, count = text.partition(SHARD_SEPARATOR)
    try:
        shard = Shard(int(index), int(count))
    except ValueError:
        shard = None
    if not separator or shard is None or not 0 <= shard.index < shard.count:
        raise typer.BadParameter(
            f"{text!r} is not K/N with 0 <= K < N, such as 0/2.", param_hint="--shard"
        )
    return shard


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge half-open ranges into sorted, non-overlapping ones, so a window in two scopes runs once.

    Parameters
    ----------
    ranges : list[tuple[int, int]]
        Ranges `[start, stop)`, possibly overlapping or empty.

    Returns
    -------
    list[tuple[int, int]]
        Their union, as sorted ranges that neither overlap nor touch.
    """

    merged: list[tuple[int, int]] = []
    for start, stop in sorted(ranges):
        if start >= stop:
            continue
        # Touching ranges merge too, so a chunk is never cut short at a scope's edge
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def accuracy_units(
    series: list[Series],
    scopes: Sequence[Scope],
    window_cap: int | None,
    chunk_channel_windows: int,
) -> list[AccuracyUnit]:
    """Cut the union of each series' scope ranges into chunks.

    Parameters
    ----------
    series : list[Series]
        The series to run.
    scopes : Sequence[Scope]
        The scopes chosen.
    window_cap : int or None
        Keep only this many windows of each scope range, or all when None.
    chunk_channel_windows : int
        The most channel-windows a chunk holds.

    Returns
    -------
    list[AccuracyUnit]
        The chunks, in series order then window order.
    """

    units = []
    for one in series:
        ranges = [scope_range(one, scope, window_cap) for scope in scopes]
        # A series with many channels still gets one window per chunk
        per_chunk = max(1, chunk_channel_windows // one.channels)
        for start, stop in _merge([found for found in ranges if found is not None]):
            for chunk_start in range(start, stop, per_chunk):
                units.append(
                    AccuracyUnit(
                        series=one.stem,
                        kind=one.kind,
                        channels=one.channels,
                        start=chunk_start,
                        stop=min(chunk_start + per_chunk, stop),
                    )
                )
    return units


def _counted_windows(series: Series, count_windows: CountWindows) -> list[int]:
    """The windows of a series the instruction stage counts.

    Parameters
    ----------
    series : Series
        The series.
    count_windows : CountWindows
        Which windows to count.

    Returns
    -------
    list[int]
        The windows, in order and without repeats.
    """

    # A multivariate series has no screen span, so its middle window stands for it
    if series.screen is None:
        return [series.windows // 2]

    # The screen's windows are `[start, stop)`, so its last window is `stop − 1` and its middle `(start + stop − 1) // 2`
    start, stop = series.screen
    last = stop - 1
    middle = (start + last) // 2
    if count_windows == CountWindows.MIDDLE:
        return [middle]
    return sorted({start, middle, last})


def instruction_units(
    series: list[Series], count_windows: CountWindows
) -> list[InstructionUnit]:
    """List the windows the instruction stage counts.

    Parameters
    ----------
    series : list[Series]
        The series to run.
    count_windows : CountWindows
        Which windows of each series to count.

    Returns
    -------
    list[InstructionUnit]
        One unit per counted window, in series order then window order.
    """

    return [
        InstructionUnit(
            series=one.stem, kind=one.kind, channels=one.channels, window=window
        )
        for one in series
        for window in _counted_windows(one, count_windows)
    ]


def order_and_shard[UnitT: (AccuracyUnit, InstructionUnit)](
    units: list[UnitT], shard: Shard | None
) -> list[UnitT]:
    """Order units largest first and keep one shard of them.

    Parameters
    ----------
    units : list[AccuracyUnit] or list[InstructionUnit]
        The units of one stage.
    shard : Shard or None
        The part to keep, or None for all.

    Returns
    -------
    list[AccuracyUnit] or list[InstructionUnit]
        Largest first, so the tail of a run is short units and its ETA settles early. The sort is stable, so ties keep their listed order and every computer cuts the same shards.
    """

    ordered = sorted(units, key=lambda unit: unit.cost, reverse=True)
    if shard is None:
        return ordered
    return ordered[shard.index :: shard.count]
