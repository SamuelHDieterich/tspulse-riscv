"""The units a `time-host` call runs: which series, which build, which setup, and in what order.

A unit is one measurement: one repeat of one series, timed with one build in one setup.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

# Internal
from score.layout import DATASETS_FOLDER


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The two series, in the order units run them
SeriesName = Literal["yahoo560", "tao120"]
SERIES_NAMES: tuple[SeriesName, ...] = get_args(SeriesName)

# What runs the model is a build's name for a C unit, or this for IBM's PyTorch pipeline, so no C build can take the name
TORCH = "torch"

# How the C runs: one process pinned to one core, or P processes over split window ranges, unpinned
CSetup = Literal["one-core", "whole-pc"]
C_SETUPS: tuple[CSetup, ...] = get_args(CSetup)
ONE_CORE: CSetup = "one-core"
C_WHOLE_PC: CSetup = "whole-pc"

# How PyTorch runs: windows per model call (b) and threads (t), where `whole-pc` is P threads, unpinned
TorchSetup = Literal["b1-t1", "b128-t1", "b128-whole-pc"]
TORCH_SETUPS: tuple[TorchSetup, ...] = get_args(TorchSetup)
B1_T1: TorchSetup = "b1-t1"
B128_T1: TorchSetup = "b128-t1"
B128_WHOLE_PC: TorchSetup = "b128-whole-pc"

Setup = CSetup | TorchSetup

# The extension of a unit's result file
RESULT_SUFFIX = ".json"

# Each series' CSV, relative to the data folder
SERIES_CSV: dict[SeriesName, Path] = {
    "yahoo560": DATASETS_FOLDER
    / "TSB-AD-U"
    / "560_YAHOO_id_10_Synthetic_tr_500_1st_893.csv",
    "tao120": DATASETS_FOLDER
    / "TSB-AD-M"
    / "120_TAO_id_5_Environment_tr_500_1st_3.csv",
}

# The most windows run of each series, counted from the first, or None for every window.
# TAO is capped because what it shows is PyTorch's speed per channel, and 128 windows of its 3 channels are enough for that
SERIES_CAP: dict[SeriesName, int | None] = {"yahoo560": None, "tao120": 128}

# Windows per model call in each PyTorch setup. 128 is the pipeline's default batch size
TORCH_BATCH_SIZE: dict[TorchSetup, int] = {B1_T1: 1, B128_T1: 128, B128_WHOLE_PC: 128}

# The PyTorch setups that run P threads, unpinned. The others run one thread pinned to the chosen CPU
WHOLE_PC_TORCH_SETUPS: frozenset[TorchSetup] = frozenset({B128_WHOLE_PC})


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Unit:
    """One measurement: one repeat of one series, with one build in one setup.

    Attributes
    ----------
    repeat : int
        Which repeat, from 1.
    series : SeriesName
        The series.
    build : str
        The C build's name, or `torch`.
    setup : Setup
        How it runs.
    """

    repeat: int
    series: SeriesName
    build: str
    setup: Setup

    @property
    def name(self) -> str:
        """The unit as log lines and file names give it.

        Returns
        -------
        str
            Such as `r1-yahoo560-linear-blocked+unroll-on-one-core`.
        """

        return f"r{self.repeat}-{self.series}-{self.build}-{self.setup}"

    @property
    def file_name(self) -> str:
        """The unit's result file.

        Returns
        -------
        str
            Such as `r1-yahoo560-baseline-one-core.json`.
        """

        return self.name + RESULT_SUFFIX

    @property
    def measurement(self) -> tuple[SeriesName, str, Setup]:
        """What the unit measures, the same for every repeat.

        Returns
        -------
        tuple[SeriesName, str, Setup]
            The series, build and setup.
        """

        return self.series, self.build, self.setup


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def all_units(repeats: int, builds: list[str]) -> list[Unit]:
    """List every unit of a call, in the order they run.

    Repeats come first, so every build and setup sees the same share of a warming or throttling machine.

    Parameters
    ----------
    repeats : int
        How many times each measurement runs.
    builds : list[str]
        The C builds' names, in the order they run.

    Returns
    -------
    list[Unit]
        Ordered by repeat, then series, then each C build in the order given, then PyTorch, then setup.
    """

    return [
        Unit(repeat, series, build, setup)
        for repeat in range(1, repeats + 1)
        for series in SERIES_NAMES
        for build in (*builds, TORCH)
        for setup in (TORCH_SETUPS if build == TORCH else C_SETUPS)
    ]
