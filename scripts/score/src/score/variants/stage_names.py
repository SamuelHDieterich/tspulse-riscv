"""The names of the stages a variant goes through."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import StrEnum


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class VariantStage(StrEnum):
    """The stages a variant goes through, in order.

    Attributes
    ----------
    BUILD : str
        The host and rv64 builds.
    REFERENCE : str
        The reference windows on the host dump build.
    RUN : str
        `benchmark run`: the accuracy windows on the host and the counted windows under Spike.
    COLLECT : str
        `benchmark collect` against the baseline.
    PROFILE : str
        `benchmark profile` on its default windows.
    SUMMARY : str
        The exactness against the baseline and the label's `summary.json`.
    """

    BUILD = "build"
    REFERENCE = "reference check"
    RUN = "run"
    COLLECT = "collect"
    PROFILE = "profile"
    SUMMARY = "summary"
