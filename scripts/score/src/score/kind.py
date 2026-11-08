"""The two TSB-AD archives a series can come from."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from enum import StrEnum


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Kind(StrEnum):
    """A TSB-AD archive, named by the letter that ends its folder and file names, as in `TSB-AD-U`.

    A `StrEnum`, so each member equals its letter and formats as it: it compares directly with the `kind` column of `benchmark/series.csv` and builds paths unchanged.

    Attributes
    ----------
    U : str
        Univariate: one channel per series.
    M : str
        Multivariate: several channels per series.
    """

    U = "U"
    M = "M"
