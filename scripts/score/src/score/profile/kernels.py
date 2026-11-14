"""Instructions per function for one window."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from score.profile.histogram import Histogram, largest_first
from score.profile.symbols import FunctionTable, split_by_function


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def kernel_counts(difference: Histogram, table: FunctionTable) -> dict[str, int]:
    """Count one window's instructions per function.

    Parameters
    ----------
    difference : Histogram
        The window's instructions per address: the run `--windows M-1:M+1` minus the run `--windows M-1:M`.
    table : FunctionTable
        The functions of the ELF both runs executed.

    Returns
    -------
    dict[str, int]
        Instructions per function, plus pk's slice, largest first.

    Raises
    ------
    ProfileError
        If an address is in no function, or a program function's difference is negative (pk's slice is exempt).
    """

    return largest_first(split_by_function(difference, table))
