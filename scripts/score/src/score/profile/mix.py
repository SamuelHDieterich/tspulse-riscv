"""Instructions per mnemonic for one window."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Internal
from score.profile.errors import ProfileError
from score.profile.histogram import Histogram, largest_first
from score.spike_check import PK_BASE


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def mix_counts(difference: Histogram, mnemonics: dict[int, str]) -> dict[str, int]:
    """Count one window's instructions per mnemonic.

    pk's instructions are left out: the ELF holds the program only, so pk's addresses have no mnemonic here.
    Compressed forms keep their own mnemonic (`c.addi` and `addi` are separate), so the table says exactly what ran, and the cycle count maps each compressed form to its base instruction.

    Parameters
    ----------
    difference : Histogram
        The window's instructions per address: the run `--windows M-1:M+1` minus the run `--windows M-1:M`.
    mnemonics : dict[int, str]
        The mnemonic at each address of the program, from `read_mnemonics`.

    Returns
    -------
    dict[str, int]
        Instructions per mnemonic of the program, largest first.

    Raises
    ------
    ProfileError
        If an executed program address holds no instruction, or a mnemonic's total is negative.
    """

    totals: dict[str, int] = {}
    for address, count in difference.items():
        if address >= PK_BASE:
            continue
        if address not in mnemonics:
            raise ProfileError(f"address {address:#x} ran but holds no instruction")
        mnemonic = mnemonics[address]
        totals[mnemonic] = totals.get(mnemonic, 0) + count
    # A negative row would be meaningless in the mix table, so the window fails loudly instead
    negative = {name: count for name, count in totals.items() if count < 0}
    if negative:
        raise ProfileError(f"negative instruction counts per mnemonic: {negative}")
    return largest_first(totals)
