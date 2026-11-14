"""The mnemonic at each address of the program, from `objdump`."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import re
import subprocess
from pathlib import Path

# Internal
from score.profile.errors import ProfileError


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The binutils that disassemble the rv64 ELF
OBJDUMP = "riscv64-none-elf-objdump"

# `no-aliases` keeps the real mnemonics: otherwise `mv`, `li`, `j` and `ret` replace them and the `c.` prefix of compressed forms is hidden
OBJDUMP_ARGUMENTS = ("-d", "-M", "no-aliases")

# An instruction line: address, the encoding's bytes, then the mnemonic with its operands
INSTRUCTION_LINE = re.compile(r"^\s*([0-9a-f]+):\t[0-9a-f ]+\t(\S+)")


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_mnemonics(elf: Path) -> dict[int, str]:
    """Map each instruction's address to its mnemonic.

    Parameters
    ----------
    elf : Path
        The linked program.

    Returns
    -------
    dict[int, str]
        The mnemonic at every instruction address of the program's code.

    Raises
    ------
    ProfileError
        If `objdump` fails or prints no instruction.
    """

    result = subprocess.run(
        [OBJDUMP, *OBJDUMP_ARGUMENTS, str(elf)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProfileError(f"{OBJDUMP} failed on {elf}: {result.stderr.strip()}")
    mnemonics: dict[int, str] = {}
    for line in result.stdout.splitlines():
        match = INSTRUCTION_LINE.match(line)
        if match is not None:
            mnemonics[int(match.group(1), 16)] = match.group(2)
    if not mnemonics:
        raise ProfileError(f"{OBJDUMP} printed no instruction for {elf}")
    return mnemonics
