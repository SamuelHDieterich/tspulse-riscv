"""The tracked variant list: read `benchmark/variants.toml` into one entry per variant and refuse a list that can't be built."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The only top-level key of the file: an array of tables, one per variant
VARIANT_KEY = "variant"

# The keys a variant table has: its settings, an inline table of setting name to value, and optionally whether it is one of the picks run on the held-back series
SETTINGS_KEY = "settings"
HELD_KEY = "held"
VARIANT_KEYS = {SETTINGS_KEY, HELD_KEY}

# The folder with the Makefile, under the git top level
MAKE_FOLDER = "tspulse"

# The make target that prints the settings table, and the table's header row, which holds no setting
SETTINGS_TARGET = "settings"
SETTINGS_HEADER = "SETTING"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class VariantError(Exception):
    """The variant list, or a variant's build, can't be used; the message says what to change."""


@dataclass(frozen=True)
class VariantEntry:
    """One `[[variant]]` of the list.

    Attributes
    ----------
    settings : dict[str, str]
        Each setting's name and its value, as passed to `make`.
    held : bool
        Whether the variant is one of the picks run on the held-back series.
    """

    settings: dict[str, str]
    held: bool = False


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def make_settings(top: Path) -> list[str]:
    """List the setting names the Makefile accepts.

    Parameters
    ----------
    top : Path
        The git top level.

    Returns
    -------
    list[str]
        The names in the settings table, in its order.

    Raises
    ------
    VariantError
        When `make` fails.
    """

    process = subprocess.run(
        ["make", "-s", "-C", MAKE_FOLDER, SETTINGS_TARGET],
        cwd=top,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise VariantError(f"`make {SETTINGS_TARGET}` failed: {process.stderr.strip()}")
    rows = [line.split() for line in process.stdout.splitlines() if line.strip()]
    return [row[0] for row in rows if row[0] != SETTINGS_HEADER]


def read_listing(path: Path, top: Path) -> list[VariantEntry]:
    """Read the variant list and check it.

    Parameters
    ----------
    path : Path
        The TOML file.
    top : Path
        The git top level, where `make` runs to learn the setting names.

    Returns
    -------
    list[VariantEntry]
        One entry per `[[variant]]`, in the file's order.

    Raises
    ------
    VariantError
        When the file can't be read, has a key other than `variant`, has a variant with an empty `settings`, a key other than `settings` and `held`, or a `held` that isn't a boolean, has a value that isn't a string, names a setting `make` doesn't have, or repeats a variant's settings.
    """

    # Step 1: read the file and check its keys
    try:
        document = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise VariantError(f"{path} can't be read: {error}") from error
    unknown_keys = sorted(document.keys() - {VARIANT_KEY})
    if unknown_keys:
        raise VariantError(
            f"{path} has unknown top-level keys {unknown_keys}: only `[[{VARIANT_KEY}]]` tables are allowed."
        )
    tables = document.get(VARIANT_KEY, [])
    if not isinstance(tables, list) or not all(isinstance(t, dict) for t in tables):
        raise VariantError(
            f"{path}: `{VARIANT_KEY}` must be `[[{VARIANT_KEY}]]` tables, one per variant."
        )
    if not tables:
        raise VariantError(f"{path} lists no variants: add a `[[{VARIANT_KEY}]]`.")

    # Step 2: check each variant's settings
    known = make_settings(top)
    entries: list[VariantEntry] = []
    for number, table in enumerate(tables, start=1):
        where = f"{path} variant {number}"
        extra_keys = sorted(table.keys() - VARIANT_KEYS)
        if extra_keys:
            raise VariantError(
                f"{where} has unknown keys {extra_keys}: only `{SETTINGS_KEY}` and `{HELD_KEY}` are allowed."
            )
        held = table.get(HELD_KEY, False)
        if not isinstance(held, bool):
            raise VariantError(
                f"{where} gives {HELD_KEY} the value {held!r}: it is true or false."
            )
        settings = table.get(SETTINGS_KEY)
        if not isinstance(settings, dict) or not settings:
            raise VariantError(
                f'{where} needs a non-empty `{SETTINGS_KEY}` table, such as {{ {known[0]} = "..." }}.'
            )
        for name, value in settings.items():
            if name not in known:
                raise VariantError(
                    f"{where} sets {name}, which make doesn't have: the settings are {', '.join(known)}."
                )
            if not isinstance(value, str):
                raise VariantError(
                    f'{where} gives {name} the value {value!r}: values are strings, such as "on".'
                )
        entry = VariantEntry(settings=dict(settings), held=held)
        if any(other.settings == entry.settings for other in entries):
            raise VariantError(f"{where} repeats the settings of an earlier variant.")
        entries.append(entry)
    return entries


def find_stack(entries: list[VariantEntry]) -> VariantEntry:
    """Find the stack of the held-back picks: the one entry whose settings every other held entry contains.

    Parameters
    ----------
    entries : list[VariantEntry]
        Every entry of the list.

    Returns
    -------
    VariantEntry
        The held entry that every other held entry builds on, which is the exact reference the picks' differences are paired against.

    Raises
    ------
    VariantError
        When no entry is held, or the held ones don't have exactly one entry that all the others contain.
    """

    held = [entry for entry in entries if entry.held]
    stacks = [
        entry
        for entry in held
        if all(
            entry.settings.items() <= other.settings.items()
            for other in held
            if other is not entry
        )
    ]
    if len(stacks) != 1:
        raise VariantError(
            f"The held variants ({len(held)} marked `{HELD_KEY} = true`) need exactly one whose settings every other contains, the stack; found {len(stacks)}."
        )
    return stacks[0]
