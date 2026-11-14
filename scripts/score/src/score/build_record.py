"""A build's `build.json`: the record `make` writes beside the binaries, and the check that a binary still matches it."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# Internal
from score.spike_check import _sha256


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The record's file name, in each build folder
BUILD_FILE = "build.json"

# The fields that say where and when a build was made, not what it is, so they stay out of a label's identity
NON_IDENTITY_FIELDS = (
    "commit",
    "tree",
    "dirty",
    "sha256",
    "source_dir",
    "generated_dir",
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class BuildRecordError(Exception):
    """A build folder has no usable `build.json`, or a binary doesn't match it."""


@dataclass(frozen=True)
class BuildRecord:
    """What `make` wrote to `build.json`.

    Attributes
    ----------
    name : str
        The build's name, from its non-default settings.
    target : str
        `host` or `rv64`.
    settings : dict[str, str]
        Each setting's lower-case name and its value.
    cc : str
        The compiler command.
    cc_version : str
        The compiler's first version line.
    cppflags : str
        The preprocessor flags.
    cflags : str
        The compile flags.
    ldlibs : str
        The libraries.
    file_flags : dict[str, str]
        Extra flags of single source files.
    sources : list[str]
        The source files compiled.
    overrides : dict[str, str]
        The command-line overrides of the compiler or its flags.
    source_dir : str
        The `tspulse` source folder.
    generated_dir : str
        The folder with the generated weight table.
    commit : str or None
        `git rev-parse HEAD` when the build ran, None outside a git repository.
    tree : str or None
        `git rev-parse HEAD^{tree}` when the build ran, None outside a git repository.
    dirty : bool or None
        Whether the tree had uncommitted changes, None when unknown.
    sha256 : dict[str, str]
        The hash of each binary, by file name.
    """

    name: str
    target: str
    settings: dict[str, str]
    cc: str
    cc_version: str
    cppflags: str
    cflags: str
    ldlibs: str
    file_flags: dict[str, str]
    sources: list[str]
    overrides: dict[str, str]
    source_dir: str
    generated_dir: str
    commit: str | None
    tree: str | None
    dirty: bool | None
    sha256: dict[str, str]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def read_build(folder: Path) -> BuildRecord:
    """Read a build folder's `build.json`.

    Parameters
    ----------
    folder : Path
        The build folder.

    Returns
    -------
    BuildRecord
        The record.

    Raises
    ------
    BuildRecordError
        When the file is missing, can't be read, isn't JSON, or lacks or adds fields.
    """

    path = folder / BUILD_FILE
    if not path.is_file():
        raise BuildRecordError(
            f"{path} is missing: rebuild with build-variant, which writes it."
        )
    try:
        stored = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BuildRecordError(
            f"{path} can't be read ({error}): rebuild with build-variant."
        ) from error
    expected = {field.name for field in fields(BuildRecord)}
    if not isinstance(stored, dict) or stored.keys() != expected:
        found = sorted(stored.keys() ^ expected) if isinstance(stored, dict) else []
        raise BuildRecordError(
            f"{path} doesn't have the fields of a build record (differing: {', '.join(found) or 'not an object'}): rebuild with build-variant."
        )
    return BuildRecord(**stored)


def check_binaries(record: BuildRecord, folder: Path, roles: tuple[str, ...]) -> None:
    """Check that each binary a tool uses still has the hash its build recorded.

    Parameters
    ----------
    record : BuildRecord
        The folder's record.
    folder : Path
        The build folder.
    roles : tuple[str, ...]
        The binaries to check, by file name.

    Raises
    ------
    BuildRecordError
        When a binary is missing, or its hash differs from the record's or the record lacks one.
    """

    for role in roles:
        path = folder / role
        if not path.is_file():
            raise BuildRecordError(f"{path} is missing: rebuild with build-variant.")
        recorded = record.sha256.get(role)
        actual = _sha256(path)
        if recorded != actual:
            raise BuildRecordError(
                f"{path} doesn't match its build.json: recorded {recorded}, found {actual}. Rebuild with build-variant."
            )


def identity_part(record: BuildRecord) -> dict[str, Any]:
    """Take the part of a record that says what the build is.

    Parameters
    ----------
    record : BuildRecord
        The record.

    Returns
    -------
    dict[str, Any]
        The record's fields without where and when it was built, so a new commit, another worktree or another machine gives the same value.
    """

    return {
        name: value
        for name, value in asdict(record).items()
        if name not in NON_IDENTITY_FIELDS
    }


def write_build(record: BuildRecord, folder: Path) -> None:
    """Write a record as `build.json` in a results folder.

    Parameters
    ----------
    record : BuildRecord
        The record.
    folder : Path
        The folder, which must exist.
    """

    (folder / BUILD_FILE).write_text(json.dumps(asdict(record), indent=2) + "\n")
