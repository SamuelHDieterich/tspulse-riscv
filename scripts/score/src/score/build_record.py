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

# The name of a build that sets nothing, which is the reference every variant is measured against
BASELINE_NAME = "baseline"

# Where the Makefile puts the builds and the generated weights, under the source folder
BUILD_FOLDER = "build"
GENERATED_FOLDER = "generated"

# The Make variables that move the build or generated folders, as `overrides` names them
FOLDER_OVERRIDES = ("BUILD", "GENERATED")

# The fields that say where and when a build was made, not what it is, so they stay out of a label's identity
NON_IDENTITY_FIELDS = (
    "commit",
    "tree",
    "dirty",
    "sha256",
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


def _build_parts(name: str, target: str) -> tuple[str, ...]:
    """Give the path of a build folder under its source folder, as the Makefile's `BUILD` rule lays it out.

    Parameters
    ----------
    name : str
        The build's name.
    target : str
        `host` or `rv64`.

    Returns
    -------
    tuple[str, ...]
        The path's parts. The baseline has no folder of its own between `build` and the target.
    """

    if name == BASELINE_NAME:
        return (BUILD_FOLDER, target)
    return (BUILD_FOLDER, name, target)


def build_path(source: Path, name: str, target: str) -> Path:
    """Find where the Makefile puts a build.

    Parameters
    ----------
    source : Path
        The source folder, which holds the Makefile.
    name : str
        The build's name.
    target : str
        `host` or `rv64`.

    Returns
    -------
    Path
        The build folder.
    """

    return source.joinpath(*_build_parts(name, target))


def source_folder(record: BuildRecord, build_folder: Path) -> Path:
    """Find the `tspulse` source folder a build folder belongs to.

    Parameters
    ----------
    record : BuildRecord
        The build folder's record.
    build_folder : Path
        The build folder.

    Returns
    -------
    Path
        The source folder, which holds the Makefile.

    Raises
    ------
    BuildRecordError
        When the build was made with a `BUILD` override, or its folder isn't where the Makefile puts a build of this name and target.
    """

    # A `make BUILD=…` folder can sit anywhere, even under another tree's `build/<target>`, so the path alone can't say which sources it came from
    moved = [name for name in FOLDER_OVERRIDES if name in record.overrides]
    if moved:
        raise BuildRecordError(
            f"{build_folder} was built with `make {moved[0]}=…`, so its source folder can't be found: build it without the override."
        )

    parts = _build_parts(record.name, record.target)
    resolved = build_folder.resolve()
    if resolved.parts[-len(parts) :] != parts:
        raise BuildRecordError(
            f"{build_folder} isn't a folder of the form {'/'.join(parts)}, so the source folder can't be found: was it moved or copied there?"
        )
    return resolved.parents[len(parts) - 1]


def generated_folder(source: Path) -> Path:
    """Give the folder with the generated weights.

    Parameters
    ----------
    source : Path
        The source folder.

    Returns
    -------
    Path
        The generated folder, as the Makefile's `GENERATED`.
    """

    return source / BUILD_FOLDER / GENERATED_FOLDER


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
