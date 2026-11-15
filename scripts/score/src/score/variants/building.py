"""Give each variant of the list its build name, and build it for the host and for rv64."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Internal
from score.benchmark.run import HOST_TARGET, RV64_TARGET, read_checked_build
from score.build_record import BASELINE_NAME, BuildRecord, build_path
from score.spike_check import CLEAN_BINARY
from score.variants.listing import MAKE_FOLDER, VariantEntry, VariantError


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Makefile snippet and target that print a build's name without compiling anything
SHOW_NAME_TARGET = "show-name"
SHOW_NAME_RULE = f"--eval={SHOW_NAME_TARGET}: ; @echo $(VARIANT_NAME)"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class ResolvedVariant:
    """A list entry with the name its build gets.

    Attributes
    ----------
    entry : VariantEntry
        The list entry.
    name : str
        The build's name, such as `linear-blocked+unroll-on`.
    """

    entry: VariantEntry
    name: str


@dataclass(frozen=True)
class BuiltVariant:
    """A variant's two builds.

    Attributes
    ----------
    name : str
        The build's name.
    host_folder : Path
        The host build folder.
    rv64_folder : Path
        The rv64 build folder.
    host : BuildRecord
        The host build's record.
    rv64 : BuildRecord
        The rv64 build's record.
    """

    name: str
    host_folder: Path
    rv64_folder: Path
    host: BuildRecord
    rv64: BuildRecord


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _make_command(entry: VariantEntry, target: str, *goals: str) -> list[str]:
    """Build a `make` command line for a variant's settings.

    Parameters
    ----------
    entry : VariantEntry
        The variant.
    target : str
        `host` or `rv64`.
    *goals : str
        The make goals and extra arguments, after the settings.

    Returns
    -------
    list[str]
        The command, run from the git top level.
    """

    settings = [f"{name}={value}" for name, value in entry.settings.items()]
    return ["make", "-C", MAKE_FOLDER, f"TARGET={target}", *settings, *goals]


def resolve_names(entries: list[VariantEntry], top: Path) -> list[ResolvedVariant]:
    """Ask `make` for each entry's build name, without compiling.

    Parameters
    ----------
    entries : list[VariantEntry]
        The list's entries.
    top : Path
        The git top level.

    Returns
    -------
    list[ResolvedVariant]
        Each entry with its name, in the list's order.

    Raises
    ------
    VariantError
        When `make` rejects a setting's value, with its own message, when the settings add up to the baseline, or when two entries get the same name.
    """

    resolved: list[ResolvedVariant] = []
    for entry in entries:
        # The target doesn't change the name, so rv64 stands for both
        command = _make_command(
            entry,
            RV64_TARGET,
            "-s",
            "--no-print-directory",
            SHOW_NAME_RULE,
            SHOW_NAME_TARGET,
        )
        process = subprocess.run(command, cwd=top, capture_output=True, text=True)
        if process.returncode:
            raise VariantError(
                f"make refused the settings {entry.settings}: {process.stderr.strip()}"
            )
        name = process.stdout.strip()
        if name == BASELINE_NAME:
            raise VariantError(
                f"The settings {entry.settings} are all defaults, which is the {BASELINE_NAME}: change a value or drop the variant."
            )
        twin = next((other for other in resolved if other.name == name), None)
        if twin is not None:
            raise VariantError(
                f"The settings {twin.entry.settings} and {entry.settings} both build as {name}: a value set to its default adds nothing to the name."
            )
        resolved.append(ResolvedVariant(entry=entry, name=name))
    return resolved


def build_variant(variant: ResolvedVariant, top: Path) -> BuiltVariant:
    """Build a variant for the host and for rv64, and read both records.

    Parameters
    ----------
    variant : ResolvedVariant
        The variant and its name.
    top : Path
        The git top level.

    Returns
    -------
    BuiltVariant
        The folders and the checked records.

    Raises
    ------
    VariantError
        When a build fails.
    typer.Exit
        With `EXIT_REFUSED` when a build's record is missing or doesn't match its binary.
    """

    folders: dict[str, Path] = {}
    for target in (HOST_TARGET, RV64_TARGET):
        # No `-j`, the way `build-variant` builds
        process = subprocess.run(_make_command(variant.entry, target), cwd=top)
        if process.returncode:
            raise VariantError(
                f"Building {variant.name} for {target} failed with exit code {process.returncode}."
            )
        folders[target] = build_path(top / MAKE_FOLDER, variant.name, target)
    records = {
        target: read_checked_build(folders[target], (CLEAN_BINARY,))
        for target in folders
    }
    for target, record in records.items():
        if record.name != variant.name:
            raise VariantError(
                f"The {target} build's record is named {record.name}, not {variant.name}: rebuild it."
            )
    return BuiltVariant(
        name=variant.name,
        host_folder=folders[HOST_TARGET],
        rv64_folder=folders[RV64_TARGET],
        host=records[HOST_TARGET],
        rv64=records[RV64_TARGET],
    )
