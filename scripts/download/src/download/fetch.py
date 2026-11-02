"""Pinned files: where they come from, and how they are fetched and verified."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Failure(Exception):
    """A step failed in a way the user can act on, and the message says how."""


@dataclass(frozen=True)
class Pin:
    """One file the project depends on, exactly as it was checked.

    Attributes
    ----------
    path : str
        Where it lives, relative to `$TSPULSE_DATA`.
    url : str
        Where it is downloaded from.
    size : int
        Its size in bytes.
    sha256 : str
        Its hash, in hex.
    """

    path: str
    url: str
    size: int
    sha256: str


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀


# thedatum.org answers 403 to Python's default user agent
USER_AGENT = "tspulse-riscv"

# How many of the offending names a failure lists
SHOWN_NAMES = 5

TIMEOUT_SECONDS = 60

HUGGING_FACE = "https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1/resolve/2e64fcdc2a06d3565dfadaf0065c0ab5055f80f2"

DATASETS = {
    "u": Pin(
        "datasets/TSB-AD-U/TSB-AD-U.zip",
        "https://www.thedatum.org/datasets/TSB-AD-U.zip",
        72900868,
        "0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961",
    ),
    "m": Pin(
        "datasets/TSB-AD-M/TSB-AD-M.zip",
        "https://www.thedatum.org/datasets/TSB-AD-M.zip",
        540383983,
        "7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d",
    ),
}

WEIGHTS = [
    Pin(
        "models/tspulse-r1/config.json",
        f"{HUGGING_FACE}/config.json",
        3521,
        "e33230f47c91dfe4e848ee743924a62a26f710fe7f194566d92835f27c1c6d8b",
    ),
    Pin(
        "models/tspulse-r1/model.safetensors",
        f"{HUGGING_FACE}/model.safetensors",
        4374640,
        "57fa03b67d1473a7253ac37801b24c391d1fa931395db986743e2f59e556b9ac",
    ),
    Pin(
        "models/tspulse-r1/README.md",
        f"{HUGGING_FACE}/README.md",
        9248,
        "4b15eb7c6be130556a7023f5706a18146965a8a48163b358bf489e8c955fe83d",
    ),
    Pin(
        "models/tspulse-r1/tspulse_overview.webp",
        f"{HUGGING_FACE}/tspulse_overview.webp",
        295284,
        "2cfc0774815e88c19dd4f87328c1a5958b2106b861f57a8d89a3765b2b9715f0",
    ),
]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def data_root(output: Path | None) -> Path:
    """Check the folder that holds the project's data.

    Parameters
    ----------
    output : Path or None
        The folder from `--output` or `$TSPULSE_DATA`, or `None` when neither was given.

    Returns
    -------
    Path
        The same folder.

    Raises
    ------
    Failure
        When no folder was given or it isn't a directory.
    """

    if output is None:
        raise Failure(
            "No output folder. Pass --output, or run this inside the dev shell, from `nix develop`, which sets $TSPULSE_DATA."
        )
    if not output.is_dir():
        raise Failure(f"{output} is not a directory.")
    return output


def sha256(path: Path) -> str:
    """Hash a file.

    Parameters
    ----------
    path : Path
        The file.

    Returns
    -------
    str
        Its sha256, in hex.
    """

    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(pin: Pin, destination: Path) -> None:
    """Download a pinned file, keeping it only if its hash matches.

    Parameters
    ----------
    pin : Pin
        The file to download.
    destination : Path
        Where it goes. Any file already there survives a failed download.

    Raises
    ------
    Failure
        When the download fails or the result doesn't match the pin.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(pin.url, headers={"User-Agent": USER_AGENT})

    try:
        with (
            urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response,
            part.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)

        actual = sha256(part)
        if actual != pin.sha256:
            raise Failure(
                f"{pin.url} gave {part.stat().st_size} bytes with sha256 {actual}, but the pin is {pin.size} bytes with sha256 {pin.sha256}."
            )
        part.replace(destination)
    except OSError as error:
        raise Failure(f"Could not download {pin.url}: {error}") from error
    finally:
        part.unlink(missing_ok=True)


def ensure(pin: Pin, root: Path, force: bool) -> bool:
    """Make one pinned file correct.

    Parameters
    ----------
    pin : Pin
        The file.
    root : Path
        The folder its path is relative to.
    force : bool
        Whether to replace a file whose hash differs.

    Returns
    -------
    bool
        Whether it was downloaded.

    Raises
    ------
    Failure
        When the file differs and `force` is off, or the download fails.
    """

    destination = root / pin.path
    if destination.exists():
        actual = sha256(destination)
        if actual == pin.sha256:
            return False
        if not force:
            raise Failure(
                f"{destination} has sha256 {actual}, but the pin is {pin.sha256}. Pass --force to replace it."
            )

    download(pin, destination)
    return True


def ensure_extracted(archive: Path, force: bool) -> int:
    """Make the CSVs beside a zip match the zip's listing, by name and size.

    Parameters
    ----------
    archive : Path
        The verified zip. Its members carry its folder's name as a prefix.
    force : bool
        Whether to re-extract CSVs of a different size and delete CSVs the zip doesn't list.

    Returns
    -------
    int
        How many CSVs were extracted.

    Raises
    ------
    Failure
        When CSVs differ or are extra and `force` is off.
    """

    folder = archive.parent
    on_disk = {path.name: path for path in folder.glob("*.csv")}

    with zipfile.ZipFile(archive) as bundle:
        members = {
            PurePosixPath(member.filename).name: member
            for member in bundle.infolist()
            if member.filename.endswith(".csv")
        }

        missing = sorted(members.keys() - on_disk.keys())
        extra = sorted(on_disk.keys() - members.keys())
        different = sorted(
            name
            for name in members.keys() & on_disk.keys()
            if on_disk[name].stat().st_size != members[name].file_size
        )

        problems = different + extra
        if problems and not force:
            shown = ", ".join(problems[:SHOWN_NAMES])
            raise Failure(
                f"{folder}: {len(problems)} CSVs don't match {archive.name}, starting with {shown}. Pass --force to repair them."
            )

        for name in extra:
            on_disk[name].unlink()
        for name in missing + different:
            bundle.extract(members[name], folder.parent)

    return len(missing) + len(different)
