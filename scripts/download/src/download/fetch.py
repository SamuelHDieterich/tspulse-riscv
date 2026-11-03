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

TSB_AD_LISTS = "https://raw.githubusercontent.com/TheDatumOrg/TSB-AD/303dffbfc11612995b17b06633f8c9d0ea0e38e5/Datasets/File_List"

IBM_SCORES = "https://raw.githubusercontent.com/ibm-granite/granite-tsfm/09c8abf70c4f971e932c54316d1637fa1a58f2af/notebooks/hfdemo/tspulse/anomaly_detection/benchmarks"

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

BENCHMARK_LISTS = [
    Pin(
        "lists/tsb-ad/TSB-AD-U-Tuning.csv",
        f"{TSB_AD_LISTS}/TSB-AD-U-Tuning.csv",
        2177,
        "7bf24d2ef834bb39ddd1c8c2b02c177339921dce694354e258ed2b77d8d5cd1c",
    ),
    Pin(
        "lists/tsb-ad/TSB-AD-U-Eva.csv",
        f"{TSB_AD_LISTS}/TSB-AD-U-Eva.csv",
        16205,
        "6f4e1d4ddbde195f9687f2c5a951b51faff54dda4b658d85f0c07e3e2879615a",
    ),
    Pin(
        "lists/tsb-ad/TSB-AD-M-Tuning.csv",
        f"{TSB_AD_LISTS}/TSB-AD-M-Tuning.csv",
        912,
        "b388541efcd8972cb2084f0aabe7c36bcf0e1ecd52adc6d0ef874afec335fa51",
    ),
    Pin(
        "lists/tsb-ad/TSB-AD-M-Eva.csv",
        f"{TSB_AD_LISTS}/TSB-AD-M-Eva.csv",
        8110,
        "ef6353b5aad8519fb2e5ff366ae21cf626ecd56d1a05942ac3b5c02d9e4b32a4",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Tuning-time.csv",
        f"{IBM_SCORES}/TSB-AD-U-Tuning-time.csv",
        2961,
        "2d8d1f4a6e01211150b847ef317e25d66e177eb642a3808a416f20b7787dd1c9",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Tuning-fft.csv",
        f"{IBM_SCORES}/TSB-AD-U-Tuning-fft.csv",
        2961,
        "7c3b9ce4ca577c813e6c7a9956064fbf1ad855ca4f595f2bb059ff70ce7288b3",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Tuning-future.csv",
        f"{IBM_SCORES}/TSB-AD-U-Tuning-future.csv",
        2961,
        "7651531ec13e3a93ff3c1dbfd0c5fe80702b6842943992bebb49776025adb710",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Tuning-ensemble.csv",
        f"{IBM_SCORES}/TSB-AD-U-Tuning-ensemble.csv",
        2961,
        "47998f89595abe7037cc62328ca68231373b920217ae0a5ff4d839364af6e09f",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Eva-time.csv",
        f"{IBM_SCORES}/TSB-AD-U-Eva-time.csv",
        21821,
        "2f0d927d58883343c452108b2599628e59fe62c382c81d6797d8630287c13b59",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Eva-fft.csv",
        f"{IBM_SCORES}/TSB-AD-U-Eva-fft.csv",
        21821,
        "22d810f1e39d021e058ee6a4b720c9259328b67991b42150790a054a2ad1fe51",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Eva-future.csv",
        f"{IBM_SCORES}/TSB-AD-U-Eva-future.csv",
        21821,
        "46a85d7b53859f9965de804e23b6732a5d9928f0140cd9db924fb168d73da725",
    ),
    Pin(
        "lists/ibm/TSB-AD-U-Eva-ensemble.csv",
        f"{IBM_SCORES}/TSB-AD-U-Eva-ensemble.csv",
        21821,
        "0123172d46ea66743588cb3134bccce6688df2f2e50a5f4ac9944ada0a8d7b2c",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Tuning-time.csv",
        f"{IBM_SCORES}/TSB-AD-M-Tuning-time.csv",
        1248,
        "0b59942ff3f26a38091e1ed2bda58db148078c000d51b01e6a6f54cbe11af8dd",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Tuning-fft.csv",
        f"{IBM_SCORES}/TSB-AD-M-Tuning-fft.csv",
        1248,
        "18ada9e8b2969b3b6143163c246383b4eff763d36452872e97e32e37d9fa0947",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Tuning-future.csv",
        f"{IBM_SCORES}/TSB-AD-M-Tuning-future.csv",
        1248,
        "35de30d65d4652d77feb8a00a76e7a24edb4fe58468e4f9ede4fa7e34b02ae63",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Tuning-ensemble.csv",
        f"{IBM_SCORES}/TSB-AD-M-Tuning-ensemble.csv",
        1248,
        "5d4cba22578011c13dcd104bf48b4189270d3ebdd8ae3d47eb4793c2a4eb3401",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Eva-time.csv",
        f"{IBM_SCORES}/TSB-AD-M-Eva-time.csv",
        11005,
        "e2446359f6eae9e553979d2e6f185f91ec9e20c2b6ad7172d5845c1cd534c0ae",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Eva-fft.csv",
        f"{IBM_SCORES}/TSB-AD-M-Eva-fft.csv",
        11005,
        "2fc2233ed2b333407668399d0d0cf4be2f3a3365c87e5339af0b945c697be8a4",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Eva-future.csv",
        f"{IBM_SCORES}/TSB-AD-M-Eva-future.csv",
        11005,
        "02bcb424da1ca747cb9a01ed90cfbfb7c9ee8c3d0afd5427814a5b6df8376c66",
    ),
    Pin(
        "lists/ibm/TSB-AD-M-Eva-ensemble.csv",
        f"{IBM_SCORES}/TSB-AD-M-Eva-ensemble.csv",
        11005,
        "121a0d65923271b65b0d0a679bf38dff7d829458f4e89d70f41a9206f8e0b99d",
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
