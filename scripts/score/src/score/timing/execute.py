"""Running one `time-host` unit with its runner, and turning what it measured into the result every unit saves."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

# Internal
from score.timing.c_runs import PreparedSeries, run_one_core, run_whole_pc
from score.timing.results import Measurement, Provenance, UnitResult
from score.timing.torch_runs import SINGLE_THREAD, TorchJob, run_torch
from score.timing.units import (
    ONE_CORE,
    SERIES_CAP,
    SERIES_CSV,
    TORCH,
    TORCH_BATCH_SIZE,
    WHOLE_PC_TORCH_SETUPS,
    Runner,
    TorchSetup,
    Unit,
)


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class Settings:
    """What every unit of one call shares.

    Attributes
    ----------
    data : Path
        The folder with the data.
    work : Path
        The folder for the row files and the C program's output.
    machine : str
        The label of the computer, which names the C units' work folder.
    binaries : dict[Runner, Path]
        Each C runner's `tspulse`.
    launcher : Path
        GNU time, which starts every C process and measures its peak memory.
    pin_cpu : int
        The CPU the pinned setups run on.
    parallel : int
        P, the processes or threads of the whole-PC setups.
    prepared : dict[str, PreparedSeries]
        Each series' row file, by short name, for the series that have C units to run.
    provenance : dict[Runner, Provenance]
        Each runner's provenance.
    """

    data: Path
    work: Path
    machine: str
    binaries: dict[Runner, Path]
    launcher: Path
    pin_cpu: int
    parallel: int
    prepared: dict[str, PreparedSeries]
    provenance: dict[Runner, Provenance]


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _torch_job(unit: Unit, settings: Settings) -> TorchJob:
    """Describe a PyTorch unit for its spawned process.

    Parameters
    ----------
    unit : Unit
        A PyTorch unit.
    settings : Settings
        What every unit shares.

    Returns
    -------
    TorchJob
        The series, batch size, threads and CPU.
    """

    setup = cast(TorchSetup, unit.setup)
    whole_pc = setup in WHOLE_PC_TORCH_SETUPS
    return TorchJob(
        data=settings.data,
        csv=SERIES_CSV[unit.series],
        cap=SERIES_CAP[unit.series],
        batch_size=TORCH_BATCH_SIZE[setup],
        threads=settings.parallel if whole_pc else SINGLE_THREAD,
        pin_cpu=None if whole_pc else settings.pin_cpu,
    )


def _measure(unit: Unit, settings: Settings) -> tuple[int, Measurement] | str:
    """Run a unit with its runner.

    Parameters
    ----------
    unit : Unit
        The unit.
    settings : Settings
        What every unit shares.

    Returns
    -------
    tuple[int, Measurement] or str
        The series' channels and the measurement, or what went wrong.
    """

    # Step 1: a C unit, one pinned process or P free ones
    if unit.runner != TORCH:
        prepared = settings.prepared[unit.series]
        binary = settings.binaries[unit.runner]
        folder = settings.work / settings.machine / unit.name
        if unit.setup == ONE_CORE:
            measured = run_one_core(
                settings.launcher, binary, prepared, folder, settings.pin_cpu
            )
        else:
            measured = run_whole_pc(
                settings.launcher, binary, prepared, folder, settings.parallel
            )
        return measured if isinstance(measured, str) else (prepared.channels, measured)

    # Step 2: a PyTorch unit, timed inside `_forward`, with the whole call kept beside it
    job = _torch_job(unit, settings)
    torch_measured = run_torch(job)
    if isinstance(torch_measured, str):
        return torch_measured
    details: dict[str, Any] = asdict(torch_measured) | {
        "batch_size": job.batch_size,
        "pin_cpu": job.pin_cpu,
    }
    return torch_measured.channels, Measurement(
        windows=torch_measured.windows,
        seconds_per_window=torch_measured.forward_seconds / torch_measured.windows,
        timed_seconds=torch_measured.forward_seconds,
        cpu_seconds=torch_measured.cpu_seconds,
        max_rss_kib=torch_measured.max_rss_kib,
        details=details,
    )


def run_unit(unit: Unit, settings: Settings) -> UnitResult | str:
    """Run a unit and shape what it measured into the result every unit saves.

    Parameters
    ----------
    unit : Unit
        The unit.
    settings : Settings
        What every unit shares.

    Returns
    -------
    UnitResult or str
        The result, not yet saved, or what went wrong.
    """

    # The load just before the unit, so a busy machine shows in the result
    load_average = list(os.getloadavg())
    began = time.perf_counter()
    measured = _measure(unit, settings)
    unit_seconds = time.perf_counter() - began
    if isinstance(measured, str):
        return measured
    channels, measurement = measured
    return UnitResult(
        unit=unit.name,
        repeat=unit.repeat,
        series=unit.series,
        runner=unit.runner,
        setup=unit.setup,
        channels=channels,
        windows=measurement.windows,
        seconds_per_window=measurement.seconds_per_window,
        timed_seconds=measurement.timed_seconds,
        cpu_seconds=measurement.cpu_seconds,
        max_rss_kib=measurement.max_rss_kib,
        load_average=load_average,
        unit_seconds=unit_seconds,
        details=measurement.details,
        provenance=settings.provenance[unit.runner],
    )
