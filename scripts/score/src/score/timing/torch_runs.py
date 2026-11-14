"""Timing IBM's PyTorch pipeline, each unit in a freshly spawned process.

A fresh process per unit sets its thread count and CPU before PyTorch does any work, and gives each unit its own peak memory.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import multiprocessing
import os
import resource
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

# External
import torch

# Internal
from score.capture import CONTEXT
from score.layout import MODEL_FOLDER
from score.pipeline import (
    LABEL,
    CapturingPipeline,
    build_pipeline,
    load_model,
    load_series,
)


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The warm-up call covers one full batch of the largest batch size, so oneDNN builds its kernels before the timed call. Each window needs CONTEXT past rows before it
WARM_UP_WINDOWS = 128
WARM_UP_ROWS = CONTEXT + WARM_UP_WINDOWS

# Threads for the pinned setups
SINGLE_THREAD = 1

# The head whose errors count the windows a call ran; every head has one per window
COUNTED_HEAD = "time"

# Where Linux lists the process's threads, and the line of each thread's status that holds the CPUs it may run on, such as `0-3,8`
TASKS_FOLDER = Path("/proc/self/task")
STATUS_FILE = "status"
CPUS_ALLOWED_FIELD = "Cpus_allowed_list:"

# The line of this process's status that holds its peak resident memory, in kB
SELF_STATUS = Path("/proc/self/status")
PEAK_MEMORY_FIELD = "VmHWM:"
CPU_LIST_SEPARATOR = ","
CPU_RANGE_SEPARATOR = "-"

# What the child sends back: a measurement, or the traceback of what it raised
OK = "ok"
FAILED = "failed"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class TorchJob:
    """What the spawned process needs to time one unit.

    Attributes
    ----------
    data : Path
        The folder with the data.
    csv : Path
        The series' CSV, relative to `data`.
    cap : int or None
        The most windows run, or None for every window.
    batch_size : int
        Windows per model call.
    threads : int
        PyTorch's intra-op threads.
    pin_cpu : int or None
        The CPU the process is pinned to, or None to leave it free.
    """

    data: Path
    csv: Path
    cap: int | None
    batch_size: int
    threads: int
    pin_cpu: int | None


@dataclass(frozen=True)
class TorchMeasurement:
    """What a PyTorch unit measured.

    Attributes
    ----------
    windows : int
        Windows the timed call ran, from the length of its errors.
    channels : int
        How many channels the series has.
    forward_seconds : float
        Wall seconds inside the pipeline's `_forward`, the model calls and the error maths.
    call_seconds : float
        Wall seconds of the whole pipeline call, pre- and post-processing included.
    cpu_seconds : float
        User plus system CPU time of the process during the timed call.
    max_rss_kib : int
        Peak resident memory of this process in KiB (`VmHWM`), the model load included.
    threads : int
        `torch.get_num_threads()` during the call.
    affinity : list[int]
        The CPUs any of the process's threads was allowed to run on, just before the timed call.
    """

    windows: int
    channels: int
    forward_seconds: float
    call_seconds: float
    cpu_seconds: float
    max_rss_kib: int
    threads: int
    affinity: list[int]


class TimingPipeline(CapturingPipeline):
    """The capturing pipeline, adding up the wall time spent in `_forward`.

    Attributes
    ----------
    forward_seconds : float
        Seconds spent in `_forward` since the instance was made.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Set up the pipeline as the parent does, with the counter at zero.

        Parameters
        ----------
        *args : Any
            Passed to `CapturingPipeline` unchanged.
        **kwargs : Any
            Passed to `CapturingPipeline` unchanged.
        """

        super().__init__(*args, **kwargs)
        self.forward_seconds = 0.0

    def _forward(self, input_tensors: Any, **kwargs: Any) -> Any:
        """Run the parent's `_forward`, the model calls and the error maths for one batch, and add its time.

        Parameters
        ----------
        input_tensors : Any
            One batch, as `run_single` passes it.
        **kwargs : Any
            Passed to the parent unchanged.

        Returns
        -------
        Any
            The parent's per-head errors for the batch.
        """

        began = time.perf_counter()
        scores = super()._forward(input_tensors, **kwargs)
        self.forward_seconds += time.perf_counter() - began
        return scores


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def _cpu_seconds() -> float:
    """Read the process's CPU time so far.

    Returns
    -------
    float
        User plus system seconds.
    """

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _parse_cpu_list(text: str) -> set[int]:
    """Read a kernel CPU list.

    Parameters
    ----------
    text : str
        Such as `0-3,8`.

    Returns
    -------
    set[int]
        The CPUs it names.
    """

    cpus: set[int] = set()
    for part in text.split(CPU_LIST_SEPARATOR):
        first, _, last = part.partition(CPU_RANGE_SEPARATOR)
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def _threads_affinity() -> list[int]:
    """Read the CPUs any thread of this process may run on.

    `os.sched_getaffinity(0)` reads only the calling thread, and the threads that imports and libraries start keep their own masks.

    Returns
    -------
    list[int]
        The union of every thread's `Cpus_allowed_list`, sorted.
    """

    cpus: set[int] = set()
    for task in TASKS_FOLDER.iterdir():
        # A thread can end between the listing and the read
        try:
            lines = (task / STATUS_FILE).read_text().splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if line.startswith(CPUS_ALLOWED_FIELD):
                cpus |= _parse_cpu_list(line.removeprefix(CPUS_ALLOWED_FIELD).strip())
    return sorted(cpus)


def _peak_memory_kib() -> int:
    """Read this process's peak resident memory.

    The kernel's `VmHWM` belongs to the address space made at exec, so it holds this process alone. `ru_maxrss` is not used: Linux carries the forking parent's resident size into it, and a fresh spawned child read 324 MB there against 17.7 MB in `VmHWM`.

    Returns
    -------
    int
        `VmHWM` in KiB, which the kernel writes as `kB`.

    Raises
    ------
    ValueError
        When the status has no `VmHWM` line.
    """

    for line in SELF_STATUS.read_text().splitlines():
        if line.startswith(PEAK_MEMORY_FIELD):
            return int(line.removeprefix(PEAK_MEMORY_FIELD).split()[0])
    raise ValueError(f"{SELF_STATUS} has no {PEAK_MEMORY_FIELD} line")


def _measure(job: TorchJob) -> TorchMeasurement:
    """Time one pipeline call, in the spawned process.

    Parameters
    ----------
    job : TorchJob
        The series, batch size, threads and CPU.

    Returns
    -------
    TorchMeasurement
        The timed call's numbers.
    """

    # Step 1: the threads, before PyTorch does any work. The CPU is already set, since `run_torch` starts the process pinned
    torch.set_num_threads(job.threads)

    # Step 2: the series, cut to its first cap windows, and the model
    series = load_series(job.data / job.csv)
    if job.cap is not None:
        series = series.iloc[: job.cap + CONTEXT]
    channels = series.columns.drop(LABEL).tolist()
    model = load_model(job.data / MODEL_FOLDER, len(channels))

    # Step 3: one untimed call, because oneDNN builds its kernels on the first call
    warm_up = build_pipeline(TimingPipeline, model, channels, job.batch_size)
    warm_up(series[channels].iloc[:WARM_UP_ROWS])

    # Step 4: the timed call, on a fresh instance so its counter holds this call's seconds alone. The affinity is read just before it, from every thread, to show the pin held
    pipeline = build_pipeline(TimingPipeline, model, channels, job.batch_size)
    affinity = _threads_affinity()
    cpu_before = _cpu_seconds()
    began = time.perf_counter()
    pipeline(series[channels])
    call_seconds = time.perf_counter() - began
    cpu_seconds = _cpu_seconds() - cpu_before
    max_rss_kib = _peak_memory_kib()

    return TorchMeasurement(
        windows=len(pipeline.errors[COUNTED_HEAD]),
        channels=len(channels),
        forward_seconds=pipeline.forward_seconds,
        call_seconds=call_seconds,
        cpu_seconds=cpu_seconds,
        max_rss_kib=max_rss_kib,
        threads=torch.get_num_threads(),
        affinity=affinity,
    )


def _child(job: TorchJob, send: Connection) -> None:
    """The spawned process's entry: measure, and send the result or the traceback back.

    Parameters
    ----------
    job : TorchJob
        What to time.
    send : Connection
        The pipe's sending end.
    """

    try:
        send.send((OK, _measure(job)))
    except Exception as error:
        send.send((FAILED, "".join(traceback.format_exception(error))))
    finally:
        send.close()


def run_torch(job: TorchJob) -> TorchMeasurement | str:
    """Time one PyTorch unit in a freshly spawned process.

    Parameters
    ----------
    job : TorchJob
        What to time.

    Returns
    -------
    TorchMeasurement or str
        The measurement, or what went wrong.
    """

    # Spawn, because forking a parent that has loaded torch can deadlock
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(job, send))
    try:
        # The child is started from a pinned thread, so it is born with the mask, and so is every thread its imports and libraries start. The parent's own mask comes back right after
        previous = os.sched_getaffinity(0)
        if job.pin_cpu is not None:
            os.sched_setaffinity(0, {job.pin_cpu})
        try:
            process.start()
        finally:
            os.sched_setaffinity(0, previous)

        # The parent's copy of the sending end is closed, so a child that dies without sending ends `recv` with EOFError instead of hanging it
        send.close()
        try:
            status, payload = receive.recv()
        except EOFError:
            process.join()
            return f"the PyTorch process ended with exit code {process.exitcode} before sending a result"
        process.join()
    except BaseException:
        if process.is_alive():
            process.kill()
            process.join()
        raise
    finally:
        receive.close()
    if status == FAILED:
        return f"the PyTorch process raised:\n{payload}"
    return payload
