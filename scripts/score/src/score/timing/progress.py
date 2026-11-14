"""How far a `time-host` call has come, with an ETA from the units already timed."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import statistics

# Internal
from score.benchmark.progress import SEPARATOR, eta
from score.timing.units import Unit


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The word each progress line starts with
COMMAND = "time-host"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Progress:
    """Counts finished units and estimates when the rest will be done.

    A unit's time is the mean of its own measurement's earlier repeats when there are any, since series and setups differ by minutes; otherwise it is the mean over every unit finished so far.
    """

    def __init__(self, pending: list[Unit], known: dict[Unit, float]) -> None:
        """Start counting.

        Parameters
        ----------
        pending : list[Unit]
            The units this call runs, in order.
        known : dict[Unit, float]
            The whole wall seconds of units already saved, which seed the per-measurement times.
        """

        self.pending = pending
        self.done = 0
        self.finished_seconds: list[float] = []
        self.by_measurement: dict[tuple[str, str, str], list[float]] = {}
        for unit, seconds in known.items():
            self.by_measurement.setdefault(unit.measurement, []).append(seconds)

    def finished(self, unit: Unit, seconds: float, outcome: str, failed: bool) -> str:
        """Count a finished unit, failed or not, and describe it.

        Parameters
        ----------
        unit : Unit
            The unit.
        seconds : float
            Its whole wall time.
        outcome : str
            What it measured, or `failed`.
        failed : bool
            Whether it failed, which keeps its time out of the ETA, since a failure usually ends early.

        Returns
        -------
        str
            Such as `time-host 3/14 · r1-yahoo560-torch-b1-t1 · 21 s · 16.2 ms per window · ETA 22:41 (7 min)`.
        """

        self.done += 1
        if not failed:
            self.finished_seconds.append(seconds)
            self.by_measurement.setdefault(unit.measurement, []).append(seconds)

        # Step 1: each remaining unit's expected time, from its measurement's earlier repeats or the mean over the units finished well
        remaining = 0.0
        if self.finished_seconds:
            overall = statistics.fmean(self.finished_seconds)
            remaining = sum(
                statistics.fmean(self.by_measurement.get(one.measurement, [overall]))
                for one in self.pending[self.done :]
            )

        # Step 2: the line
        fields = [
            f"{COMMAND} {self.done}/{len(self.pending)}",
            unit.name,
            f"{seconds:.0f} s",
            outcome,
        ]
        if remaining > 0:
            fields.append(f"ETA {eta(remaining)}")
        return SEPARATOR.join(fields)
