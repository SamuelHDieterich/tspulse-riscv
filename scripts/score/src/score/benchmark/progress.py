"""The benchmark's log: one plain line per event, on screen and appended to a file, with done/total and an ETA.

Plain lines rather than a redrawing screen, so `tail -f` over ssh shows the same thing as the terminal that runs it.
"""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import time
from datetime import datetime, timedelta
from pathlib import Path

# External
import typer

# Internal
from score.benchmark.results import Platform


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# Wall seconds per channel-window on one thread, for the first ETA before any unit has finished: the host's measured clean-build rate, and Spike's without `-g`
ASSUMED_SECONDS = {Platform.HOST: 0.2, Platform.SPIKE: 35.0}

# The local time each line starts with
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# The ETA's clock time: the time alone within a day, the date too beyond it
CLOCK_FORMAT = "%H:%M"
DATED_CLOCK_FORMAT = "%Y-%m-%d %H:%M"
ONE_DAY = timedelta(days=1)

# Between the fields of a line
SEPARATOR = " · "

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


class Log:
    """Prints each line with a local timestamp and appends the same line to a file.

    Only the thread that collects finished units writes to it, so lines never interleave.
    """

    def __init__(self, path: Path) -> None:
        """Make the log file's folder.

        Parameters
        ----------
        path : Path
            The log file, which several calls append to in turn.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def line(self, text: str) -> None:
        """Print one event and append it to the file.

        Parameters
        ----------
        text : str
            The event, without a timestamp.
        """

        stamped = f"{datetime.now().strftime(TIMESTAMP_FORMAT)} {text}"
        typer.echo(stamped)
        # Opened per line, so a `tail -f` elsewhere sees each line as soon as it is written
        with self.path.open("a") as file:
            file.write(stamped + "\n")


class StageProgress:
    """How far a stage has come, as finished channel-windows against the stage's total.

    The ETA is the remaining channel-windows over the channel-windows finished per wall second since the stage began.
    Units run largest first, so the tail is short units and the rate settles early.
    """

    def __init__(self, stage: str, units: int, channel_windows: int) -> None:
        """Start the stage's clock.

        Parameters
        ----------
        stage : str
            The stage's name, which starts each of its lines.
        units : int
            How many units the stage runs.
        channel_windows : int
            How many channel-windows they hold.
        """

        self.stage = stage
        self.units = units
        self.channel_windows = channel_windows
        self.done_units = 0
        self.done_channel_windows = 0
        self.began = time.perf_counter()

    def finished(self, name: str, cost: int, detail: str) -> str:
        """Count a finished unit, failed or not, and describe it.

        Parameters
        ----------
        name : str
            The unit, such as `090_SVDB w5000-7500`.
        cost : int
            Its channel-windows.
        detail : str
            What the unit did, such as `5,000 ch-win in 1,104 s`.

        Returns
        -------
        str
            Such as `accuracy 37/212 · 090_SVDB w5000-7500 · 5,000 ch-win in 1,104 s · 185,000/765,221 ch-win (24.2%) · 26.3 ch-win/s · ETA 07:12 (6 h 58 min)`.
        """

        self.done_units += 1
        self.done_channel_windows += cost
        elapsed = time.perf_counter() - self.began

        # rate = finished channel-windows / elapsed seconds; ETA = remaining channel-windows / rate
        rate = self.done_channel_windows / elapsed if elapsed > 0 else 0.0
        remaining = self.channel_windows - self.done_channel_windows
        share = 100 * self.done_channel_windows / max(self.channel_windows, 1)
        fields = [
            f"{self.stage} {self.done_units}/{self.units}",
            name,
            detail,
            f"{self.done_channel_windows:,}/{self.channel_windows:,} ch-win ({share:.1f}%)",
            _rate(rate),
        ]
        if remaining > 0 and rate > 0:
            fields.append(f"ETA {eta(remaining / rate)}")
        return SEPARATOR.join(fields)

    def elapsed(self) -> float:
        """The stage's wall seconds so far.

        Returns
        -------
        float
            Seconds since the stage began.
        """

        return time.perf_counter() - self.began


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def duration(seconds: float) -> str:
    """Write a duration the way a person reads it.

    Parameters
    ----------
    seconds : float
        The duration.

    Returns
    -------
    str
        Such as `6 h 58 min`, `12 min` or `40 s`.
    """

    whole = round(seconds)
    if whole < SECONDS_PER_MINUTE:
        return f"{whole} s"
    hours, rest = divmod(whole, SECONDS_PER_HOUR)
    minutes = rest // SECONDS_PER_MINUTE
    if hours:
        return f"{hours:,} h {minutes} min"
    return f"{minutes} min"


def _rate(channel_windows_per_second: float) -> str:
    """Write a stage's measured speed so it reads well both on the host and under Spike.

    Parameters
    ----------
    channel_windows_per_second : float
        Finished channel-windows per wall second.

    Returns
    -------
    str
        Such as `26.3 ch-win/s`, or `29.4 s/ch-win` below one channel-window per second, which Spike always is.
    """

    if channel_windows_per_second >= 1 or channel_windows_per_second == 0:
        return f"{channel_windows_per_second:.1f} ch-win/s"
    return f"{1 / channel_windows_per_second:.1f} s/ch-win"


def eta(seconds: float) -> str:
    """Write when something ends, as a clock time and a duration from now.

    Parameters
    ----------
    seconds : float
        How long until it ends.

    Returns
    -------
    str
        Such as `07:12 (6 h 58 min)`, with the date when it is more than a day away.
    """

    span = timedelta(seconds=seconds)
    clock = datetime.now() + span
    form = CLOCK_FORMAT if span < ONE_DAY else DATED_CLOCK_FORMAT
    return f"{clock.strftime(form)} ({duration(seconds)})"
