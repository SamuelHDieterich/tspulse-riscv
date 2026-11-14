"""The scalar device classes the port is judged against, and a window's time on each."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from dataclasses import dataclass

# Internal
from score.profile.cycles import CycleRange


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class DeviceClass:
    """One kind of chip a sensor could run the model on.

    Attributes
    ----------
    name : str
        The class's name.
    clock_low : int
        The slowest clock in Hz.
    clock_high : int
        The fastest clock in Hz.
    memory : str
        How much memory such chips have.
    boards : str
        Example boards.
    note : str
        A caveat on this class's times, empty when there is none.
    """

    name: str
    clock_low: int
    clock_high: int
    memory: str
    boards: str
    note: str


@dataclass(frozen=True)
class ClassTiming:
    """One window's time on one class, as a range.

    Attributes
    ----------
    seconds_low : float
        Seconds per window at the low cycle count and the fastest clock.
    seconds_high : float
        Seconds per window at the high cycle count and the slowest clock.
    readings_low : float
        The fastest sensor, in readings per second for one channel, that the slow end keeps up with.
    readings_high : float
        The same at the fast end.
    """

    seconds_low: float
    seconds_high: float
    readings_low: float
    readings_high: float


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

MEGAHERTZ = 1_000_000

MICROCONTROLLER = DeviceClass(
    name="Microcontroller",
    clock_low=100 * MEGAHERTZ,
    clock_high=400 * MEGAHERTZ,
    memory="under 1 MB SRAM",
    boards="ESP32-P4",
    note="Times come from rv64 instruction counts, a rough proxy for an rv32 chip.",
)

LARGE_MICROCONTROLLER = DeviceClass(
    name="Large microcontroller",
    clock_low=400 * MEGAHERTZ,
    clock_high=400 * MEGAHERTZ,
    memory="several MB SRAM",
    boards="Kendryte K210",
    note="",
)

SMALL_LINUX_BOARD = DeviceClass(
    name="Small Linux board",
    clock_low=1000 * MEGAHERTZ,
    clock_high=1000 * MEGAHERTZ,
    memory="tens of MB DRAM",
    boards="Milk-V Duo, Lichee RV (T-Head C906)",
    note="",
)

DEVICE_CLASSES = (MICROCONTROLLER, LARGE_MICROCONTROLLER, SMALL_LINUX_BOARD)


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def class_timing(cycles: CycleRange, device: DeviceClass) -> ClassTiming:
    """Turn a window's cycle range into seconds and readings per second on one class.

    Parameters
    ----------
    cycles : CycleRange
        The window's cycles at both ends.
    device : DeviceClass
        The class whose clock range applies.

    Returns
    -------
    ClassTiming
        The time range, and its inverse.
    """

    # seconds per window = cycles / clock, the CPU time equation
    # https://en.wikipedia.org/wiki/Cycles_per_instruction
    # readings per second = 1 / seconds per window, the fastest sensor the chip keeps up with
    seconds_low = cycles.low / device.clock_high
    seconds_high = cycles.high / device.clock_low
    return ClassTiming(
        seconds_low=seconds_low,
        seconds_high=seconds_high,
        readings_low=1 / seconds_high,
        readings_high=1 / seconds_low,
    )
