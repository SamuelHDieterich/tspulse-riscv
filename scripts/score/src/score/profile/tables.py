"""One window's numbers from its three runs, and the six tidy CSVs of all the windows."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
import csv
import filecmp
import io
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Internal
from score.profile.classes import DEVICE_CLASSES, MEGAHERTZ, class_timing
from score.profile.cycles import CycleRange, Weight, cycle_range
from score.profile.errors import ProfileError
from score.profile.histogram import (
    Histogram,
    largest_first,
    read_histogram,
    subtract,
)
from score.profile.kernels import kernel_counts
from score.profile.memory import memory_rows
from score.profile.mix import mix_counts
from score.profile.modules import ModuleRow, nest_modules, read_profile
from score.profile.symbols import FunctionTable
from score.profile.traffic import (
    Traffic,
    read_cache_report,
    traffic_floor,
    window_traffic,
)
from score.profile.units import (
    COUNTS_FILE,
    ERRORS_FILE,
    PROFILE_FILE,
    STDERR_FILE,
    STDOUT_FILE,
    Kind,
    ProfileSettings,
    ProfileUnit,
    read_counts,
)
from score.profile.weights import read_tensor_sizes
from score.spike_results import _replace


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The table files
KERNELS_TABLE = "kernels.csv"
MODULES_TABLE = "modules.csv"
MIX_TABLE = "mix.csv"
CYCLES_TABLE = "cycles.csv"
MEMORY_TABLE = "memory.csv"
TRAFFIC_TABLE = "traffic.csv"

# What the `window` column holds on the row that averages the windows, and on a fact that doesn't depend on the window
MEAN_WINDOW = "mean"
ANY_WINDOW = "all"

# The histogram's difference holds the window's leftover work too (reading and pushing one row, writing a line to each CSV), so it may sit this far above the program's own count
HISTOGRAM_TOLERANCE = 1e-4

# The profile build's root, with its hooks subtracted, may differ this much from the clean build's count
PROFILE_TOLERANCE = 1e-3

# The cycle rows' own name for what is not tied to a device class, and the quantities they carry
CORE_CLASS = "all classes"
CYCLES = "cycles"
CLOCK = "clock"
SECONDS = "seconds per window"
READINGS = "readings per second"

# Decimals of a time or a rate, so a rerun writes the same bytes
DECIMALS = 6

# The traffic table's measures, in the order they are written
TRAFFIC_FLOOR = "floor_bytes"


# ░█▀▀░█░░░█▀█░█▀▀░█▀▀░█▀▀░█▀▀
# ░█░░░█░░░█▀█░▀▀█░▀▀█░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀


@dataclass(frozen=True)
class WindowNumbers:
    """Everything one measured window gives.

    Attributes
    ----------
    window : int
        The measured window.
    kernels : dict[str, int]
        Instructions per function, pk's slice included.
    mix : dict[str, int]
        Instructions per mnemonic of the program.
    cycles : CycleRange
        The window's cycles at both ends.
    traffic : Traffic
        The window's cache misses, writebacks and bytes.
    modules : list[ModuleRow]
        The nested module rows, roots first.
    difference : Histogram
        Instructions per address, the longer run minus the shorter.
    """

    window: int
    kernels: dict[str, int]
    mix: dict[str, int]
    cycles: CycleRange
    traffic: Traffic
    modules: list[ModuleRow]
    difference: Histogram


# ░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
# ░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
# ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀


def analyse_window(
    window: int,
    settings: ProfileSettings,
    functions: FunctionTable,
    mnemonics: dict[int, str],
    weights: dict[str, Weight],
) -> WindowNumbers:
    """Turn a window's three runs into its numbers, after checking them against each other.

    Parameters
    ----------
    window : int
        The measured window.
    settings : ProfileSettings
        Where the runs' folders are.
    functions : FunctionTable
        The functions of the clean ELF.
    mnemonics : dict[int, str]
        The mnemonic at each address of the clean ELF.
    weights : dict[str, Weight]
        The cycle weights.

    Returns
    -------
    WindowNumbers
        The window's numbers.

    Raises
    ------
    ProfileError
        If a run's files can't be read, the profile build's errors differ from the clean build's, or a total disagrees with the program's own count beyond its tolerance.
    """

    short = settings.folder(ProfileUnit(window, Kind.SHORT))
    long = settings.folder(ProfileUnit(window, Kind.LONG))
    profiled = settings.folder(ProfileUnit(window, Kind.PROFILE))

    # Step 1: the profile build must compute what the clean build does, byte for byte
    if not filecmp.cmp(long / ERRORS_FILE, profiled / ERRORS_FILE, shallow=False):
        raise ProfileError(f"window {window}: the profile build's errors differ")

    # Step 2: kernels, mix and cycles from the histograms' difference
    longer = read_histogram(long / STDERR_FILE)
    shorter = read_histogram(short / STDERR_FILE)
    difference = subtract(longer, shorter)
    kernels = kernel_counts(difference, functions)
    mix = mix_counts(difference, mnemonics)
    cycles = cycle_range(mix, weights)

    # Step 3: traffic from the cache counters' difference
    traffic = window_traffic(
        read_cache_report(long / STDOUT_FILE), read_cache_report(short / STDOUT_FILE)
    )

    # Step 4: the modules, with the hooks' cost subtracted at every level
    modules = nest_modules(read_profile(profiled / PROFILE_FILE, window))

    # Step 5: the totals must agree with the program's own `--counts` row for this window
    counted = read_counts(long / COUNTS_FILE)[window]
    histogram_total = sum(difference.values())
    if not counted <= histogram_total <= counted * (1 + HISTOGRAM_TOLERANCE):
        raise ProfileError(
            f"window {window}: the histogram difference is {histogram_total:,} instructions, the program counted {counted:,}"
        )
    root = modules[0].instructions
    if abs(root - counted) > counted * PROFILE_TOLERANCE:
        raise ProfileError(
            f"window {window}: the profile build's root is {root:,} instructions, the program counted {counted:,}"
        )
    return WindowNumbers(window, kernels, mix, cycles, traffic, modules, difference)


def _mean(values: list[int]) -> int:
    """Average whole numbers, rounded to a whole number.

    Parameters
    ----------
    values : list[int]
        One value per window.

    Returns
    -------
    int
        The mean.
    """

    return round(sum(values) / len(values))


def _mean_counts(per_window: list[dict[str, int]]) -> dict[str, int]:
    """Average counts by name, a name a window lacks counting as 0.

    Parameters
    ----------
    per_window : list[dict[str, int]]
        One mapping per window.

    Returns
    -------
    dict[str, int]
        The means, largest first.
    """

    names = {name for counts in per_window for name in counts}
    means = {
        name: _mean([counts.get(name, 0) for counts in per_window]) for name in names
    }
    return largest_first(means)


def _write_csv(path: Path, header: list[str], rows: Iterable[list[object]]) -> None:
    """Write a table whole, through a temporary file.

    Parameters
    ----------
    path : Path
        The CSV.
    header : list[str]
        Its columns.
    rows : Iterable[list[object]]
        Its rows; None becomes an empty cell.
    """

    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if cell is None else cell for cell in row])
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace(path, text.getvalue())


def _count_rows(
    per_window: list[WindowNumbers], counts: list[dict[str, int]]
) -> list[list[object]]:
    """Make one row per window and name, then one per name for the mean.

    Parameters
    ----------
    per_window : list[WindowNumbers]
        The windows.
    counts : list[dict[str, int]]
        Each window's counts by name, in the same order.

    Returns
    -------
    list[list[object]]
        `[window, name, instructions]` rows.
    """

    rows: list[list[object]] = [
        [numbers.window, name, count]
        for numbers, window_counts in zip(per_window, counts, strict=True)
        for name, count in window_counts.items()
    ]
    rows += [[MEAN_WINDOW, name, count] for name, count in _mean_counts(counts).items()]
    return rows


def _module_rows(per_window: list[WindowNumbers]) -> list[list[object]]:
    """Make the module rows of each window, then the mean of each module.

    Parameters
    ----------
    per_window : list[WindowNumbers]
        The windows.

    Returns
    -------
    list[list[object]]
        `[window, path, depth, instructions, calls, self]` rows.
    """

    rows: list[list[object]] = [
        [numbers.window, m.path, m.depth, m.instructions, m.calls, m.own]
        for numbers in per_window
        for m in numbers.modules
    ]
    # A path a window lacks counts as 0, and the paths keep the first window's order, then the order new ones appear in
    by_path = [{m.path: m for m in numbers.modules} for numbers in per_window]
    first_seen: dict[str, ModuleRow] = {}
    for numbers in per_window:
        for m in numbers.modules:
            first_seen.setdefault(m.path, m)
    for path, first in first_seen.items():
        found = [modules.get(path) for modules in by_path]
        # A module a window lacks, or a row with no `own`, counts as 0
        own = (
            None
            if first.own is None
            else _mean([(m.own or 0) if m else 0 for m in found])
        )
        rows.append(
            [
                MEAN_WINDOW,
                path,
                first.depth,
                _mean([m.instructions if m else 0 for m in found]),
                _mean([m.calls if m else 0 for m in found]),
                own,
            ]
        )
    return rows


def _cycle_rows(cycles: CycleRange, window: str | int) -> list[list[object]]:
    """Make one window's cycle rows: its cycles, then each class's clock, time and rate.

    Parameters
    ----------
    cycles : CycleRange
        The window's cycles at both ends.
    window : str or int
        The window, or `MEAN_WINDOW`.

    Returns
    -------
    list[list[object]]
        `[window, class, quantity, low, high, unit, note]` rows.
    """

    rows: list[list[object]] = [
        [window, CORE_CLASS, CYCLES, cycles.low, cycles.high, "cycles", ""]
    ]
    for device in DEVICE_CLASSES:
        timing = class_timing(cycles, device)
        rows += [
            [
                window,
                device.name,
                CLOCK,
                device.clock_low // MEGAHERTZ,
                device.clock_high // MEGAHERTZ,
                "MHz",
                device.note,
            ],
            [
                window,
                device.name,
                SECONDS,
                f"{timing.seconds_low:.{DECIMALS}f}",
                f"{timing.seconds_high:.{DECIMALS}f}",
                "s",
                device.note,
            ],
            [
                window,
                device.name,
                READINGS,
                f"{timing.readings_low:.{DECIMALS}f}",
                f"{timing.readings_high:.{DECIMALS}f}",
                "readings/s",
                device.note,
            ],
        ]
    return rows


def _traffic_rows(per_window: list[WindowNumbers]) -> list[list[object]]:
    """Make the traffic rows of each window and their mean.

    Parameters
    ----------
    per_window : list[WindowNumbers]
        The windows.

    Returns
    -------
    list[list[object]]
        `[window, measure, value]` rows.
    """

    measures = [
        {
            **vars(numbers.traffic.counters),
            "data_bytes": numbers.traffic.data_bytes,
            "instruction_bytes": numbers.traffic.instruction_bytes,
            "total_bytes": numbers.traffic.total_bytes,
        }
        for numbers in per_window
    ]
    rows: list[list[object]] = [
        [numbers.window, name, value]
        for numbers, window_measures in zip(per_window, measures, strict=True)
        for name, value in window_measures.items()
    ]
    rows += [
        [MEAN_WINDOW, name, _mean([each[name] for each in measures])]
        for name in measures[0]
    ]
    return rows


def write_tables(
    folder: Path,
    per_window: list[WindowNumbers],
    elf: Path,
    functions: FunctionTable,
    weights_table: Path,
    stack_folder: Path,
    source_folder: Path,
) -> None:
    """Write the six tables.

    Parameters
    ----------
    folder : Path
        Where the tables go.
    per_window : list[WindowNumbers]
        The measured windows, in the order they are written.
    elf : Path
        The clean rv64 program.
    functions : FunctionTable
        The functions of `elf`.
    weights_table : Path
        The generated `weights_table.c`.
    stack_folder : Path
        The `.ci` files of the clean build.
    source_folder : Path
        The `tspulse` source folder.

    Raises
    ------
    ProfileError
        If the memory table can't be built.
    """

    # Step 1: kernels, modules and mix
    _write_csv(
        folder / KERNELS_TABLE,
        ["window", "function", "instructions"],
        _count_rows(per_window, [n.kernels for n in per_window]),
    )
    _write_csv(
        folder / MODULES_TABLE,
        ["window", "path", "depth", "instructions", "calls", "self"],
        _module_rows(per_window),
    )
    _write_csv(
        folder / MIX_TABLE,
        ["window", "mnemonic", "instructions"],
        _count_rows(per_window, [n.mix for n in per_window]),
    )

    # Step 2: cycles, and the mean from the mean cycles
    mean_cycles = CycleRange(
        _mean([n.cycles.low for n in per_window]),
        _mean([n.cycles.high for n in per_window]),
    )
    cycle_rows = [
        row for n in per_window for row in _cycle_rows(n.cycles, n.window)
    ] + _cycle_rows(mean_cycles, MEAN_WINDOW)
    _write_csv(
        folder / CYCLES_TABLE,
        ["window", "class", "quantity", "low", "high", "unit", "note"],
        cycle_rows,
    )

    # Step 3: memory, whose code row counts the functions any of the windows ran
    ran: Histogram = {}
    for numbers in per_window:
        for address, count in numbers.difference.items():
            ran[address] = ran.get(address, 0) + count
    memory = memory_rows(
        elf, functions, weights_table, stack_folder, source_folder, ran
    )
    _write_csv(
        folder / MEMORY_TABLE,
        ["kind", "item", "bytes", "formula", "source"],
        [[row.kind, row.item, row.bytes, row.formula, row.source] for row in memory],
    )

    # Step 4: traffic, then the least a window must stream, which no window changes
    floor = traffic_floor(read_tensor_sizes(weights_table))
    _write_csv(
        folder / TRAFFIC_TABLE,
        ["window", "measure", "value"],
        [*_traffic_rows(per_window), [ANY_WINDOW, TRAFFIC_FLOOR, floor]],
    )
