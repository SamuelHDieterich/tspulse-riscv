"""Where things live under the data folder, `$TSPULSE_DATA`."""

# ░█░░░▀█▀░█▀▄░█▀▄░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
# ░█░░░░█░░█▀▄░█▀▄░█▀█░█▀▄░░█░░█▀▀░▀▀█
# ░▀▀▀░▀▀▀░▀▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀▀▀░▀▀▀

# Built-in
from pathlib import Path


# ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀█░█▀█░▀█▀░█▀▀
# ░█░░░█░█░█░█░▀▀█░░█░░█▀█░█░█░░█░░▀▀█
# ░▀▀▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀░▀░▀░░▀░░▀▀▀

# The TSB-AD series, one folder per archive
DATASETS_FOLDER = Path("datasets")

# The tspulse-r1 checkpoint
MODEL_FOLDER = Path("models") / "tspulse-r1"

# The captured reference windows and the pooled noise ratios
REFERENCE_FOLDER = Path("reference")

# The host-against-Spike check, one folder per machine and one per unit inside it
SPIKE_CHECK_FOLDER = Path("results") / "spike-check"

# The benchmark's runs, one folder per label holding every machine's units
BENCHMARK_FOLDER = Path("results") / "benchmark"

# The host timing's runs, one folder and one log per machine
HOST_TIMING_FOLDER = Path("results") / "host-timing"
