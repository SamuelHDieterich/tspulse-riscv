# TSPulse on RISC-V: Edge anomaly detection

A C port of [TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1), simulated on RISC-V to study the feasibility of running a SOTA time series anomaly detection model on an embedded device, and the trade-off between accuracy and speed.

Final project for the Embedded Systems class (MIC08, UFRGS).

## Overview

1. **Edge sensor**: streams a time series.
2. **TSPulse**: a model of about 1M parameters that works on both time and frequency.
3. **C port**: only the C standard library, newlib on RISC-V, and no other libraries.
4. **RISC-V**: simulated with Spike.
5. **Trade-offs**: accuracy against speed, across quantization, numerical approximations and compiler optimizations.

## Getting started

You need [Nix](https://nixos.org/download/) with flakes enabled. Everything else comes from the repository.

```bash
nix develop           # or: direnv allow
```

`flake.nix` is generated from `flake/` by [flake-file](https://github.com/vic/flake-file), so don't edit it by hand. Run `nix run .#write-flake` after changing inputs.

Entering the shell installs a pre-commit hook that formats Python, Nix and C.


## Workflow

Every command runs inside the dev shell. A full run goes in this order:

1. `download-datasets`, `download-weights` and `download-lists` fetch the data into `$TSPULSE_DATA`.
2. `build-variant` builds the baseline program for the host and for RISC-V.
3. `capture-reference` saves PyTorch's output of every module on 7 fixed windows.
4. `score-baseline` scores the working series in Python, which `check-model --series` compares against.
5. `check-model` checks the C against PyTorch, and `check-spike` checks the RISC-V build against the host build.
6. `benchmark all tspulse/build/host tspulse/build/rv64` benchmarks the baseline under the label `baseline`.
7. `benchmark variants` builds, runs and compares each variant against the baseline.
8. `time-host` times the C builds against PyTorch on this computer.

## Commands

| Command | What it does | Docs |
| --- | --- | --- |
| `download-datasets`, `download-weights`, `download-lists` | Fetch or verify the pinned datasets, weights and lists | [Data](docs/data.md) |
| `select-series` | Regenerate `benchmark/series.csv`, the series the benchmark uses | [Data](docs/data.md#benchmark-series) |
| `prepare-rows` | Turn a CSV into the binary row file the C program reads | [Data](docs/data.md#row-files) |
| `make`, `export-weights` | Build the C program and its checks, with the weights compiled in | [C port](docs/c-port.md#building) |
| `tspulse` | Run the model on a row file and write each window's errors | [C port](docs/c-port.md#the-program) |
| `build-variant` | Build host and RISC-V programs with alternative code or flags | [C port](docs/c-port.md#variants) |
| `spike` | Run the RISC-V build and count its instructions | [C port](docs/c-port.md#running-under-spike) |
| `score-series` | Run IBM's TSPulse pipeline on one series | [Checks](docs/checks.md#reference-run) |
| `score-baseline` | Score the working series in Python, the baseline the C is held to | [Checks](docs/checks.md#python-baseline) |
| `capture-reference`, `compare-reference` | Save PyTorch's per-module outputs and compare a C dump with them | [Checks](docs/checks.md#reference-outputs) |
| `check-model` | Check the C against PyTorch, step by step and on whole series | [Checks](docs/checks.md#checking-the-c-against-pytorch) |
| `check-spike` | Check that the RISC-V build matches the host build byte for byte | [Checks](docs/checks.md#checking-the-risc-v-build) |
| `benchmark run`, `collect`, `all` | Run a build over the benchmark's windows and score it | [Benchmark](docs/benchmark.md) |
| `benchmark profile` | Show where a build's instructions, cycles and memory go | [Benchmark](docs/benchmark.md#profiling-a-build) |
| `benchmark variants` | Build, run and compare a list of variants against the baseline | [Running variants](docs/variants.md) |
| `time-host` | Time C builds against PyTorch on this computer | [Host timing](docs/host-timing.md) |
| `spike-matmul` | Project Spike core-hours from the matrix multiplies of one pass | [Spike cost estimate](docs/spike-cost.md) |

The Python commands print their options with `--help`.

## Repository

- `tspulse/`: the C port, its Makefile and its checks.
- `scripts/`: the Python tools behind the commands, one project per tool.
- `benchmark/`: the benchmark's series, the variant list, the Python baseline's tables and the tracked results.
- `experiments/`: the Spike cost estimate's matrix multiplies.
- `flake/`: the dev shell and its commands.
- `docs/`: how to use each command in detail.
