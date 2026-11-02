# TSPulse on RISC-V: Edge anomaly detection

A C port of [TSPulse](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1), simulated on RISC-V to study the feasibility of running a SOTA time series anomaly detection model on an embedded device, and the trade-off between accuracy and speed.

Final project for the Embedded Systems class (MIC08, UFRGS).

## Overview

1. **Edge sensor**: streams a time series.
2. **TSPulse**: a model of about 1M parameters that works on both time and frequency.
3. **C port**: static memory, no libraries.
4. **RISC-V**: simulated with Spike.
5. **Trade-offs**: accuracy against speed, across quantization, numerical approximations and compiler optimizations.

## Getting started

You need [Nix](https://nixos.org/download/) with flakes enabled. Everything else comes from the repository.

```bash
nix develop           # or: direnv allow, once per worktree
```

`flake.nix` is generated from `flake/` by [flake-file](https://github.com/vic/flake-file), so don't edit it by hand. Run `nix run .#write-flake` after changing inputs.

Entering the shell installs a pre-commit hook that formats Python, Nix and C.

## Data

Two dev-shell commands fetch the pinned files into `$TSPULSE_DATA`, or verify what is already there:

```bash
download-datasets     # TSB-AD-U and TSB-AD-M; --only u|m picks one
download-weights      # tspulse-r1
```

Neither overwrites a file that fails its check unless you pass `--force`. Pass `--output DIR` to use another folder.
