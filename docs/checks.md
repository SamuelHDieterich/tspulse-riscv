# Checks

## Reference run

One command runs IBM's own TSPulse pipeline on one series, with any number of channels, and saves each head's score and the ensemble:

```bash
score-series    # writes build/reference/<series>.csv under the current folder
```

Pass `--series` to pick another CSV, such as `TSB-AD-M/<file>.csv` or an absolute path, and `--output` to use another folder.
Time and memory grow with channels, so the widest TSB-AD-M files may not fit in RAM.
Run it from the dev shell, which provides the system libraries the wheels need.

## Python baseline

`score-baseline` runs the same pipeline over the benchmark's working series, with every head and IBM's metric, and saves each series' scores. `check-model --series` compares the C against these scores.

```bash
score-baseline --kind U        # or --series NAME, repeatable; writes $TSPULSE_DATA/results/python-baseline/
score-baseline --collect       # builds the tracked tables in benchmark/baseline/ from the saved results
```

Each series is saved as a `.npz` and a `.json`. A series already scored is skipped, so the same command resumes. `--jobs N` scores that many series at once (default 8), each in its own process.

## Reference outputs

The C port is checked step by step against PyTorch's output of every module, saved on 7 fixed windows:

```bash
capture-reference    # writes $TSPULSE_DATA/reference/, about 0.65 GB
compare-reference $TSPULSE_DATA/reference/<series>/w<index>.tdump <dump>.tdump
```

`capture-reference` skips windows already on disk, unless you pass `--force`.
`compare-reference` lists every record, marks the first that differs by more than PyTorch's own float32 error, and exits 1 if any does.
`--exact` requires identical bits instead, for comparing two C builds, and `--only PREFIX` limits the comparison to records whose names start with it.
The `.tdump` format and its record names are specified in `scripts/score/src/score/tdump.py`.

## Checking the C against PyTorch

From `tspulse/`, inside the dev shell, after `make` and `capture-reference`:

```bash
check-model build/host/tspulse-dump
check-model build/host/tspulse-dump --series <name> --clean build/host/tspulse
```

The first form runs the dump build on the 7 reference windows and compares every record with the reference under the noise rule of `compare-reference`. It exits 1 on any failing record, any record the reference has and the program lacks without being on the list of what the baseline skips, and any extra record. `--only PREFIX` limits it to records whose names start with it.

The 7 windows are the precise check: every inner step and each window's `error.*` values are held to PyTorch's own float32 noise, which catches a ×1.00001 fault in one layer.

With `--series`, which needs that series scored by `score-baseline` first, it then checks detection on a whole series. The clean build runs over every window, and its three errors per window go through the anomaly pipeline's own post-processing (`score/rescore.py`) into per-row scores and the ensemble, exactly as the Python baseline's were. Each of the four scores then gets VUS-PR, VUS-ROC and AUC-PR from the same code as the baseline's stored scores in `$TSPULSE_DATA/results/python-baseline/<name>.npz`.

The series passes when every score's VUS-PR is within 1e-3 of the baseline's. It fails on any NaN or Inf error from the C. It prints a table per score with both VUS-PR values, their difference and the verdict, plus the VUS-ROC, AUC-PR and largest per-row score differences for information, and the times of the C run and of the scoring.
Because it compares detection, it is not expected to catch tiny numeric faults, or a constant scale or offset on one head, which the pipeline's per-head normalization hides. Those are the 7-window stage's job.

## Checking the RISC-V build

`check-spike` checks that the RISC-V build computes what the host build does, byte for byte, from `tspulse/` after `make` and `make TARGET=rv64`:

```bash
check-spike build/host build/rv64
check-spike build/host build/rv64 --stage reference --jobs 1
```

A unit is one window, run on the host and then under Spike.
The reference stage runs the dump build on each of the 7 reference windows and compares every inner output. The series stage runs the clean build on the middle window of each working series and compares its three errors.
`--stage reference|series|all` picks the stages (default `all`).
`--reference LABEL` (such as `flat/w0`) and `--series NAME` (a CSV name without `.csv`) can be repeated, and each keeps only the named units of its own stage. The other stage still runs in full unless `--stage` leaves it out: `--stage reference --reference flat/w0` runs just that one unit.
`--jobs N` runs that many units at once, one Spike each, which is limited by the cores and the memory of the computer. It defaults to 1.
`--work DIR` moves the row files and the scratch errors and dumps, which default to `tspulse/build/check-spike` in the repository.

Each unit writes its result under `$TSPULSE_DATA/results/spike-check/<machine>/<unit>/`, and the command rewrites a tidy `results.csv` with one row per unit in `<machine>/` as units complete.
`--machine NAME` labels the computer, so several can share the data folder. It defaults to the hostname.
`--isa STRING` is the ISA Spike runs with, `rv64gc_zicntr` by default. For an rv32 build, run it with the rv32 proxy kernel and a machine name of its own, so the rv64 results stay: `RISCV_PK="$RISCV32_PK" check-spike build/host build/rv32 --isa rv32gc_zicntr --machine NAME`.

A rerun resumes: a unit already run with the same four binaries is printed and skipped, and one run with different binaries reruns.
`--redo` forces every selected unit to rerun.
Interrupting with Ctrl-C or SIGTERM stops every Spike started, and running the same command again picks up the rest.
It exits 1 when something needed is missing, any program fails, or any unit the call selected differs, including a skipped one whose stored result differs. It exits 0 otherwise.
