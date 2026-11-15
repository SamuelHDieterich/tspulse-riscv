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

## Data

Three dev-shell commands fetch the pinned files into `$TSPULSE_DATA`, or verify what is already there:

```bash
download-datasets     # TSB-AD-U and TSB-AD-M; --only u|m picks one
download-weights      # tspulse-r1
download-lists        # TSB-AD's Eval and Tuning lists, and IBM's per-series scores
```

None overwrites a file that fails its check unless you pass `--force`. Pass `--output DIR` to use another folder.

## Reference run

One command runs IBM's own TSPulse pipeline on one series, with any number of channels, and saves each head's score and the ensemble:

```bash
score-series    # writes build/reference/<series>.csv under the current folder
```

Pass `--series` to pick another CSV, such as `TSB-AD-M/<file>.csv` or an absolute path, and `--output` to use another folder.
Time and memory grow with channels, so the widest TSB-AD-M files may not fit in RAM.
Run it from the dev shell, which provides the system libraries the wheels need.

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

### Benchmark

`benchmark` runs a build over the benchmark's windows and writes scored tables, inside the dev shell:

```bash
benchmark run tspulse/build/host tspulse/build/rv64 [--label NAME]  # runs the units; the same command resumes
benchmark collect --label NAME [--against OTHER]                    # scores what is done, writes the tables
benchmark all tspulse/build/host tspulse/build/rv64 [--label NAME]  # run, then collect
```

`run` has two stages:

- **instructions** counts a few windows per series under Spike, and checks each against the host build byte for byte;
- **accuracy** runs every window of the chosen `--scope`s on the host build: `screen` (each univariate working series' screen span) and `whole-python` (the 35 series the Python baseline ran whole), both by default.

With `benchmark run` and `all`, `--label` defaults to the build's name, and each tool refuses a binary that doesn't match its folder's `build.json`.
A label is one build and its settings. Running it again with other binaries or settings is refused: use another label, or `--redo` to start it over.

- `--jobs N` and `--spike-jobs N` run that many units at once on the host and under Spike (default 1).
- `--series STEM` keeps only the named series, and `--window-cap N` the first N windows of each scope.
- `--shard K/N` runs one computer's share, for N computers sharing the data folder. Their builds must match: compare `sha256sum tspulse/build/host/tspulse tspulse/build/rv64/tspulse "$RISCV_PK"`.
- `--isa`, `--spike-option`, `--accuracy-on spike` and `--host-check off` are for builds the host can't match, such as vector ones.

Follow a run with `tail -f "$TSPULSE_DATA/results/benchmark/<label>/<machine>.log"`.
Results go to `$TSPULSE_DATA/results/benchmark/<label>/`, and `collect` writes the tracked tables to `benchmark/results/<label>/`:

- `accuracy.csv`: VUS-PR, VUS-ROC and AUC-PR per series, scope and head, beside the Python baseline's and IBM's values;
- `instructions.csv`: the instructions of each counted window.

`collect` can run during a run: it scores the series that are complete and lists the rest. `--against OTHER` adds each row's difference from another label.
Each command exits 1 when a unit failed or differed or a series could not be scored. `collect` and `all` also exit 1 when a saved `result.json` can't be read or came from other binaries. Each of these is listed as a `problem:` line.

### Profiling a build

`benchmark profile` shows where a build's instructions go, how many cycles a window costs on a simple model of a RISC-V core, and what the program keeps in memory. It runs inside the dev shell, on the rv64 build folder (default `tspulse/build/rv64`, which needs both `tspulse` and `tspulse-profile`):

```bash
benchmark profile --label NAME [--windows 1,429,908] [--jobs N] [--work DIR]
```

`--label` is required, and the tool refuses a binary that doesn't match its build folder's `build.json`, which it copies beside the tables.
Each window M takes three Spike runs, which are the slow part: the clean build over `M-1:M+1` and over `M-1:M` with the instruction histogram and the cache counters, and the profile build over `M-1:M+1`. The window's numbers are the difference of the first two.
The runs go to `$TSPULSE_DATA/results/benchmark/<label>/profile/` (or `--work`), one folder per run with Spike's raw output, the exact command (run it from inside that folder) and its wall seconds. A run already done is skipped, so the same command resumes, and `--jobs` stays at 1 unless the computer has room for several Spikes.
The default windows are 1, 429 and 908 of the `560_YAHOO_id_10_Synthetic_tr_500_1st_893` series (`--series` changes it). Window 0 can't be profiled, since it has no window before it to subtract, and a vector build is refused: only scalar rv64 builds have a weight table.

The tables go to `benchmark/results/<label>/profile/`, and resuming writes the same bytes again. `kernels.csv`, `modules.csv`, `mix.csv`, `cycles.csv` and `traffic.csv` have rows for each window plus a `mean` row, `traffic.csv` adds a `floor_bytes` row with the window `all`, and `memory.csv` is one table for the build, with no window column. Fresh Spike runs can move the traffic counters slightly, because the cache model replaces lines at random:

- `kernels.csv`: instructions per function (its own instructions only), with the proxy kernel's own work as the `pk` row, which can be slightly negative since its handlers vary a little between runs;
- `modules.csv`: instructions per PyTorch module, children included, with the module's own share in `self`. A path is the copy (`hidden`, the twelve hidden copies summed, or `plain`), a `/`, then the module's dotted names; `depth` counts the copy as 1. The `window` row is the whole window with the profiling hooks' cost removed, and `raw` the same with nothing removed (its `calls` are the hook calls);
- `mix.csv`: instructions per mnemonic, compressed forms apart (`c.addi` and `addi`), without pk;
- `cycles.csv`: cycles per window as a range, and per device class its clock, seconds per window and the readings per second one channel keeps up with. The low end charges each instruction its issue cost, the high end its longest latency. The weights come from one manual, the OpenC906's (`score/profile/c906.csv`), so these are a model for comparing builds and not a measurement. The microcontroller rows carry a note that their times are a rough proxy for an rv32 chip;
- `memory.csv`: weights, code, static buffers, the stack's worst path and the heap as a formula of the channel count;
- `traffic.csv`: cache misses and writebacks, and the bytes they move, per window for one fixed cache (a 32 KB 2-way instruction cache and a 64 KB 4-way data cache with 64-byte lines), plus the least a window must read, `floor_bytes`. Traffic is never turned into time.

The totals are checked against the program's own instruction count, and the command fails when they disagree.

### Host timing

`time-host` times any number of C builds against IBM's PyTorch pipeline on this computer, inside the dev shell, with the computer otherwise idle:

```bash
time-host BUILD [BUILD …] --label NAME    # each BUILD a host build folder, such as tspulse/build/host
```

Each build is timed under the name its `build.json` gives it, and a folder whose `tspulse` doesn't match its `build.json`, or whose name another folder already has, is refused before anything runs. `--label` is required and names the run: its results are kept apart from other labels', and a label that is a machine's own results folder is refused. The fast x86 build from [The program](#the-program) is an ordinary extra folder, `tspulse/build/host-fast`, and shows up as `baseline+custom`.

It times two series: `560_YAHOO_id_10_Synthetic_tr_500_1st_893` whole (1 channel, 909 windows) and the first 128 windows of `120_TAO_id_5_Environment_tr_500_1st_3` (3 channels). Both sides time model work only:

- **C**, each build given in two setups. `one-core` runs one process pinned to `--pin-cpu` (default 2), twice: on every window and on the first alone, so the time per window is (T_N − T_1) / (N − 1), without start-up. `whole-pc` splits the windows over `--parallel` processes (default 10), unpinned, timed from the first start to the last exit.
- **PyTorch**, in its own process per measurement, after one untimed warm-up call: `b1-t1` (1 window per model call, 1 thread, pinned), `b128-t1` (128 windows per call, 1 thread, pinned) and `b128-whole-pc` (128 per call, `--parallel` threads, unpinned). The time is what the pipeline spends in `_forward`, the model calls and the error maths; the whole call is saved beside it.

Each measurement runs `--repeats` times (default 3), all of repeat 1 first. Every unit saves a JSON as soon as it finishes, with its CPU time, peak memory, load average and commands, so running the same command again skips what is done. A unit made with another build or binaries, PyTorch versions, `--pin-cpu` or `--parallel` runs again, and `--redo` deletes the label's saved units for this machine and reruns everything. `--parallel` must not be more than a series' windows.

Follow a run with `tail -f "$TSPULSE_DATA/results/host-timing/<label>/<machine>.log"`, where `--machine` defaults to the hostname.
Results go to `$TSPULSE_DATA/results/host-timing/<label>/<machine>/`: one JSON per unit, `runs.csv` with one row per unit (for one-core C, its `cpu_seconds` is the whole run on every window, start-up included, while `timed_seconds` is T_N − T_1), and `summary.csv` with the median over repeats per series, build and setup. Both tables have a `build` column with the build's name, or `torch`, and the columns `build_commit`, `build_tree` and `build_dirty` from its `build.json`, empty for PyTorch: the commit the binary was built from, and the tree, which still names that code after a history rewrite. The summary gives seconds per window and per channel-window, Hz per channel (channel-windows per second, the readings per second one channel can take), windows per second, and each row's seconds per channel-window divided by PyTorch `b1-t1`'s and `b128-whole-pc`'s, so above 1 is slower. At the end the summary is copied to the tracked `benchmark/results/host-timing/<label>/<machine>.csv`, which holds the builds of that call only.
A failed unit is listed as a `problem:` line, saves nothing, and makes the command exit 1 once the other units are done.

### Running variants

`benchmark variants` builds each variant of a list, runs it, and measures how it differs from the baseline, inside the dev shell:

```bash
benchmark variants [--only NAME] [--redo NAME] [--jobs N] [--spike-jobs N] [--suffix TEXT] [--table-only]   # the common options
```

It needs the baseline's results, in `$TSPULSE_DATA/results/benchmark/baseline/` and `benchmark/results/baseline/`.

The list is `benchmark/variants.toml`, or the file given to `--list`. It holds one `[[variant]]` table per variant, whose `settings` are the same Make settings `build-variant` takes, with string values (`build-variant --list` shows them):

```toml
[[variant]]
settings = { LINEAR = "blocked", UNROLL = "on" }
```

The variant's name is its build name, here `linear-blocked+unroll-on`. The baseline is never listed. A list is refused when it has no variants, an unknown setting or key, a value that is not a string, two variants with the same settings, settings that are all defaults, or two entries that build under the same name.

- `--only NAME` runs only that variant, and `--redo NAME` deletes that variant's results and starts it again. Both take the build name and can be repeated.
- `--jobs N` runs that many accuracy units at once on the host, and `--spike-jobs N` that many Spike runs at once, for the counted windows and the profile (default 1 each).
- `--series STEM` keeps only that working series, and can be repeated. `--window-cap N` keeps the first N windows of each series' screen span. They can be combined, and both need `--suffix`.
- `--suffix TEXT` is appended to each variant's name to make its label, so a cut-down run never writes into a full run's results. A label is the name plus the suffix, and every path below uses it. A label started with other series or another window cap is refused: use another `--suffix`, or `--redo`.
- `--machine NAME` names this computer in each result and in the log, and defaults to the hostname. `--data` defaults to `$TSPULSE_DATA`.
- `--table-only` builds and runs nothing and only writes the merged table from what is on disk.

Each variant goes through these stages, one variant after the other:

- **build**: the host and rv64 builds;
- **reference check**: the host build on the reference windows, as in [Checking the C against PyTorch](#checking-the-c-against-pytorch). A window that differs does not stop the variant: it is counted in the summary;
- **run**: `benchmark run` with the `screen` scope on the host and the `middle` counted windows under Spike, the same settings for every variant;
- **collect**: `benchmark collect --against baseline`;
- **profile**: `benchmark profile` on its default windows;
- **summary**: compares the variant's per-window errors with the baseline's, text for text, and writes `summary.csv`.

A stage that fails is logged as a `problem:` line, the variant's later stages are skipped, and the next variant starts.
A variant whose binaries differ from the ones its label first ran is also a problem, and `--redo` starts it again.
A `collect` that finds problems still writes its tables, and the profile and summary still run. Exactness problems are listed in the summary's `status`. Both make the command exit 1.

Each variant's results go to four folders, `<label>` being its label:

- `$TSPULSE_DATA/results/benchmark/<label>/`: what `benchmark run` writes, plus `reference.json`, `selection.json` (the series and window cap the label was started with), `collect.json` (how many problems the last collect found) and the log `<machine>.log`. The command's own lines in the log (a stage's start, end, skip and problems) begin with `variants:`, the stages' lines do not;
- `$TSPULSE_DATA/results/benchmark/<label>/profile/`: the profile's raw Spike runs;
- `benchmark/results/<label>/`: `accuracy.csv` and `instructions.csv` from `collect`, and `summary.csv`, the variant's one row;
- `benchmark/results/<label>/profile/`: the profile tables.

At the end of every call, also after Ctrl-C, the summaries are merged into `benchmark/results/variants<suffix>.csv`, and its short form is printed: each variant's name, verdict (or status when it has no summary), instruction ratio, ensemble VUS-PR and its difference from the baseline, and cycle range.
The first row is the baseline, whose row is computed from its tracked tables, and the others follow the list's order. A variant with no `summary.csv` has only its `name`, `settings` and a `status` of `not run`.
`--table-only` writes that table again at any time.

The same command resumes: a stage whose results are already on disk is skipped, and the log says so.

- `run` is skipped when every unit of the label's `selection.json` has a result.
- `collect` is skipped when `run` was, both tables exist, and `collect.json` holds the label's current binaries and records no problem. A missing record, such as after an interrupted collect, or one with problems runs it again, so a variant with a collect problem exits 1 on every call until a collect finds none.
- The reference check and the profile are skipped while the binaries they were made with are the ones the label has.
- The summary is written again on every call that reaches that stage.

The command exits 1 when any variant had a problem, and 3 when the list, a label's selection, or the data folder can't be used, which is checked before any variant builds. When the merge at the end fails, it exits 1 too, also with `--table-only`. After Ctrl-C or SIGTERM it stops the running stage and exits 130, and the rest waits for the next call.

#### Columns

`summary.csv` and the merged table have these columns:

- `name` and `settings`: the variant's build name and its settings as `A=b B=c`, in the list's order. The baseline's `settings` is empty;
- `build_commit`, `build_tree` and `build_dirty`: the commit the build was made from, its tree, and whether the tree was dirty, from the profile's `build.json`. Empty for the baseline;
- `status`: `done` when the label holds every result, else what it lacks, joined by `; `: `reference check`, `run`, `collect`, `collect found N problems`, `profile`, `N series left out`, `N exactness problems` or `unreadable files`. A variant that has not been run shows `not run`. Empty for the baseline;
- `verdict`: `identical` when at least one window was compared, none differs, no series was left out and the comparison had no problem. Otherwise `within-tolerance` when every reference window passed, and `beyond-tolerance` when one did not. The baseline's is `reference`;
- `windows_compared`: the windows whose error row both the variant and the baseline hold, over the variant's selected series. `windows_differing`: those whose row differs as text. Series with missing units are left out of both and counted in `status`;
- `reference_pass` and `reference_fail`: how many of the 7 reference windows passed and failed;
- `instructions_u_mean` and `instructions_m_mean`: the mean of `per_channel` in `instructions.csv` over the univariate and the multivariate counted windows. The baseline's means cover only the windows the variants count;
- `instructions_ratio`: the mean of the `ratio` column, the variant's instructions over the baseline's for the same window, over the univariate windows;
- `cycles_low` and `cycles_high`: the cycle range of the mean window in `profile/cycles.csv`, over all classes;
- `weights_bytes`, `text_bytes`, `static_bytes` and `stack_bytes`: the weights, the `.text` section, the sum of the static buffers and the stack's worst path, from `profile/memory.csv`;
- `heap_bytes_per_channel`: the sum of the heap formulas of `profile/memory.csv` at one channel;
- `traffic_bytes`: `total_bytes` of the mean window in `profile/traffic.csv`;
- `vus_pr_<head>`, for the heads `time`, `fft`, `forecast` and `ensemble`: the mean VUS-PR over the series, in the `screen` scope, of `accuracy.csv`. `vus_pr_<head>_diff` is the mean of the `difference` from the baseline, and `vus_pr_<head>_worst` the most negative one, the series that got worst. The baseline has no difference cells.

A cell is empty when it has nothing to hold: the baseline's cells that compare it with itself, or an average over no rows.

## Benchmark series

`benchmark/series.csv` fixes which TSB-AD series the benchmark uses. It is generated by `select-series` and tracked:

```bash
select-series    # needs download-datasets and download-lists first
```

`working` series are for choosing variants, and `held` series stay untouched until the final runs.
`U` working series also have a span of rows for the quick screen, in the `screen_*` columns.

## Spike cost estimate

One command runs the matrix multiplies of one TSPulse pass under Spike, and projects Spike core-hours for each series group:

```bash
spike-matmul    # writes build/matmul/<machine>/ under the current folder
```

It builds four setups, `rv64gc` and `rv64gcv` at `-O2` and `-O3`, counts instructions with `rdinstret`, and checks each build against a host build.
The projection is a floor, since it leaves out GELU, softmax, LayerNorm and the FFT.
A rerun skips setups that already finished, unless you pass `--force`.

Instruction counts are the same on every machine, but Spike's speed is not. `--machine NAME` labels a computer's results, so several computers can share one folder. It defaults to the hostname.

## C port

The C code lives in `tspulse/`, one folder per role:

- `kernels/` holds the math the modules share, one file per kernel: `linear`, `layer_norm`, `gelu`, `softmax`, `revin`, the radix-2 `fft`, the pairwise `sum`, and the small element-wise ones.
- `modules/` has one function per PyTorch module, grouped by family: `mixer`, `block`, `embedding`, `backbone`, `decoder` and `heads`.
- `model/` runs the model: `pass` is one forward pass of one channel, `window` runs the 13 passes of every channel and stitches the errors, `history` keeps the last 513 rows, and `config` and `sizes` hold the constants from `config.json` and the model's sizes.
- `weights/` has the struct that mirrors the PyTorch module tree, and the table of names the checks walk.
- `io/` has the row file reader and the `.tdump` writer.
- `checks/` has programs that only verify.
- `build/` holds objects and executables per target, and the generated weights both targets share.

The weights are compiled in. `export-weights` reads `model.safetensors` and writes the C that fills `struct tspulse_weights`, plus a table for the checks and a `.tdump` of the same values. It leaves out the frequency signature head, which the baseline never runs, so the program carries 226 tensors and 4,045,272 bytes of read-only data.

### The program

`make` builds the program in three forms from the same sources:

- `tspulse` is the one to measure. It carries no dump code.
- `tspulse-dump` is built with the macro `TSPULSE_DUMP`, which turns on `--dump-window` and `--dump`: they write every module's output for one window to a `.tdump`. The two write bit-identical errors.
- `tspulse-profile`, built only by `make TARGET=rv64`, is built with the macro `TSPULSE_PROFILE`, which turns on `--profile PATH`: it writes the instructions each module took in each window to a CSV. The three write bit-identical errors.

`make TARGET=rv64` also compiles the program's objects once more with GCC's call-graph output, in `build/rv64/stack/`, which `benchmark profile` reads for the stack's frame sizes.

For x86 timing only, `NIX_ENFORCE_NO_NATIVE=0 make -C tspulse BUILD=build/host-fast ARCH_FLAGS="-O3 -march=native"` builds a vectorized copy in `tspulse/build/host-fast`. It is not bit-identical to rv64, and without `NIX_ENFORCE_NO_NATIVE=0` the dev shell's compiler silently drops `-march=native`.

```bash
tspulse ROWS ERRORS [--windows START:STOP]
```

`ROWS` is a row file. The program reads it one row at a time, keeps the last 513 rows, and runs every window in the range (window `w` covers rows `w` to `w + 512`; STOP is not run, and the default is every window). It truncates `ERRORS` and writes the header `window,time,fft,forecast` followed by one line per window, with the three errors that feed the scores. Channels run one at a time, so working memory does not grow with the channel count.

The clean build's buffers are static, except three allocated once at start: the 513-row history, one row, and each channel's average and spread. The dump build's writer also allocates.

The model itself is all float, with no double. Double appears only in standardizing the history, which scikit-learn also does in double, and in building the FFT's twiddle table once at start.

### Variants

A setting picks alternative C code or compiler flags for a build, as a Make variable:

```bash
build-variant --list              # prints every setting with its values and its default
build-variant [SETTING=value …]   # builds host and rv64 with those settings, and prints the two build folders
```

Pass `-j` through `MAKEFLAGS`, as in `MAKEFLAGS=-j8 build-variant`. A build that sets nothing is the baseline, in `tspulse/build/host` and `tspulse/build/rv64`. Any other build lands in `tspulse/build/<name>/host` and `.../rv64`, where the name lists the settings that differ from their default, as in `linear-blocked+unroll-on`. A build that sets `CC` or `ARCH_FLAGS` on the command line gets `+custom` added to its name.

Each build folder holds a `build.json` with the build's name, settings, compiler, flags, source files, commit, whether the tree was dirty, and the hash of each binary. `benchmark run`, `benchmark profile` and `time-host` refuse a binary whose hash differs from its `build.json`, and say to rebuild.

### Row files

Python only turns a CSV into a binary row file, which the C program reads. The format is specified in `scripts/score/src/score/rows.py`: the raw float64 values, row by row, with the average and spread of each channel over the whole file. The program standardizes each window itself.

```bash
prepare-rows CSV OUT         # CSV is relative to $TSPULSE_DATA/datasets, or absolute
prepare-rows --flat OUT      # the flat window: 513 zeros, average 0, spread 1
```

### Checking the C against PyTorch

```bash
check-model build/host/tspulse-dump
check-model build/host/tspulse-dump --series <name> --clean build/host/tspulse
```

The first form runs the dump build on the 7 reference windows and compares every record with the reference under the noise rule of `compare-reference`. It exits 1 on any failing record, any record the reference has and the program lacks without being on the list of what the baseline skips, and any extra record. `--only PREFIX` limits it to records whose names start with it.

The 7 windows are the precise check: every inner step and each window's `error.*` values are held to PyTorch's own float32 noise, which catches a ×1.00001 fault in one layer.

With `--series`, it then checks detection on a whole series. The clean build runs over every window, and its three errors per window go through the anomaly pipeline's own post-processing (`score/rescore.py`) into per-row scores and the ensemble, exactly as the Python baseline's were. Each of the four scores then gets VUS-PR, VUS-ROC and AUC-PR from the same code as the baseline's stored scores in `$TSPULSE_DATA/results/python-baseline/<name>.npz`.

The series passes when every score's VUS-PR is within 1e-3 of the baseline's. The tolerance is a stated choice, below the third decimal VUS-PR results are reported at, not a measured noise bound. It fails on any NaN or Inf error from the C. It prints a table per score with both VUS-PR values, their difference and the verdict, plus the VUS-ROC, AUC-PR and largest per-row score differences for information, and the times of the C run and of the scoring.
Because it compares detection, it is not expected to catch tiny numeric faults, or a constant scale or offset on one head, which the pipeline's per-head normalization hides. Those are the 7-window stage's job.

From `tspulse/`, inside the dev shell and after `download-weights`:

```bash
make                    # builds check-weights, tspulse and tspulse-dump for the host
make check              # runs the weight check and compares its dump with the one `export-weights` wrote from the safetensors file, bit for bit
make TARGET=rv64        # the same programs for RISC-V
make TARGET=rv64 check  # the weight check under Spike
make clean
```

The generated files are rebuilt only when the safetensors file or `export-weights` changes (after re-entering the dev shell). If you delete one by hand, run `make clean`.

### Running under Spike

After `make TARGET=rv64`, the plain command runs a window range on the RISC-V build, from `tspulse/` inside the dev shell:

```bash
spike --isa=rv64gc_zicntr "$RISCV_PK" build/rv64/tspulse ROWS ERRORS --windows START:STOP
```

`$RISCV_PK` is the proxy kernel (pk) the dev shell provides, since Spike does not search `PATH` for it.
`--counts PATH` writes each window's instruction count to `PATH`. Only the RISC-V build has it.
The rv64 build carries its own `_open`, because pk reads the open flags as Linux numbers and newlib's differ.

Spike is slow: with `-g`, a window takes roughly a minute per channel, so a 55-channel window takes close to an hour.
The instruction count does not depend on the computer, and is the number to compare.
One window costs about 2.7 billion program instructions per channel with the dump build (`tspulse-dump`) and about 2.4 billion with the clean build (`tspulse`), whatever the series.

`check-spike` checks that the RISC-V build computes what the host build does, byte for byte:

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

A rerun resumes: a unit already run with the same four binaries is printed and skipped, and one run with different binaries reruns.
`--redo` forces every selected unit to rerun.
Interrupting with Ctrl-C or SIGTERM stops every Spike started, and running the same command again picks up the rest.
It exits 1 when something needed is missing, any program fails, or any unit the call selected differs, including a skipped one whose stored result differs. It exits 0 otherwise.
