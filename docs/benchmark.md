# Benchmark

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

`collect` can run during a run: it scores the series that are complete and lists the rest. It scores each label with the aggregation length of its build (96 when the build sets none). `--against OTHER` adds each row's difference from another label.
Each command exits 1 when a unit failed or differed or a series could not be scored. `collect` and `all` also exit 1 when a saved `result.json` can't be read or came from other binaries. Each of these is listed as a `problem:` line.

## Profiling a build

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
- `modules.csv`: instructions per PyTorch module, children included, with the module's own share in `self`. A path is the copy (`hidden`, the hidden copies summed, or `plain`), a `/`, then the module's dotted names; `depth` counts the copy as 1. The `window` row is the whole window with the profiling hooks' cost removed, and `raw` the same with nothing removed (its `calls` are the hook calls);
- `mix.csv`: instructions per mnemonic, compressed forms apart (`c.addi` and `addi`), without pk;
- `cycles.csv`: cycles per window as a range, and per device class its clock, seconds per window and the readings per second one channel keeps up with. The low end charges each instruction its issue cost, the high end its longest latency. The weights come from one manual, the OpenC906's (`score/profile/c906.csv`), so these are a model for comparing builds and not a measurement. The microcontroller rows carry a note that their times are a rough proxy for an rv32 chip;
- `memory.csv`: weights, code, static buffers, the stack's worst path and the heap as a formula of the channel count;
- `traffic.csv`: cache misses and writebacks, and the bytes they move, per window for one fixed cache (a 32 KB 2-way instruction cache and a 64 KB 4-way data cache with 64-byte lines), plus the least a window must read, `floor_bytes`. Traffic is never turned into time.

The totals are checked against the program's own instruction count, and the command fails when they disagree.
