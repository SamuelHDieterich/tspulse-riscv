# Host timing

`time-host` times any number of C builds against IBM's PyTorch pipeline on this computer, inside the dev shell, with the computer otherwise idle:

```bash
time-host BUILD [BUILD …] --label NAME    # each BUILD a host build folder, such as tspulse/build/host
```

Each build is timed under the name its `build.json` gives it, and a folder whose `tspulse` doesn't match its `build.json`, or whose name another folder already has, is refused before anything runs. `--label` is required and names the run: its results are kept apart from other labels', and a label that is a machine's own results folder is refused. The fast x86 build from [The program](c-port.md#the-program) is an ordinary extra folder, `tspulse/build/host-fast`, and shows up as `baseline+custom`.

It times two series: `560_YAHOO_id_10_Synthetic_tr_500_1st_893` whole (1 channel, 909 windows) and the first 128 windows of `120_TAO_id_5_Environment_tr_500_1st_3` (3 channels). Both sides time model work only:

- **C**, each build given in two setups. `one-core` runs one process pinned to `--pin-cpu` (default 2), twice: on every window and on the first alone, so the time per window is (T_N − T_1) / (N − 1), without start-up. `whole-pc` splits the windows over `--parallel` processes (default 10), unpinned, timed from the first start to the last exit.
- **PyTorch**, in its own process per measurement, after one untimed warm-up call: `b1-t1` (1 window per model call, 1 thread, pinned), `b128-t1` (128 windows per call, 1 thread, pinned) and `b128-whole-pc` (128 per call, `--parallel` threads, unpinned). The time is what the pipeline spends in `_forward`, the model calls and the error maths; the whole call is saved beside it.

Each measurement runs `--repeats` times (default 3), all of repeat 1 first. Every unit saves a JSON as soon as it finishes, with its CPU time, peak memory, load average and commands, so running the same command again skips what is done. A unit made with another build or binaries, PyTorch versions, `--pin-cpu` or `--parallel` runs again, and `--redo` deletes the label's saved units for this machine and reruns everything. `--parallel` must not be more than a series' windows.

Follow a run with `tail -f "$TSPULSE_DATA/results/host-timing/<label>/<machine>.log"`, where `--machine` defaults to the hostname.
Results go to `$TSPULSE_DATA/results/host-timing/<label>/<machine>/`: one JSON per unit, `runs.csv` with one row per unit (for one-core C, its `cpu_seconds` is the whole run on every window, start-up included, while `timed_seconds` is T_N − T_1), and `summary.csv` with the median over repeats per series, build and setup. Both tables have a `build` column with the build's name, or `torch`, and the columns `build_commit`, `build_tree` and `build_dirty` from its `build.json`, empty for PyTorch: the commit the binary was built from, and the tree, which still names that code after a history rewrite. The summary gives seconds per window and per channel-window, Hz per channel (channel-windows per second, the readings per second one channel can take), windows per second, and each row's seconds per channel-window divided by PyTorch `b1-t1`'s and `b128-whole-pc`'s, so above 1 is slower. At the end the summary is copied to the tracked `benchmark/results/host-timing/<label>/<machine>.csv`, which holds the builds of that call only.
A failed unit is listed as a `problem:` line, saves nothing, and makes the command exit 1 once the other units are done.
