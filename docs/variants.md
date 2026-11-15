# Running variants

`benchmark variants` builds each variant of a list, runs it, and measures how it differs from the baseline, inside the dev shell:

```bash
benchmark variants [--only NAME] [--redo NAME] [--jobs N] [--spike-jobs N] [--suffix TEXT] [--table-only]   # the common options
```

It needs the baseline's results, in `$TSPULSE_DATA/results/benchmark/baseline/` and `benchmark/results/baseline/`. `benchmark all tspulse/build/host tspulse/build/rv64` makes them, since a build's label defaults to its name, `baseline`.

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
- **reference check**: the host build on the reference windows, as in [Checking the C against PyTorch](checks.md#checking-the-c-against-pytorch). A window that differs does not stop the variant: it is counted in the summary;
- **run**: `benchmark run` with the `screen` scope on the host and the `middle` counted windows under Spike, the same settings for every variant;
- **collect**: `benchmark collect --against baseline`;
- **profile**: `benchmark profile` on its default windows;
- **summary**: compares the variant's per-window errors with the baseline's, text for text, and writes `summary.json`.

A stage that fails is logged as a `problem:` line, the variant's later stages are skipped, and the next variant starts.
A variant whose binaries differ from the ones its label first ran is also a problem, and `--redo` starts it again.
A `collect` that finds problems still writes its tables, and the profile and summary still run. Exactness problems are listed in the summary's `missing`. Both make the command exit 1.

Each variant's results go to four folders, `<label>` being its label:

- `$TSPULSE_DATA/results/benchmark/<label>/`: what `benchmark run` writes, plus `reference.json`, `selection.json` (the series and window cap the label was started with), `collect.json` (how many problems the last collect found) and the log `<machine>.log`. The command's own lines in the log (a stage's start, end, skip and problems) begin with `variants:`, the stages' lines do not;
- `$TSPULSE_DATA/results/benchmark/<label>/profile/`: the profile's raw Spike runs;
- `benchmark/results/<label>/`: `accuracy.csv` and `instructions.csv` from `collect`, and `summary.json`, the variant's summary;
- `benchmark/results/<label>/profile/`: the profile tables.

At the end of every call, also after Ctrl-C, the summaries are merged into `benchmark/results/variants<suffix>.json`, and its short form is printed: each variant's name, verdict (or what it lacks when it has no summary), instruction ratio, ensemble VUS-PR and its difference from the baseline, and cycle range.
The first summary is the baseline's, computed from its tracked tables, and the others follow the list's order. A variant with no `summary.json` has only its `name`, `settings` and a `missing` of `["not run"]`.
`--table-only` writes that file again at any time.

The same command resumes: a stage whose results are already on disk is skipped, and the log says so.

- `run` is skipped when every unit of the label's `selection.json` has a result.
- `collect` is skipped when `run` was, both tables exist, and `collect.json` holds the label's current binaries and records no problem. A missing record, such as after an interrupted collect, or one with problems runs it again, so a variant with a collect problem exits 1 on every call until a collect finds none.
- The reference check and the profile are skipped while the binaries they were made with are the ones the label has.
- The summary is written again on every call that reaches that stage.

The command exits 1 when any variant had a problem, and 3 when the list, a label's selection, or the data folder can't be used, which is checked before any variant builds. When the merge at the end fails, it exits 1 too, also with `--table-only`. After Ctrl-C or SIGTERM it stops the running stage and exits 130, and the rest waits for the next call.

## Summaries

`summary.json` holds one summary, and `variants.json` is a list of them: the baseline's, then each variant's in the list's order.
A field that is empty or has nothing to hold is left out, and so is an object or a list left with nothing in it, such as a VUS-PR head with no values. So the baseline has no `settings`, `build`, `missing`, `exactness` or `reference`, no `instructions.ratio`, and no `difference` or `worst` in its VUS-PR.

```json
{
  "name": "linear-blocked",
  "settings": {
    "LINEAR": "blocked"
  },
  "build": {
    "commit": "12fdf08cb1f9868ac3d2386876bb3770445131d6",
    "tree": "51ef8924932555620b4f5e186a8406571b5daafb",
    "dirty": false
  },
  "verdict": "identical",
  "exactness": {
    "windows_compared": 200641,
    "windows_differing": 0
  },
  "reference": {
    "pass": 7,
    "fail": 0
  },
  "instructions": {
    "u_mean": 1066404483.6458334,
    "m_mean": 1061516023.6633333,
    "ratio": 0.4390880980862562
  },
  "cycles": {
    "low": 1076114073,
    "high": 2631404295
  },
  "memory_bytes": {
    "weights": 4045272,
    "text": 60094,
    "static": 224112,
    "stack": 848,
    "heap_per_channel": 4128
  },
  "traffic_bytes": 352873600,
  "vus_pr": {
    "time": {
      "value": 0.46455526481416265,
      "difference": 0.0,
      "worst": 0.0
    },
    "fft": {
      "value": 0.45018826133056616,
      "difference": 0.0,
      "worst": 0.0
    },
    "forecast": {
      "value": 0.37188642149722634,
      "difference": 0.0,
      "worst": 0.0
    },
    "ensemble": {
      "value": 0.49783069296433946,
      "difference": 0.0,
      "worst": 0.0
    }
  }
}
```

- `name` and `settings`: the variant's build name and its settings, in the list's order;
- `build`: the `commit` the build was made from, its `tree`, and whether the tree was `dirty`, from the profile's `build.json`;
- `missing`: what the label lacks, among `reference check`, `run`, `collect`, `collect found N problems`, `profile`, `N series left out`, `N exactness problems` and `unreadable files`. A variant that has not been run has `["not run"]`. A summary without `missing` is done;
- `verdict`: `identical` when at least one window was compared, none differs, no series was left out and the comparison had no problem. Otherwise `within-tolerance` when every reference window passed, and `beyond-tolerance` when one did not. The baseline's is `reference`;
- `exactness`: `windows_compared`, the windows whose error row both the variant and the baseline hold, over the variant's selected series, and `windows_differing`, those whose row differs as text. Series with missing units are left out of both and counted in `missing`;
- `reference`: how many of the reference windows passed (`pass`) and failed (`fail`);
- `instructions`: `u_mean` and `m_mean`, the mean of `per_channel` in `instructions.csv` over the univariate and the multivariate counted windows (the baseline's means cover only the windows the variants count), and `ratio`, the mean of the `ratio` column, the variant's instructions over the baseline's for the same window, over the univariate windows;
- `cycles`: the `low` and `high` ends of the cycle range of the mean window in `profile/cycles.csv`, over all classes;
- `memory_bytes`: from `profile/memory.csv`, the `weights`, the `.text` section (`text`), the sum of the static buffers (`static`), the stack's worst path (`stack`) and the sum of the heap formulas at one channel (`heap_per_channel`);
- `traffic_bytes`: `total_bytes` of the mean window in `profile/traffic.csv`;
- `vus_pr`: for the heads `time`, `fft`, `forecast` and `ensemble`, the mean `value` of the VUS-PR over the series, in the `screen` scope, of `accuracy.csv`, the mean `difference` from the baseline and the most negative one, `worst`, the series that got worst.
