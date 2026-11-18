# C port

The C code lives in `tspulse/`, one folder per role:

- `kernels/` holds the math the modules share, one file per kernel: `linear`, `layer_norm`, `gelu`, `softmax`, `revin`, the radix-2 `fft`, the pairwise `sum`, and the small element-wise ones.
- `modules/` has one function per PyTorch module, grouped by family: `mixer`, `block`, `embedding`, `backbone`, `decoder` and `heads`.
- `model/` runs the model: `pass` is one forward pass of one channel, `window` runs the 13 passes of every channel and stitches the errors, `history` keeps the last 513 rows, and `config` and `sizes` hold the constants from `config.json` and the model's sizes.
- `weights/` has the struct that mirrors the PyTorch module tree, and the table of names the checks walk.
- `io/` has the row file reader and the `.tdump` writer.
- `checks/` has programs that only verify.
- `build/` holds objects and executables per target, and the generated weights both targets share.

The weights are compiled in. `export-weights` reads `model.safetensors` and writes the C that fills `struct tspulse_weights`, plus a table for the checks and a `.tdump` of the same values. It leaves out the frequency signature head, which the baseline never runs, so the program carries 226 tensors and 4,045,272 bytes of read-only data.

## Building

From `tspulse/`, inside the dev shell and after `download-weights`:

```bash
make                    # builds check-weights, tspulse and tspulse-dump for the host
make check              # runs the weight check and compares its dump with the one `export-weights` wrote from the safetensors file, bit for bit
make approx-errors      # host only: writes each exp and GELU approximation's error against double precision to `build/host/approx-errors.csv`
make TARGET=rv64        # the same programs for RISC-V
make TARGET=rv64 check  # the weight check under Spike
make clean
```

The generated files are rebuilt only when the safetensors file or `export-weights` changes (after re-entering the dev shell). If you delete one by hand, run `make clean`.

## The program

`make` builds the program in three forms from the same sources:

- `tspulse` is the one to measure. It carries no dump code.
- `tspulse-dump` is built with the macro `TSPULSE_DUMP`, which turns on `--dump-window` and `--dump`: they write every module's output for one window to a `.tdump`. The two write bit-identical errors.
- `tspulse-profile`, built only by `make TARGET=rv64`, is built with the macro `TSPULSE_PROFILE`, which turns on `--profile PATH`: it writes the instructions each module took in each window to a CSV. The three write bit-identical errors.
- With `OUTPUTS=read`, the dump build still computes every row, so `check-model` checks the model but not the code that skips rows, which the variant's errors comparison covers.
- Under `AGGREGATION`, `check-model` reports the `hiddenNN/` records of the passes that did not run as missing.

`make TARGET=rv64` also compiles the program's objects once more with GCC's call-graph output, in `build/rv64/stack/`, which `benchmark profile` reads for the stack's frame sizes.

For x86 timing only, `NIX_ENFORCE_NO_NATIVE=0 make -C tspulse BUILD=build/host-fast ARCH_FLAGS="-O3 -march=native"` builds a vectorized copy in `tspulse/build/host-fast`. It is not bit-identical to rv64, and without `NIX_ENFORCE_NO_NATIVE=0` the dev shell's compiler silently drops `-march=native`.

```bash
tspulse ROWS ERRORS [--windows START:STOP]
```

`ROWS` is a row file. The program reads it one row at a time, keeps the last 513 rows, and runs every window in the range (window `w` covers rows `w` to `w + 512`; STOP is not run, and the default is every window). It truncates `ERRORS` and writes the header `window,time,fft,forecast` followed by one line per window, with the three errors that feed the scores. Channels run one at a time, so working memory does not grow with the channel count.

The clean build's buffers are static, except three allocated once at start: the 513-row history, one row, and each channel's average and spread. The dump build's writer also allocates.

## Variants

A setting picks alternative C code or compiler flags for a build, as a Make variable:

```bash
build-variant --list              # prints every setting with its values and its default
build-variant [SETTING=value …]   # builds host and rv64 with those settings, and prints the two build folders
```

Pass `-j` through `MAKEFLAGS`, as in `MAKEFLAGS=-j8 build-variant`. A build that sets nothing is the baseline, in `tspulse/build/host` and `tspulse/build/rv64`. Any other build lands in `tspulse/build/<name>/host` and `.../rv64`, where the name lists the settings that differ from their default, as in `linear-blocked+unroll-on`. A build that sets `CC` or `ARCH_FLAGS` on the command line gets `+custom` added to its name.

Each build folder holds a `build.json` with the build's name, settings, compiler, flags, source files, commit, whether the tree was dirty, and the hash of each binary. `benchmark run`, `benchmark profile` and `time-host` refuse a binary whose hash differs from its `build.json`, and say to rebuild.

## Running under Spike

After `make TARGET=rv64`, the plain command runs a window range on the RISC-V build, from `tspulse/` inside the dev shell:

```bash
spike --isa=rv64gc_zicntr "$RISCV_PK" build/rv64/tspulse ROWS ERRORS --windows START:STOP
```

`$RISCV_PK` is the proxy kernel (pk) the dev shell provides, since Spike does not search `PATH` for it.
`--counts PATH` writes each window's instruction count to `PATH`. Only the RISC-V build has it.

Spike is slow: with `-g`, a window takes roughly a minute per channel, so a 55-channel window takes close to an hour.
The instruction count does not depend on the computer, and is the number to compare.
One window costs about 2.7 billion program instructions per channel with the dump build (`tspulse-dump`) and about 2.4 billion with the clean build (`tspulse`), whatever the series.
