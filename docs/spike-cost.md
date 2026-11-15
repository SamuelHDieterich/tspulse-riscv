# Spike cost estimate

One command runs the matrix multiplies of one TSPulse pass under Spike, and projects Spike core-hours for each series group:

```bash
spike-matmul    # writes build/matmul/<machine>/ under the current folder
```

It builds four setups, `rv64gc` and `rv64gcv` at `-O2` and `-O3`, counts instructions with `rdinstret`, and checks each build against a host build.
The projection is a floor, since it leaves out GELU, softmax, LayerNorm and the FFT.
A rerun skips setups that already finished, unless you pass `--force`.

Instruction counts are the same on every machine, but Spike's speed is not. `--machine NAME` labels a computer's results, so several computers can share one folder. It defaults to the hostname.
