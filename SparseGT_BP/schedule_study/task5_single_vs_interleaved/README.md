# Task 5: single-stage vs interleaved BP+SGT

`run_single_vs_interleaved.py` compares interleaved 5x10 / 5x40 (4 SGT stages) with
single-stage 2x25 / 2x100 (1 SGT stage) at the same total BP iterations, and plain
BP(50) / BP(200). Single-stage is evaluated with the interleaved weights (free) and,
with `--train-minutes 80`, with a model trained as single-stage (same recipe and
config otherwise). `compare.py` writes `SINGLE_VS_INTERLEAVED.md`.
