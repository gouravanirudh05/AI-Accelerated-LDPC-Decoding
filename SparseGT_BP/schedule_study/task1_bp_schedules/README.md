# Task 1: BP iteration schedules in interleaved BP+SGT

`run_schedules.py` runs the final BP-SGT models (trained as 5x10) with other BP
schedules, inference only: 5x10, 5x5, 5x2 (same 4 SGT stages, fewer iterations per
run), 10x5 and 25x2 (same 50-iteration budget, more SGT stages), 5x40, plus BP(10),
BP(25), BP(50), BP(200) at the matching budgets. Any schedule via
`--schedules 5x10,20-10-10-5-5,...`. `plot_schedules.py` writes `SCHEDULES.md` and
`bler_<code>.png` (BLER ratios vs BP at the same budget and vs 5x10).
See `../README.md` for Kaggle and caveats.
