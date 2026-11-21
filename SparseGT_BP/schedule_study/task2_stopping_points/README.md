# Task 2: stopping points across the pipeline stages

`plot_stopping_points.py --traces <task1 or task5 output>` writes `STOPPING.md`,
`stop_runs_<code>.png` (per schedule and Eb/N0: fraction of frames that stopped in BP
run 1, 2, ..., converged to a wrong codeword, never converged; and the same only for
frames that reached the SGT) and `stop_iters_<code>.png` (histogram of the total BP
iterations at the stop, with the SGT stages marked). No GPU needed.
