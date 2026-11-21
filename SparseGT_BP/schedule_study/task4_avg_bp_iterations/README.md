# Task 4: average total number of BP iterations

`avg_iterations.py --traces <task1 or task5 output>` writes `AVG_ITERATIONS.md` and
`avg_iters_<code>.png`: mean (and std) of the total BP iterations per frame for every
decoder and Eb/N0, the mean over the frames that reached the SGT, the mean number of
SGT calls, and a cross-code table at each code's middle Eb/N0. No GPU needed.
