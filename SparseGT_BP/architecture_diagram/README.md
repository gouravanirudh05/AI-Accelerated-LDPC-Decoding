# BP-SGT architecture diagrams

Block diagrams of the BP + sparse graph transformer decoder, drawn from the code
(`../sgt/sgt_decoder.py`, `SparseGTBPDecoder`, and `Nvidia_GNN+BP(lLatest)/common/bp.py`,
`staged_decode`) with the final configuration from `../README.md` ("Winner"):
4 SGT stages, BP restarted at every stage (`warm=False`), k = ceil(0.5 n),
2 blocks, d = 64, 4 heads, FF hidden 128, global attention with Tanner-graph
bias, |L| selection, residual fix on the previous prior.

| file | shows |
|---|---|
| `sgt_pipeline.{svg,png}` | the interleaved pipeline: BP(T) → SGT → BP(T) → … (5 BP runs, 4 SGT stages) and per-iteration early exit |
| `sgt_stage.{svg,png}` | one SGT stage (`refine()`): per-bit statistics, top-k selection, token features, Tanner-graph attention bias, transformer block, correction head and scatter back into the prior |

## Components of one SGT stage and their cost per codeword

From `macs_per_refinement()`; k = number of selected bits, m = checks, E = edges.

| # | component | code | cost |
|---|---|---|---|
| 1 | per-bit statistics (syndrome, unsatisfied-check count, c2v disagreement / support) | `refine()` scatter_sum over edges | O(E) |
| 2 | select k least-confident bits | `torch.topk` on \|L\| | O(n log k) |
| 3 | 9 features → Linear 9→d + VN-type embedding | `inp`, `vn_emb` | k·9·d |
| 4 | structure bias: shared / shared-unsatisfied checks between selected bits → per-head bias | `bmm(H_k, H_kᵀ)`, `pair_bias` | 2·k²·m (dense) |
| 5 | 2 pre-LN transformer blocks over k tokens | `Block` | 2·(k·(4d² + 2dh) + 2k²d) |
| 6 | LayerNorm → Linear d→1 (zero-init), δ·s added to prior at idx | `ln`, `head`, `scatter` | k·d |

With k = 0.5 n, item 4 (the dense k × k × m product) dominates on long codes
(768 M MACs per stage on 5G n = 1408), which is why the SGT costs more than the
BP-GNN there (see `../README.md`, Results).

## Re-rendering

Sources are Graphviz (`*.dot`):

```
dot -Tsvg sgt_pipeline.dot -o sgt_pipeline.svg && dot -Tpng -Gdpi=150 sgt_pipeline.dot -o sgt_pipeline.png
dot -Tsvg sgt_stage.dot    -o sgt_stage.svg    && dot -Tpng -Gdpi=150 sgt_stage.dot    -o sgt_stage.png
```
