# GAT+BP architecture

This document records the tensor flow of the current efficient decoder.

For a parity-check matrix `H` with shape `[m, n]` and `E = nnz(H)`:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `channel_llr` | `[B, n]` | Channel LLR; positive means bit 0 |
| `variable_to_check` | `[B, E]` | Current variable-to-check messages |
| `check_to_variable` | `[B, E]` | Current check-to-variable messages |
| `bp_llr` | `[B, n]` | Channel LLR plus incident check messages before the current GAT correction |
| `variable_features` | `[B, n, 5 + P]` | Channel, BP posterior, residual, magnitude, normalized variable degree, and optional spectral features |
| `check_features` | `[B, m, 6 + P]` | Aggregated BP messages, normalized degree, soft-check residual, and optional spectral features |
| `edge_features` | `[B, E, 6]` | Signed/magnitude/tanh versions of both directed BP messages |
| `v_to_c_attention` | `[B, E, heads]` | Variable-to-check attention, normalized per check node |
| `c_to_v_attention` | `[B, E, heads]` | Check-to-variable attention, normalized per variable node |
| `v2c_delta`, `c2v_delta` | `[B, E]` | Bounded learned residuals to the two directed message channels in iterative mode |
| `edge_correction` | `[B, E]` | Bounded learned posterior correction in one-shot mode, or the latest message correction in iterative mode |
| `correction` | `[B, n]` | Posterior residual relative to the current BP state |
| `final_llr` | `[B, n]` | `bp_llr + correction` |

The edge order is canonical and grouped by variable node. `check_order` is a
permutation of the same physical edges grouped by check node for BP updates.
The graph therefore has one physical edge entry per non-zero element of `H`;
the two directed BP messages are separate channels on that edge.

## One-shot mode (`gat_every=0`)

```text
channel_llr [B, n]
    │
    ▼
SumProductBP, K flooding iterations
    ├── bp_llr [B, n]
    ├── variable_to_check [B, E]
    └── check_to_variable [B, E]
    │
    ▼
_features()
    ├── variable_features [B, n, 5 + P]
    ├── check_features [B, m, 6 + P]
    └── edge_features [B, E, 6]
    │
    ▼
BidirectionalEdgeGATv2
    ├── V→C local edge-conditioned attention
    └── C→V local edge-conditioned attention
    │
    ▼
posterior correction MLP, initialized to zero and bounded by correction_scale
    │
    ▼
scatter-add correction to variable nodes
    │
    ▼
final_llr [B, n]
```

## Iterative mode (`gat_every>0`)

The Colab notebook uses six BP rounds and applies the shared GAT block every
two rounds. The GAT updates both directed message channels, so later BP rounds
receive the learned graph correction:

```text
for each BP round:
    SumProductBP.step()
    if round % gat_every == 0:
        features → bidirectional V→C and C→V attention
                 → Δ(variable_to_check), Δ(check_to_variable)
                 → update both directed message channels
                 → posterior LLR for deep supervision
```

The same GAT weights are reused at every scheduled round to keep parameter
count low. `SumProductBP.step()` exposes one differentiable flooding update,
while the training script adds frame-focused loss, optional soft frame loss,
and intermediate-output supervision.
