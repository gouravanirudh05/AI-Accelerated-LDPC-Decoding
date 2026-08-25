# Current GAT+BP architecture

This document records the tensor flow of the current efficient decoder.

For a parity-check matrix `H` with shape `[m, n]` and `E = nnz(H)`:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `channel_llr` | `[B, n]` | Channel LLR; positive means bit 0 |
| `variable_to_check` | `[B, E]` | Final BP variable-to-check messages |
| `check_to_variable` | `[B, E]` | Final BP check-to-variable messages |
| `bp_llr` | `[B, n]` | Channel LLR plus incident BP check messages |
| `variable_features` | `[B, n, 5 + P]` | Channel, BP posterior, residual, magnitude, normalized variable degree, and optional spectral features |
| `check_features` | `[B, m, 6 + P]` | Aggregated BP messages, normalized degree, soft-check residual, and optional spectral features |
| `edge_features` | `[B, E, 6]` | Signed/magnitude/tanh versions of both directed BP messages |
| `v_to_c_attention` | `[B, E, heads]` | Variable-to-check attention, normalized per check node |
| `c_to_v_attention` | `[B, E, heads]` | Check-to-variable attention, normalized per variable node |
| `edge_correction` | `[B, E]` | Bounded learned correction to each check-to-variable message |
| `correction` | `[B, n]` | Scatter-sum of edge corrections |
| `final_llr` | `[B, n]` | `bp_llr + correction` |

The edge order is canonical and grouped by variable node. `check_order` is a
permutation of the same physical edges grouped by check node for BP updates.
The graph therefore has one physical edge entry per non-zero element of `H`;
the two directed BP messages are separate channels on that edge.

## Current data flow

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
edge correction MLP, initialized to zero and bounded by 2.0 LLR units
    │
    ▼
scatter-add correction to variable nodes
    │
    ▼
final_llr [B, n]
```

## Learned refinement path

The model now uses the intended insertion point after `_features()`:

```text
features → bidirectional V→C and C→V attention
         → edge representation
         → Δ(check_to_variable) on [B, E]
         → scatter-add to [B, n]
         → final_llr
```

The BP implementation already exposes both directed message tensors when
`return_messages=True`, so no BP redesign is required for that refactor.
