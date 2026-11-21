# Survey: state-of-the-art GNNs for bipartite / heterogeneous graphs, applied to Tanner graphs

Scope, as asked by Gourav (2026-11-21): the general GNN literature only (bipartite and
heterogeneous message passing, hypergraph networks, graph transformers with structural
encodings, long-range and expressive GNNs). LDPC / neural-decoding papers were deliberately
**not** searched. Every candidate is judged for the slot it would fill: the module between
two BP runs (`BP(T) -> module -> BP(T)`), which has to

1. produce one correction per bit (node-level output on the bit side);
2. see past BP-GNN's 2-hop view, without BP-SGT's "only the weak top-k bits attend" restriction;
3. stay cheap on 5G n = 1408 (E ~ 10^4 edges) and on a GPU at small batch, where the learned
   hybrids already lose to BP on latency.

A Tanner graph is a bipartite graph with two node types (bits, checks), or equivalently a
hypergraph whose hyperedges are the checks. In the GNN literature, the closest match is not a
molecule or social graph but the **variable-constraint bipartite graph** of combinatorial
optimisation (MILP, SAT). Bits are the variables, checks the constraints, H the constraint matrix.

## Candidates

| Family | Representative work | Core idea | Fit to Tanner graphs | Cost per layer | Verdict |
|---|---|---|---|---|---|
| **Variable-constraint bipartite transformers** | Dual-attention MILP backbone (Huang et al., ICML 2026) | Two branches per layer: linear self-attention *within* each node type (all variables, all constraints) + sparse, coefficient-aware cross attention on the nonzeros of A | Same graph class. The cross branch is the Tanner graph's own edges; the self branch gives every bit a global view without selecting nodes. The paper reports that a two-stage bipartite GNN's influence decays beyond ~3 hops while theirs stays non-zero at 7 | O((n+m+E) d^2) | **Implemented (DAT, main model)** |
| **Non-backtracking message passing** | NBA-GNN (Park et al., arXiv 2310.07430) | Directed edge states; the message along i->j excludes what came from j (sum minus self). Shown to reduce over-squashing and to improve sensitivity on sparse graphs (LRGB Peptides-func 0.62 -> 0.72 over GCN) | This is BP's extrinsic rule. BP-GNN and BP-PGNN layers are *backtracking*: a bit's own state returns to it through its checks after one layer. On graphs full of short cycles that echo self-reinforces, the trapping-set failure mode | O(E d h) | **Implemented (DAT-NB, second candidate)** |
| Bipartite GCNN for MILP + expressivity | Gasse et al. (NeurIPS 2019); Chen et al. (ICLR 2023); branching-capacity follow-ups (2024) | Two half-convolutions V->C->V. 1-WL-bounded, fails on symmetric ("foldable") instances, and random features fix that | BP-GNN is exactly this architecture. Quasi-cyclic codes are highly symmetric, but per-frame LLRs already break the symmetry, so the random-feature fix brings little | O(E d h) | Already the baseline (BP-GNN) |
| Linear graph transformers | SGFormer (Wu et al., NeurIPS 2023), Polynormer (Deng et al., ICLR 2024) | One global linear-attention layer, or polynomial-expressive linear attention, combined with local MP. Scales to millions of nodes | Supplies the "attend to every bit" part, but treats the graph as homogeneous. DAT uses the same linear-kernel idea, per node type | O(N d^2) | Folded into DAT's self branch |
| GPS-style hybrids / sparse transformers | GraphGPS (Rampášek et al., NeurIPS 2022), Exphormer (Shirzad et al., ICML 2023) | MPNN + global attention (+ PE); Exphormer's sparse attention runs on an expander overlay + virtual nodes | An LDPC Tanner graph *is* an expander, so Exphormer's overlay adds nothing beyond attention on H + a virtual node. Dense softmax GPS is O(N^2), the BP-SGT cost problem | O(N^2 d) or O(E d) | Not used; its useful part (attention on the graph's edges) is DAT's cross branch |
| Hypergraph networks | AllSet (Chien et al., ICLR 2022), ED-HNN (Wang et al., ICLR 2023), Sheaf Hypergraph Networks (Duta et al., NeurIPS 2023) | Checks as hyperedges. ED-HNN: equivariant hyperedge->node messages that depend on the receiver. Sheaf HNN: learned restriction maps per incidence | ED-HNN's receiver-dependent message is already in BP-GNN's EdgeMLP (`wr(h_recv)`). Sheaf maps are the interesting part: a parity check relates bits by *sign flips* (heterophily), which O(d) or diagonal restriction maps can represent. Costs E d^2 for full maps | O(E d) to O(E d^2) | **Follow-up**: sign-aware sheaf maps on the cross branch |
| Heterogeneous GNNs | HGT (Hu et al., WWW 2020), SimpleHGN (Lv et al., KDD 2021) | Type-specific projections / attention per relation type | Only two node types and one relation. 5G base-graph edge types are already embeddings. DAT's per-type Q/K/V *is* the HGT idea for two types | O(E d) | Folded into DAT |
| Adaptive rewiring / cooperative MP | Co-GNN (Finkelshtein et al., ICML 2024), MAVN adaptive virtual nodes (Lee & Whang, KDD 2026) | Each node learns to listen/broadcast/isolate (Co-GNN). Learned virtual nodes and node-VN edges per layer (MAVN, +13-23 % on LRGB) | Attractive idea, "satisfied checks stay silent, violated ones broadcast", but Co-GNN runs a second GNN to pick actions (about 2x cost). MAVN overlaps BP-PGNN's learned pooling | O(E d) x 2 | Not used (cost; overlaps PGNN) |
| State-space graph models | Graph Mamba (Behrouz & Hashemi, KDD 2024; Wang et al., 2024) | Order nodes, run a selective scan | Needs an arbitrary node order. The scan is sequential in N, a poor fit for GPU latency at small batch | O(N d) but sequential | Not used |
| Higher-order / cellular | CW Networks (Bodnar et al., NeurIPS 2021), subgraph GNNs | Lift cycles to 2-cells and pass messages between cycles | Strongest motivation of all: trapping sets are unions of short cycles. But enumerating 6/8-cycles of 5G BG1 at Z = 32 is large and code-specific, and the cost grows with the cycle count | O(#cycles x len x d) | Follow-up only; RWSE (kept) is the cheap proxy |
| Structural / positional encodings | RWSE (GPS), SignNet/BasisNet, SPE (Huang et al., ICLR 2024) | Stable PEs from the Laplacian spectrum | Regular Tanner graphs have highly degenerate eigenspaces, which is where LapPE breaks and SPE is needed. RWSE return probabilities count short cycles directly | precomputed | RWSE kept from BP-PGNN |

## Why the two implemented ones should help on Tanner graphs (hypotheses, to be tested)

**DAT, the dual-attention backbone.**

* *Global self-attention without selection.* BP-SGT ranked bits by |L| and let only the weakest
  k attend among themselves. DAT's self branch uses linear attention with a sigmoid kernel over
  **all n bits** (and, separately, all m checks). Every bit's query sees a content-weighted
  summary of every bit, the confident-and-wrong ones included. Cost is O(N d^2), not O(k^2 m).
  On 5G 1408 the whole DAT stage costs 36 M MACs, against 768 M for BP-SGT (k = 0.5 n) and
  45 M for BP-GNN.
* *Check-to-check global attention.* Violated checks that sit in the same stuck structure are
  usually more than 2 hops apart on long codes. The check self-attention lets them exchange state
  directly, which neither BP-GNN (2 hops) nor BP-SGT (bits only) can.
* *Cross attention instead of mean aggregation.* BP-GNN averages d_c messages, so the one
  disagreeing bit is diluted (the CCSDS failure). DAT's cross branch is a softmax over a check's
  bits, scored with the BP messages v2c and c2v as edge features (the analogue of the MILP
  coefficient A_ij). A check can single out the bit that disagrees with it.
* *Type-separate attention* keeps the bipartite structure (HGT-style): bits never attend to
  checks through the global branch, and the cross branch only uses the edges of H.

**DAT-NB, the non-backtracking cross branch.** Same self branch, but the Tanner half-steps run
on directed edge states and exclude the reverse edge: s_{c->v} = MLP(sum_{v' != v} s_{v'->c}, h_c, e).
This is the extrinsic principle of BP, lifted to d-dimensional learned messages. In BP-GNN, a
bit's own belief comes back to it through each of its checks one layer later. On a code with
many short cycles, that echo makes a wrong confident bit look confirmed by its neighbourhood.
NBA-GNN's analysis is that non-backtracking walks also mix faster and reduce over-squashing on
sparse graphs, and a Tanner graph is a sparse graph with a large girth-limited tree-like
neighbourhood, the regime where the bound helps most.

**Ablation.** DAT-NoSelf (cross attention only) against DAT measures whether the global branch
is where any gain comes from. DAT-NB against DAT compares the two local branches.

## Sources

- Huang, Wu, Ma, Wu, Zhang, Song. *A General Neural Backbone for Mixed-Integer Linear Optimization via Dual Attention*, ICML 2026. https://arxiv.org/pdf/2601.04509
- Park, Ryu, Kim, Woo, Yun, Ahn. *Non-backtracking Graph Neural Networks*. https://arxiv.org/pdf/2310.07430
- Lee, Whang. *Learn When and Where to Connect: Adaptive Virtual Nodes for Dynamic Message Passing on Graphs*, KDD 2026. https://arxiv.org/pdf/2606.03068
- Finkelshtein, Huang, Ravanbakhsh, Bronstein. *Cooperative Graph Neural Networks*. https://arxiv.org/pdf/2310.01267
- Deng et al. *Polynormer: Polynomial-Expressive Graph Transformer in Linear Time*, ICLR 2024. https://arxiv.org/abs/2403.01232
- Wang, Yang, Chen, Li. *Equivariant Hypergraph Diffusion Neural Operators*, ICLR 2023. https://arxiv.org/pdf/2207.06680
- Duta, Cassarà, Silvestri, Liò. *Sheaf Hypergraph Networks*, NeurIPS 2023. https://arxiv.org/pdf/2309.17116
- Behrouz, Hashemi. *Graph Mamba: Towards Learning on Graphs with State Space Models*. https://arxiv.org/pdf/2402.08678
- *Rethinking the Capacity of Graph Neural Networks for Branching Strategy* (MILP bipartite GNN expressivity). https://arxiv.org/pdf/2402.07099
- *Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation* (2025). https://arxiv.org/html/2512.17762v1
- Also cited from background knowledge (not re-fetched): GraphGPS (NeurIPS 2022), Exphormer (ICML 2023),
  SGFormer (NeurIPS 2023), AllSet (ICLR 2022), HGT (WWW 2020), SimpleHGN (KDD 2021), CW Networks
  (NeurIPS 2021), SPE (ICLR 2024), Gasse et al. bipartite GCNN (NeurIPS 2019), Chen et al. MILP-GNN
  expressivity (ICLR 2023), GNN+ (Luo, Shi & Wu, ICML 2025).
