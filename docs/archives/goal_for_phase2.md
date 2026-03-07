# Phase 2: From Lookup Table to Neural Network

## Goal

Replace the `.dpairs2` lookup table with a neural network that predicts boundary
CFVs, so we never need to solve the turn/river subtree.

## Training Data: What's Needed vs What We Have

The `.dpairs2` files store **outputs only** (CFV labels). The **input** the NN
needs — `cfreach` (opponent reach probabilities at each boundary) — is not stored.

```
NN training pair:
  input:  cfreach[opponent]  — float32[~500-900]  ← MISSING from .dpairs2
  output: cfv[player]        — float32[~500-900]  ← stored in .dpairs2
```

| Field | Stored? | Needed? | Role |
|-------|---------|---------|------|
| CFV vectors | Yes | Yes | Labels (NN output) |
| cfreach (opponent reach) | **No** | **Yes** | Features (NN input) |
| iteration, exploitability, convergence_mode | Yes | No | Solver bookkeeping only — reach probs already encode all state |
| starting_pot / boundary pot | Yes/Implicit | Maybe | Useful if training across multiple bet configs |

**Action**: extend `.dpairs2` to v3, add cfreach alongside CFVs per boundary.
File size ~2x (~25 MB → ~50 MB per board).

## Speed Problem and Alternatives

Current: **10-20 min per board** (full flop+turn+river solve). Bottleneck is
turn/river subtree. To cover all 1,755 strategic flops: ~18 days on one machine.

| Strategy | Speedup | Trade-off |
|----------|---------|-----------|
| Parallelize across boards | Linear (N cores) | None — boards are independent |
| Record every Kth iteration | 2-5x smaller data | Lose fine-grained trajectory |
| Early stop at 1% exploit | ~40% fewer iters | Slightly noisier labels |
| Sample 10/45 turn cards | ~4x faster | Noisier CFVs per sample |

At 100 cores: **~4 hours** for all 1,755 boards.

## Scale: Our Data vs DeepStack

| Metric | Our data | DeepStack |
|--------|----------|-----------|
| Strategic flops | 1,755 | 1,755 |
| Samples per flop | 4,500-10,000 | ~5,700 |
| Total samples | 7.9M-17.6M | 10M |
| Dataset size (est.) | ~50-100 GB | ~100 GB |

Numbers are directly comparable. We may need **less** because:
- Fixed action tree (same 25 boundaries across boards) = simpler function
- Structured DCFR trajectory vs DeepStack's random sampling
- Start with 50-100 boards, scale if needed

## Current Inventory

10 boards, ~85K samples, 228 MB (CFVs only, no cfreach yet).

## Ideas to Explore

### Hand Bucketing

Raw input/output is ~500-900 floats per hand combo. Most hands behave similarly
(e.g., all off-suit Kx hands on a low board). Bucketing groups similar hands
into clusters, reducing dimensionality:

- **Input**: cfreach averaged per bucket instead of per hand → ~50-200 floats
- **Output**: predict per-bucket CFV, expand back to per-hand
- Bucketing methods: equity-based, hand strength, k-means on CFV correlation
- Trade-off: lossy compression, but may regularize training and generalize better
- DeepStack uses ~200 buckets — good starting point

### Board Embedding

The NN must generalize across 1,755 flops. How to represent the board?

- **Card indices**: 3 integers (simple but no structure)
- **One-hot**: 52×3 binary (sparse)
- **Suit/rank features**: encode rank + flush/pair/straight texture
- **Learned embedding**: let the model learn board representation end-to-end
- Key question: does the model need board info at all, or is cfreach sufficient?

### Per-Boundary vs Single Model

- **Per-boundary model**: 25 separate small NNs, one per boundary. Simple, but
  no parameter sharing.
- **Single model with boundary conditioning**: one NN takes boundary index (or
  pot size) as extra input. Shares weights, may generalize better.
- **Hierarchical**: shared trunk + per-boundary head

### Loss Function

- **MSE** on raw CFVs (baseline)
- **Weighted MSE**: weight by reach probability (hands that are reached more
  matter more for strategy correctness)
- **Strategy-aware loss**: measure exploitability of the resulting strategy
  instead of CFV error directly (expensive but most aligned with goal)

### Warm-Start / Hybrid Approach

Instead of using NN for all 180 iterations, run exact solver for the first N
iterations (cheap when strategies are bad), then switch to NN once strategies
stabilize. Reduces accuracy requirements for early-iteration predictions.

## Steps

1. **Extend format**: add cfreach to `.dpairs2` v3
2. **Regenerate 10 boards** with cfreach, verify CFVs unchanged
3. **Train minimal NN** (MLP) on 1-3 boards: cfreach → CFV
4. **Test replay**: plug NN into `solve_with_pairs_v2`, measure exploitability
5. **Scale or iterate** based on results
