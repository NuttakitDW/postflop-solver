# NN-Accelerated Postflop Solver — Model Spec

## Goal

Replace turn/river subtree simulation with a Neural Network during DCFR solving.
The NN predicts counterfactual values (CFVs) at the flop-to-turn boundary,
eliminating the most expensive part of the computation.

## Background: Why Static Values Don't Work

DCFR equilibrium selection depends on the CFV **evolution path**, not just the
final Nash values. A static oracle (fixed Nash CFV matrix) produces ~10% average
strategy difference because it freezes turn/river CFVs at Nash from iteration 0,
causing DCFR to converge to a different equilibrium at near-indifferent hands.

Both the standard and oracle outputs are valid Nash equilibria, but they differ
in strategy. The problem is structural — no parameter tuning can fix it.

## Proven Concept: Dynamic Boundary Oracle

### What the NN must learn

The NN replaces a **per-iteration CFV matrix** at the flop-to-turn boundary.
During standard DCFR solving, the turn/river subtree returns different CFVs at
each iteration as strategies evolve. The NN must predict these evolving CFVs,
not just the final converged values.

### Boundary interface

At each turn boundary node, during each DCFR iteration:

**Inputs (flop → turn):**
- `cfreach`: opponent's counterfactual reach probabilities (vector, dim = n_opp_hands)
- `amount`: pot size at boundary (identifies which flop action sequence led here)
- `player`: which player's CFVs to compute (0=OOP, 1=IP)
- Convergence state (see [Open Questions](#open-questions))

**Output (turn → flop):**
- `cfv`: player's counterfactual values (vector, dim = n_player_hands)

The relationship is linear: `cfv = M_t × cfreach` where M_t is the boundary
matrix at iteration t. The NN can either predict M_t or directly predict cfv
given cfreach.

### Critical constraint: player-asymmetric extraction

The library's `solve_step` uses **alternating updates**:
1. Player 0 (OOP) traversal → updates player 0's regrets everywhere
2. Player 1 (IP) traversal → sees player 0's UPDATED regrets

This means at the turn/river boundary:
- **Player 0's matrix** uses pre-iteration regrets (before any traversal)
- **Player 1's matrix** uses player 0's updated regrets (after player 0's traversal)

The NN must respect this asymmetry. Player 1's prediction must account for
player 0's regret updates from the same iteration.

### Validation result

A perfect lookup table with correct per-player extraction timing reproduces the
standard solver output with **max diff 0.000003** (floating-point rounding only).

| Metric | Lookup Table | Static Oracle |
|--------|-------------|---------------|
| Avg strategy diff | 0.000000 | 10.45% |
| Max strategy diff | 0.000003 | 60%+ |
| Elements >1% diff | 0 (0.00%) | many |

## Model Architecture Considerations

### Option A: Matrix prediction

- **Input**: (convergence_state, amount, player)
- **Output**: full matrix M_t [n_player × n_opp]
- Pro: one forward pass per (boundary, player, iteration)
- Con: output dim is huge (e.g. 863×526 = 454K floats for toy config)

### Option B: Direct CFV prediction

- **Input**: (cfreach, amount, player, convergence_state)
- **Output**: cfv vector [n_player]
- Pro: much smaller output, natural for NN
- Con: one forward pass per (boundary, player, iteration) — same frequency,
  but input includes the full cfreach vector

### Option C: Low-rank matrix approximation

- **Input**: (convergence_state, amount, player)
- **Output**: factors U [n_player × k], V [k × n_opp] where M ≈ U × V
- Pro: compressed representation, captures structure
- Con: rank k selection, training complexity

Option B is likely the most practical starting point.

## Training Data Pipeline

### Generation (per spot)

1. Run `build_dynamic_oracle` with a config — performs full standard DCFR solve
2. At each iteration, extracts boundary matrix M_t for each (amount, player)
   with correct per-player timing
3. Stores boundary pairs to `.dpairs` file

From each snapshot's matrix M_t, training pairs can be generated:
- Sample cfreach vectors (random, or record actual cfreach during solving)
- Compute cfv = M_t × cfreach
- Label with (iteration, exploitability, amount, player)

### Cost

- Requires full-tree solve per spot (same cost as normal solving + extraction overhead)
- Toy config (863 OOP, 526 IP, 3 boundaries): ~326s, 509 MB per spot
- This is a one-time cost per spot; the trained NN amortizes across inference

### Generalization across spots

One spot is insufficient for a generalizable model. Training data must span:

| Dimension | Examples |
|-----------|----------|
| Board texture | Paired, monotone, rainbow, connected, disconnected |
| Ranges | Different preflop action sequences (SRP, 3bet, 4bet) |
| Stack depth | 10bb to 200bb+ effective |
| Tree structure | Different bet sizes, raise counts |

The number of training spots needed is an empirical question.

## Open Questions

### Convergence state representation

Iteration number alone won't generalize across spots (spot A converges in 48
iterations, spot B in 500). Candidates:

| Feature | Pro | Con |
|---------|-----|-----|
| Iteration number | Simple | Doesn't generalize |
| Exploitability | Directly measures convergence | Expensive to compute |
| Normalized iteration (iter/total) | Simple | Requires knowing total in advance |
| Flop-side regret magnitudes | Cheap, local signal | Unclear if sufficient |
| cfreach delta between iterations | Measures stability | Requires tracking history |

### Spot parameterization for generalization

How to encode spot-level features (board, ranges, stack) as NN inputs so the
model generalizes to unseen spots.

### Training objective

- MSE on CFV vectors?
- Strategy-weighted loss (weight by actual cfreach)?
- Loss that prioritizes early iterations (where dynamics matter most)?

## Files

| File | Description |
|------|-------------|
| `examples/build_dynamic_oracle/main.rs` | Training data generator (records boundary pairs during DCFR) |
| `examples/compare_flop_files.rs` | Strategy comparison tool |
| `config/toy.json` | Toy config (C-oracle-3 ranges, small tree) |
| `data/oracles/toy.dpairs` | Training pairs (294 pairs, ~1.6 KB) |
| `data/out/toy.flop` | Standard-solved tree |

## Commands

```bash
# Generate training data (records cfreach/cfv pairs at turn boundary)
cargo run --example build_dynamic_oracle --release --features "bincode rayon" -- config/toy.json
```
