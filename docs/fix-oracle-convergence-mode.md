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

## Proven Concept: Dynamic Boundary Pairs

### What the NN must learn

The NN replaces a **per-iteration CFV** at the flop-to-turn boundary.
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
- **Player 0's boundary CFV** is computed with pre-iteration regrets (before any traversal)
- **Player 1's boundary CFV** is computed with player 0's updated regrets (after player 0's traversal)

The NN must respect this asymmetry. Player 1's prediction must account for
player 0's regret updates from the same iteration.

### Validation result

A perfect lookup table reproduces the standard solver output with **zero diff**
on both simple and complex tree configurations.

| Config | Nodes | Elements | Max Diff | >10% |
|--------|-------|----------|----------|------|
| toy.json (simple) | 4 | 5,556 | **0.000000** | 0 |
| 9s6d6c.json (complex) | 26 | 51,393 | **0.000000** | 0 |

This proves the boundary pairs concept works perfectly — recording CFVs at turn
boundaries during standard DCFR and replaying them reproduces bit-identical flop
strategies regardless of tree complexity.

## Bug History (all fixed)

Three bugs were discovered and fixed across v1→v2 iterations:

### Bug 1: Custom DCFR missing convergence_mode (v1, 70% diff)

The library activates `convergence_mode` when `exploitability < 1% of pot`,
changing discount params (beta_t: 0.5→0.9). Our v1 custom DCFR hardcoded
`convergence_mode = false`.

**Fix**: Use `solve_step_for_player` (library's own per-player DCFR).

### Bug 2: Duplicate boundary keys (v1, 42% diff)

The v1 `.dpairs` format used pot amount as HashMap key. Different action
sequences can reach the same pot size → key collision → wrong CFVs.

**Fix**: Use sequential DFS index instead of pot amount.

### Bug 3: Float precision divergence (v2 early, 3.56% >10% diff)

Even after bugs 1-2 were fixed, v2 still had 3.56% of elements with >10% diff
on 9s6d6c.json. Root cause: the build recorded boundary CFVs via a separate
read-only traversal (`collect_boundary_cfvs_recursive`) that reimplemented
the library's CFV computation, while the replay reimplemented DCFR from scratch.
Both used mathematically identical algorithms, but different code paths produced
~1e-8 float precision differences (e.g. `v * (1/c)` vs `v / c` for cfreach
scaling, different regret_matching loop structure). These tiny differences got
**amplified through regret sign-flips** at near-zero values, causing cascading
divergence over 150 iterations.

**Fix**: Added `solve_recursive_recording` and `solve_recursive_replay` to
`src/solver.rs` — copies of the library's `solve_recursive` with boundary hooks.
Recording captures CFVs during the actual DCFR traversal (not a separate pass).
Replay injects pre-recorded CFVs at turn boundaries using the library's exact
same code for all other operations. This eliminates all float precision
differences, producing bit-identical results.

## Pipeline v2

### Process

Two-step: **build oracle first**, then **solve with oracle**.

1. `build_pairs_v2` — runs the full standard DCFR solve (flop + turn + river),
   recording boundary CFVs at each turn boundary per iteration. Outputs both
   the `.dpairs2` oracle file and a `.flop` ground truth file.

2. `solve_with_pairs_v2` — builds only the flop tree (9 GB → flop-only memory),
   replays the recorded boundary CFVs using the library's exact DCFR code.
   Produces a `.flop` file that should be bit-identical to the ground truth.

### Tools

| Tool | Input | Output | Speed |
|------|-------|--------|-------|
| `build_pairs_v2` | config.json | .dpairs2 + .flop | ~1x standard solve time |
| `solve_with_pairs_v2` | config.json + .dpairs2 | .flop | ~0.02s (flop-only) |
| `compare_flop_files` | file1.flop file2.flop config.json | comparison stats | instant |

### Commands

```bash
# Step 1: Build oracle (runs full solve, records boundary CFVs)
cargo run --example build_pairs_v2 --release --features "bincode rayon" -- config/9s6d6c.json

# Step 2: Replay with oracle (flop-only DCFR, instant)
cargo run --example solve_with_pairs_v2 --release --features "bincode rayon" -- config/9s6d6c.json

# Step 3: Compare results (should show max diff = 0.000000)
cargo run --example compare_flop_files --release --features "bincode rayon" -- \
  data/out/9s6d6c-standard.flop data/out/9s6d6c-pairs2.flop config/9s6d6c.json
```

### File format v2

**.dpairs2** (Dynamic Pairs v2) — Boundary cfv values indexed by DFS order:
- Header: magic `DPAIRS2\0`, version, num_oop, num_ip, num_boundaries, num_iterations, starting_pot
- Per iteration: iteration, exploitability, then for each boundary × each player: cfv[n_player]
- Sequential access by iteration; boundary index is implicit from DFS order

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

1. Run `build_pairs_v2` — uses library's DCFR with boundary cfv recording
2. At each iteration, records cfv at each turn boundary for each player
3. Stores to `.dpairs2` file

### Generalization across spots

One spot is insufficient for a generalizable model. Training data must span:

| Dimension | Examples |
|-----------|----------|
| Board texture | Paired, monotone, rainbow, connected, disconnected |
| Ranges | Different preflop action sequences (SRP, 3bet, 4bet) |
| Stack depth | 10bb to 200bb+ effective |
| Tree structure | Different bet sizes, raise counts |

## Open Questions

### Convergence state representation

Iteration number alone won't generalize across spots. Candidates:

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

### Current (v2)

| File | Description |
|------|-------------|
| `src/solver.rs` | Library — added `solve_step_for_player_recording` and `solve_step_for_player_replay` |
| `examples/build_pairs_v2/main.rs` | Step 1: build oracle using library's recording DCFR |
| `examples/solve_with_pairs_v2/main.rs` | Step 2: replay solver using library's replay DCFR |
| `examples/compare_flop_files.rs` | Strategy comparison tool |
| `config/toy.json` | Toy config — simple tree, verified 0-diff |
| `config/9s6d6c.json` | Complex config — verified 0-diff |

### Library API (src/solver.rs)

| Function | Description |
|----------|-------------|
| `solve_step_for_player_recording` | DCFR for one player, returns `Vec<Vec<f32>>` of boundary CFVs |
| `solve_step_for_player_replay` | DCFR for one player, injects pre-recorded boundary CFVs |
| `solve_recursive_recording` | Internal: `solve_recursive` + boundary CFV capture |
| `solve_recursive_replay` | Internal: `solve_recursive` + boundary CFV injection |

### Legacy (v1, known bugs — do not use)

| File | Description |
|------|-------------|
| `examples/build_pairs/main.rs` | Buggy (no convergence_mode, amount-keyed) |
| `examples/solve_with_pairs/main.rs` | Buggy (amount-keyed, reimplemented DCFR) |
| `examples/build_dynamic_oracle/main.rs` | Full data generator (slow, same bugs) |
| `examples/solve_with_dynamic_oracle/main.rs` | Replay solver using .doracle |
