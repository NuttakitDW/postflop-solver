# Final Attempt: Tree-Extracted Oracle (SUCCESS)

## Previous Attempts — Why They Failed

### Frozen Solver
Froze turn/river strategies in the tree, re-read them during flop DCFR.
**Failed because**: `compute_cfvalue_recursive` uses `normalized_strategy` (average), but `solve_recursive` uses `regret_matching` (instantaneous). Frozen regrets break DCFR convergence — regrets must evolve at ALL nodes.

### Independent Oracle (OracleLookupTable)
Solved turn subgames independently, built CFV matrices per (amount, turn_card), averaged across 49 turn cards dividing by `num_turn_cards = 49`.
**Failed because**: the full tree uses `chance_factor = 45` (not 49) and applies isomorphism (suit swaps that reduce unique turn children). The independent oracle had no knowledge of these — structural mismatch caused Phase 3 max_diff = 0.022–0.025.

### Root Cause Summary
Both failures came from **trying to reproduce the tree's behavior externally** without capturing its exact internal mechanics (strategy source, chance_factor, isomorphism).

## Final Fix: Extract Directly from the Solved Tree

Instead of rebuilding the turn/river behavior externally, **probe the solved tree itself** to extract the exact CFV matrix.

### How It Works
CFVs are linear in opponent reaches: `CFV = M × cfreach`. To extract matrix M:
1. For each opponent hand j, set `cfreach[j] = 1`, all others = 0 (basis vector)
2. Call `compute_cfvalue_recursive` at the turn chance node → get column j of M
3. Repeat for all opponent hands → full matrix

This captures **everything**: chance_factor=45, isomorphism swaps, Nash strategies — because it uses the same code path the full solver uses.

### Result
- **Phase 3** (oracle vs tree): max_diff = 0.000000
- **Phase 5** (exploitability): delta = 0.002%
- **POC proven**: flop-only DCFR with tree-extracted oracle = full DCFR

### Trade-off
Requires a full solve first to extract the oracle. This is the POC — production replaces the matrix with a neural network trained on many solved examples.

## Key Code
- `oracle_solver.rs:TreeOracle::build()` — basis vector probing (parallelized with rayon)
- `oracle_solver.rs:evaluate_turn_boundary()` — matrix × cfreach at runtime
- `oracle_solver.rs:solve_recursive_with_oracle()` — DCFR that intercepts at turn chance nodes
