# Oracle Summary

## What It Does
The oracle replaces turn+river subtree traversal with precomputed CFVs at turn boundary nodes during flop-only solving.

## Build Phase
1. For each `(boundary_amount, turn_card)` pair, solve a full turn+river game
2. Extract the solution into a `MatrixTurnCfv` — a matrix per player of shape `n_player_hands × n_opponent_hands`
3. Save all matrices to `.oracle` file

## Inference Phase (Flop Solve)
Each DCFR iteration calls `evaluate_turn_boundary` at every turn boundary node.

### Inputs
- **player** — which player (0 or 1)
- **amount** — chips at boundary; selects which precomputed matrix to use via `oracle.get(amount, card)`
- **cfreach** — opponent's counterfactual reach probabilities (changes every iteration as strategy updates)

### CFV Computation (`evaluate_turn_boundary`)
```
For each turn card:
  1. Map cfreach from flop indexing → turn indexing
  2. Select matrix via oracle.get(amount, card)
  3. CFV[i] = Σ_j matrix[i][j] * cfreach[j]    (MatrixTurnCfv::evaluate)
  4. Map CFVs back from turn indexing → flop indexing
Average across all turn cards (scale by 1/num_turn_cards)
```

### Output
- `result: [f32]` — one CFV per player hand in flop indexing

## Key Files
- `src/oracle.rs` — `OracleContext::evaluate_turn_boundary` (inference), `OracleLookupTable::build` (build)
- `src/turn_cfv.rs` — `MatrixTurnCfv::evaluate` (matrix-vector multiply), `MatrixTurnCfv::from_exact` (build matrix from solved game)

## Verified Correct
- cfreach changes every iteration (confirmed via logging)
- amount selects the matrix, cfreach is multiplied against it
- Oracle CFVs match freshly solved ExactTurnCfv perfectly (max_diff ~1e-6) — verified via `examples/verify_oracle.rs`
- Config settings (pot, stack, ranges, bet sizes) identical between oracle build and standard solve

## Ruled Out as Root Cause
- **Convergence mode**: Disabled in both oracle and standard solver — results still differ
- **Compression**: Set both to `useCompression: false` — results still differ
- **Oracle CFV accuracy**: Precomputed matrix matches fresh solve exactly

## Current Status
- Oracle flop strategy: OOP Check ~97.8%, Bet 18 ~2.2%
- Standard flop strategy: OOP Check 100%
- The recursive regret update logic (`solve_recursive_oracle` vs `solve_recursive`) is identical
- The outer loop control flow is identical (with convergence mode disabled in both)
- The CFVs at turn boundary are correct
- **Root cause is still unknown** — something causes different regret accumulation despite identical inputs/outputs at each node
