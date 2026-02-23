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

## Verified Behavior
- cfreach changes every iteration (confirmed via logging)
- amount selects the matrix, cfreach is multiplied against it
- If results are off, the issue is likely in how matrices are built (from_exact or ExactTurnCfv)
