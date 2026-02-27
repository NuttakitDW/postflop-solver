# Attempt 1: Lookup Table at Turn Boundary (DeepStack-style POC)

## Context

We want to prove the DeepStack concept: **solve only the flop using DCFR, with a lookup table replacing the entire turn+river subtree at the boundary**.

### Why the frozen approach failed
The frozen solver (examples/poc_precompute/) tried to freeze turn/river strategies inside the tree and re-read them during flop DCFR. This failed because:
- `compute_cfvalue_recursive` uses `normalized_strategy(node.strategy())` (average strategy)
- `solve_recursive` uses `regret_matching(node.regrets())` (per-iteration strategy)
- Switching to regret_matching also failed — DCFR needs regrets to evolve at ALL nodes; frozen regrets break convergence

### Why lookup tables should work
Instead of reading from frozen tree nodes, use an **external** matrix that represents the Nash CFV mapping:
- `CFV = Matrix × cfreach` — correct for ANY cfreach (proven linear)
- Matrix is extracted from independently solved turn subgames (Nash equilibrium)
- No dependency on internal tree storage — clean separation
- This is exactly what a neural network would approximate

## Approach

We already have all the infrastructure:
- `OracleLookupTable` (src/oracle.rs) — builds MatrixTurnCfv for all (amount, turn_card) pairs
- `OracleContext` (src/oracle.rs) — maps hand indices between flop and turn, calls matrix.evaluate()
- `solve_with_oracle()` (src/oracle.rs:1060) — DCFR on full PostFlopGame, uses oracle at turn chance nodes

The existing `solve_with_oracle` has **two problems**:
1. It calls `finalize(game)` which marks the game as solved (can't re-solve)
2. It can't compute exploitability (turn/river nodes have no strategies)

### The POC plan

**Phase 1: Full solve (ground truth)**
- Build PostFlopGame, solve with `solve_step()` loop (avoids finalize)
- Record exploitability and root CFVs

**Phase 2: Build oracle lookup tables**
- Use `OracleLookupTable::build()` to solve independent turn subgames for each (amount, turn_card)
- Extract MatrixTurnCfv for each
- Create `OracleContext` for hand mapping

**Phase 3: Validate oracle against full tree**
Before running DCFR, verify the oracle gives the same CFVs as the full tree at the turn boundary:
- Navigate to each turn chance node in the solved tree
- Compare `compute_cfvalue_recursive` result vs `oracle_ctx.evaluate_turn_boundary` result
- If they match, the oracle is correct; if not, we have a bug to fix first

**Phase 4: Flop-only DCFR with oracle**
- Reset flop regrets/strategy (reuse `reset_flop_storage()` from frozen_solver)
- Run DCFR loop that at turn chance nodes calls `oracle_ctx.evaluate_turn_boundary()`
- This is essentially `solve_recursive_oracle` but on the existing PostFlopGame tree (not a new FlopSolver)

**Phase 5: Compare**
- Compute exploitability on full tree (turn/river strategies from Phase 1 still exist)
- Compare root CFVs and EV with Phase 1
- If delta < 0.1%, concept is proven

## Implementation

### File: `examples/poc_precompute/main.rs` (modify existing)

```
Phase 1: Full solve (same as before)
Phase 2: Build oracle
  - Get boundary_amounts from ActionTree
  - Call OracleLookupTable::build(flop, ranges, pot, stack, amounts, bet_config, ...)
  - Create OracleContext::new(game, oracle)
Phase 3: Validate oracle vs full tree
  - For each player, call compute_cfvalue_recursive at root → get full_cfvs
  - Reset flop, call oracle DCFR → get oracle_cfvs
  - Compare
Phase 4: Oracle DCFR
  - reset_flop_storage()
  - Run DCFR loop using oracle at turn boundaries
Phase 5: Compare exploitability + CFVs
```

### File: `examples/poc_precompute/frozen_solver.rs` (rewrite key function)

Replace `compute_cfvalue_with_regret_matching` with oracle-based evaluation:

```rust
fn solve_recursive_with_oracle(
    result, game, oracle_ctx, node, player, cfreach, params,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }
    // Turn chance node → oracle lookup (THE KEY CHANGE)
    if node.is_chance() && node.turn() == NOT_DEALT {
        oracle_ctx.evaluate_turn_boundary(result, node.amount(), player, cfreach);
        return;
    }
    // Single action passthrough
    // Player node: regret matching + update regrets/strategy (same as standard DCFR)
    // Opponent node: update cfreach by strategy + recurse (same as standard DCFR)
}
```

### Key difference from frozen approach
- Frozen: reads strategies from tree nodes (regret/strategy storage) → wrong strategy source
- Oracle: reads from external matrix (Nash CFV mapping) → correct for any cfreach
- Oracle matrices come from independently solved turn games, not from the full tree's internal state

### TurnBetConfig
The oracle needs turn/river bet sizes. For the POC, use the same bet sizes from the config file.
Parse from config JSON: `config.solver.turn_bet_sizes` / `config.solver.river_bet_sizes`
Or use the common config parser that already exists in `examples/common/mod.rs`.

## Verification

1. **Oracle validation (Phase 3)**: Compare oracle CFVs vs full-tree CFVs at turn boundary
2. **Exploitability match**: frozen solve exploitability ≈ full solve exploitability (< 0.1% delta)
3. **CFV match**: root CFVs max_diff < 0.001
4. **EV match**: weighted EV difference < 0.01

## Run command

```bash
cargo run --example poc_precompute --release --features "bincode rayon" -- config/template.json
```
