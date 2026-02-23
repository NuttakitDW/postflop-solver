# Oracle Solver (DeepStack-style)

## Quick Start
# Note: drop iterations number after oracle was built!! (suggest 150+)

```bash
# Step 1: Build the oracle (full solve + extract CFV matrices)
cargo run --example build_lookup_table --release --features "bincode rayon" -- config/A-oracle.json

# Step 2: Solve using the oracle (flop-only DCFR)
cargo run --example solve_with_oracle --release --features "bincode rayon" -- config/A-oracle.json
```

Step 1 produces `config/A-oracle.toracle`. Step 2 reads it and outputs the `.flop` file specified in config.

## How It Works

Standard CFR solves the entire game tree — flop, turn, and river — every iteration. The turn/river subtree is massive (49 turn cards x 48 river cards x action sequences), making each iteration expensive.

The oracle solver splits this into two phases:

**Build phase** — Solve the full tree once. At each turn boundary node (where flop actions end and the turn card is dealt), extract a CFV matrix `M` such that:

```
CFV = M × cfreach
```

where `cfreach` is the opponent's counterfactual reach probability vector. This works because CFVs are linear in opponent reaches — a known property of counterfactual values in imperfect-information games.

The matrix captures everything about the turn/river subtree: all 49 turn cards, river runouts, isomorphism, chance factors, and Nash strategies. It's extracted by probing the solved tree with basis vectors (one per opponent hand).

**Solve phase** — Run DCFR on flop nodes only. When the traversal hits a turn boundary, instead of recursing into the turn/river subtree, compute `M × cfreach` — a single matrix-vector multiply. This replaces thousands of recursive node evaluations with one O(n²) operation.

### Why It's Faster

| | Standard CFR | Oracle CFR |
|---|---|---|
| Tree depth | Flop + Turn + River | Flop only |
| Turn boundary | Recurse 49 turn cards | Matrix multiply |
| Per-iteration cost | O(full tree) | O(flop tree + n²) |

The speedup depends on how large the turn/river subtree is relative to the flop. With many bet sizes, the turn/river dominates — oracle iterations can be 10-50x faster.

### Trade-off

The build phase (full solve + extraction) is expensive — it's a one-time cost. The oracle file (`.toracle`) can then be reused to solve the same flop with different iteration counts.

## Exploitability Limitation

The oracle solver **cannot compute exploitability**. Exploitability requires a best-response traversal of the entire tree (flop + turn + river). Since the oracle solve only has strategies for flop nodes — turn/river nodes are replaced by the matrix — `compute_exploitability()` would traverse into empty turn/river storage and return meaningless results.

In practice this is fine: DCFR converges monotonically. More iterations always gets closer to Nash, never worse. Run a fixed iteration count and trust convergence. The `build_lookup_table` step reports the full-solve exploitability, which serves as the quality baseline.

## File Format

- `.toracle` — Oracle file containing CFV matrices for all boundary amounts and both players
- `.flop` — Standard solver output file with flop strategies

## Config

Uses the same JSON config as standard solves. The oracle solver reads `maxIterations` for the flop-only DCFR and writes to `output.filename`.
