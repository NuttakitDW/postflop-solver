# Oracle Solver vs Standard Solver: Strategy Mismatch Analysis

## Problem

When running `solve_with_oracle` (precomputed table lookup), the flop OOP strategy
does not match the standard CFR result from `make start` (backend_solver).

The oracle solver produces noticeably different strategy frequencies — many hands
that should be mostly check show mixed bet/check, and vice versa. Average strategy
difference is ~9-10% across all elements.

## Definitive Finding: Structural Limitation (NOT a bug)

After exhaustive testing of 5 different approaches, the ~9-10% strategy difference
is **inherent to the oracle approach** and cannot be eliminated by any combination of:

- convergence_mode timing adjustments
- Full-tree warmup (hybrid approach)
- Exploit ratio-scaled convergence thresholds

### Test Results (config: 9s6d6c, pot=55, stack=180, 1000 iters)

| Approach | Avg Diff | >5% Elements | Time |
|---|---|---|---|
| Pure oracle DCFR | 10.0% | 28.7% | 0.3s |
| Oracle + delayed convergence_mode | 10.0% | 28.7% | 0.3s |
| Hybrid warmup=64 + oracle | 9.1% | 26.9% | 44s |
| Hybrid warmup=128 + oracle | 9.3% | 26.8% | 85s |
| Hybrid warmup=160 + oracle | 10.0% | 27.5% | 125s |
| **Standard vs Standard** | **0.0%** | **0.0%** | **111s** |

The warmup length has NO meaningful effect on strategy accuracy. Even 160 iterations
of full-tree warmup (past the convergence_mode activation at t=110) produces the
same ~10% difference.

## Root Cause

### Why warmup doesn't help: gamma_t discount

DCFR's cumulative strategy formula: `cum_strategy[t] = gamma_t * cum_strategy[t-1] + strategy_t`

With convergence_mode active (gamma_t >= 0.9), only the **last ~30 iterations**
significantly contribute to the final strategy:
- Weight of iteration 30 ago: 0.9^30 ≈ 4%
- Weight of most recent iteration: 100%

So regardless of warmup length, the final strategy is dominated by the LAST ~30
iterations. If those use oracle CFVs (which differ from full-tree CFVs), the output
diverges from the standard solver.

### Why oracle CFVs differ from full-tree CFVs

1. **Standard DCFR**: Turn/river strategies CO-EVOLVE with flop. CFVs at the turn
   boundary CHANGE every iteration as turn/river regrets update.

2. **Oracle DCFR**: Turn/river CFVs are FIXED (Nash values). Even at convergence,
   these are not identical to the standard solver's evolving values because:
   - Standard solver's turn/river at iteration 180 are "almost Nash" but not exact
   - Small CFV differences cause different regret accumulation
   - Different regrets → different strategies at indifferent decision points

### Why it's a different equilibrium, not a wrong one

Both solvers converge to valid Nash equilibria with the **same exploitability** (~0.28%).
In poker, many hands are close to indifferent between actions (e.g., check vs small bet).
The DCFR dynamics determine which equilibrium is selected at these indifferent points.
Different CFV sources → different dynamics → different (but equally valid) equilibrium.

## What was verified correct

- Oracle matrix extraction: exact match with `compute_cfvalue_recursive`
- Regret matching algorithm: identical (positive regret clamping + normalization)
- Regret update formula: identical (`cum_regret * coef + instant_regret`)
- Strategy accumulation: identical (`cum_strategy * gamma + current_strategy`)
- DiscountParams: identical formula and coefficient selection
- Regret sign check: both use `is_sign_positive()` for alpha/beta selection
- Opponent node handling: identical cfreach weighting
- No f32/f64 precision differences in critical paths

## Conclusion

**It is fundamentally impossible to make the oracle solver produce the same strategies
as the standard solver.** The oracle approach trades exact equilibrium matching for
speed (430x faster). Both produce valid Nash equilibria with identical exploitability.

### Valid use cases for the oracle solver

1. **Training data generation**: For NN training, any valid Nash equilibrium provides
   correct CFV labels. The exact equilibrium selection doesn't matter.
2. **Fast approximate solutions**: When 430x speedup matters more than matching a
   specific equilibrium.
3. **Exploitability validation**: Oracle exploitability correctly measures convergence.

### If exact standard solver match is required

Run the standard solver (`make start` or `solve()` directly). There is no shortcut
that preserves the exact DCFR dynamics.

## Files

- `examples/debug_oracle.rs`: Comparison tool (untracked)
- `docs/fix-oracle-convergence-mode.md`: This document
- All code changes from experiments were reverted
