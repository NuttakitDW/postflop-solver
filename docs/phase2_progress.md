# Phase 2 Progress

## Experiment 1: Static Oracle (2025-02-28)

**Hypothesis**: Can we use only the final converged boundary CFVs (from the last iteration) as a static oracle for all iterations, instead of per-iteration CFVs?

**Why it matters**: If static oracle works, an NN only needs to predict converged equilibrium values — much simpler to train.

**Result**: Failed.

| Board | Root Avg Diff | Root Max Diff | Full Tree Avg Diff |
|-------|--------------|---------------|-------------------|
| KcQh7s | 0.22% | 5.3% | 19.6% |
| 9s6d6c | 7.0% | 57.3% | 27.4% |

Code: `examples/solve_with_static_oracle/main.rs`

**Conclusion**: Co-evolution matters. DCFR boundary CFVs must evolve per-iteration alongside flop strategies. A static oracle produces wrong strategies, especially on complex boards.

---

## Insight: What Makes Boundary CFVs Change Between Iterations

The CFV computation mechanism is identical every iteration — it's a pure function:

```
CFV_t(boundary) = tree_traversal(subtree_strategies_t, cfreach_t)
```

What changes between iterations are the **inputs**:

1. **subtree_strategies_t** — derived from cumulative regrets via regret matching: `strategy[a] = max(0, cumR[a]) / Σ max(0, cumR)`. Discounting updates cumR with different coefficients for positive regrets (alpha) vs negative regrets (beta), causing strategy shifts.

2. **cfreach_t** — opponent's strategy above the boundary changes each iteration, producing different reach probabilities at the boundary.

Discounting parameters drive the fluctuation pattern:
- **Early iterations**: alpha ≈ 0, so positive regrets barely carry over → strategies can flip completely → large CFV swings
- **Late iterations**: alpha ≈ 1, regrets accumulate stably → strategies converge → CFV stabilizes

The CFV fluctuation is NOT random — it's deterministic given the cumulative regret state.

---

## Key Realization: cfreach as NN Input Is Sufficient (Practical)

The equation `CFV_t = f(subtree_strategies_t, cfreach_t)` has two inputs. `subtree_strategies_t` is huge (entire turn/river tree state ~14 GB) and unavailable at inference time — we can't use it as NN input.

However, **cfreach_t already changes every iteration**. It reflects the current state of the flop solve. Unlike the static oracle (which used a FIXED CFV for all iterations), an NN receiving cfreach_t sees **different inputs** at each iteration and produces **different outputs**. The subtree strategy state is a hidden variable, but it's correlated with cfreach through the shared DCFR dynamics — both evolve together.

The NN learns: "when opponent reach looks like THIS → boundary CFV looks like THAT."

**Phase 1 (.dpairs2) is the foundation.** It proved boundary CFVs are sufficient to solve the flop correctly. The .bt1 format extends this by recording cfreach alongside CFV, giving us valid (input, output) training pairs.

**Data per board**: ~150 iterations × 25 boundaries × 2 players = ~7,500 pairs.
**Data across 10 boards**: ~75,000 pairs.

The NN doesn't need to be perfect — small prediction errors are tolerable as long as the flop DCFR converges to approximately correct strategies.

---

## Design Decision: Fixed-Size NN I/O via Canonical 1326 Indexing

**Problem**: cfreach and CFV vectors vary in size per board (e.g., OOP=860 on KcQh7s, OOP=863 on 9s6d6c). NN requires fixed-size input/output.

**Solution**: Map to canonical 1326-dim vectors (all possible hole card combos = C(52,2)).

```
NN input:  cfreach_nn[1326]  — opponent reach in canonical indexing, 0 for blocked/out-of-range hands
NN output: cfv_nn[1326]      — player CFV in canonical indexing, 0 for blocked/out-of-range hands
```

Conversion: solver uses compressed indices (only valid hands). Need a mapping `solver_hand_idx → canonical_1326_idx` from the game API to expand/compress between solver format and NN format.

**Why not bucketing?** Start simple and lossless. Bucketing (K=200, DeepStack-style) is an optimization to explore later if 1326-dim is too large or hurts generalization.

---

## Experiment 2: Model v1 — Single-Trajectory NN (2025-02-28)

**Setup**: Overfit MLP on KcQh7s .bt1 data (180 iters × 25 boundaries × 2 players = 9,000 samples).

- Architecture: 5-layer MLP, 512 hidden, LayerNorm + GELU
- Input: boundary_onehot(25) + player(1) + cfreach(1326) = 1352 dims
- Output: cfv(1326), scaled by 1/pot (55)
- Training RMSE: 0.016 chips (on 55-chip pot = 0.03% error)
- Code: `trainings/train_bt1.py`, `examples/solve_with_model_v1/main.rs`

**Result**: Failed badly. Worse than static oracle.

| Metric | Model v1 | Static Oracle |
|--------|----------|---------------|
| Root avg diff | ~35% | 0.22% |
| Strategy quality | Many hands 100% wrong action | Roughly correct |

**Root Cause: Compounding Error from Distribution Shift**

The model was trained on cfreach from ONE solve trajectory (the correct DCFR run). During inference:

1. **Iteration 0**: All regrets = 0 → uniform strategy → cfreach matches training → prediction ~OK
2. **Small CFV error** → slightly wrong regret update → slightly wrong strategies
3. **Iteration 1**: cfreach differs from training trajectory (because strategies diverged)
4. **Model sees out-of-distribution cfreach** → worse prediction → worse regret update
5. **Error compounds exponentially** — by iteration 20-30, strategies are completely wrong

This is the classic **autoregressive distribution shift** problem (same as DAgger in imitation learning, teacher-forcing gap in seq2seq).

**Why training RMSE is misleading**: 0.016 RMSE is measured *in-distribution* — the model memorized 9,000 (cfreach, CFV) pairs from the correct trajectory. But during inference, cfreach quickly goes out-of-distribution, and the model has NO guarantees on OOD inputs. It can output arbitrarily wrong CFVs.

**Why model v1 is worse than static oracle**: The static oracle uses a valid converged CFV (wrong iteration, but bounded error). The model can output garbage for OOD inputs → unbounded error.

**Key comparison — pairs_v2 vs model_v1**:

```
pairs_v2 (works):  CFV = exact_lookup[iteration_t][boundary_b][player]
                   → identical to original → zero error → zero drift

model_v1 (fails):  cfreach = collect_from_current_game_state(t)
                   CFV = model.predict(cfreach)
                   → prediction error → strategy drift → cfreach drift → compounding error
```

The fundamental issue: `f(cfreach) → CFV` is not well-defined because CFV depends on subtree strategies too. During training, cfreach correlates with subtree strategies (they co-evolve). During inference, this correlation breaks because the trajectory diverges.

**Lessons learned**:
1. Single-trajectory training doesn't generalize to the inference trajectory
2. Even 0.03% in-distribution error compounds destructively over 200 iterations
3. Need training strategies that are robust to distribution shift (DAgger, multi-trajectory, or fundamentally different approach)

---

## Deeper Investigation: Why the Model Doesn't Truly Overfit (2025-03-01)

### Finding 1: ONNX Model Is Correct in Python

Verified by feeding training data through the exported ONNX model in Python. The model produces outputs that match training targets — confirming the ONNX export and the training pipeline are correct.

### Finding 2: Sparse Output Target Problem

**The 1326-dim canonical output sabotages training.** For IP (player=1), only 502 of 1326 output dims are valid. The other 824 are always zero. The model minimizes average loss by learning to output near-zero everywhere — correctly predicting the 824 zeros at the expense of the 502 valid hands.

Verification on iteration 0 training data (ONNX model):

| Boundary | True CFV (first 3 hands) | Predicted CFV | Error |
|----------|-------------------------|---------------|-------|
| P1 B20 | [0.151, 0.152, 0.152] | [-0.023, -0.040, 0.036] | 0.36 chips max |
| P1 B22 | [0.151, 0.152, 0.152] | [-0.023, -0.041, 0.035] | 0.36 chips max |
| P1 B24 | [-0.216, -0.216, -0.216] | [-0.025, -0.017, -0.018] | 0.24 chips max |

The model predicts near-zero where true values are 0.15 or -0.22 chips. Overall player 1 mean error: 0.034 chips with worst boundary at 0.21 chips mean error.

Player 0 (OOP, 860/1326 valid) is less affected: mean error 0.010 chips, but still has worst boundary at 0.052 chips.

### Finding 3: Model Capacity Is NOT the Issue

Model has 2.4M params (5×512) for 6.1M valid output values. A 5×700 model would have 3.85M params — still under-parameterized on paper. But the real problem isn't capacity. The model could memorize the data if it weren't wasting capacity predicting zeros for blocked hands.

**Root cause**: MSE/Huber loss on 1326-dim output is dominated by the ~650 zero-valued blocked positions per sample. The model optimizes for the majority (zeros) at the expense of the minority (valid hand CFVs).

### Fix: Predict in Solver Indexing

For single-board overfit, canonical 1326 indexing provides no benefit. Predict directly in solver indexing:
- OOP: 860 output dims (all valid)
- IP: 502 output dims (all valid)
- Pad to max(860, 502) = 860, or use separate forward paths

This eliminates the sparse target problem — every output dimension carries meaningful signal.

---

## Experiment 3: Model v1 with Solver Indexing (2025-03-01)

**Changes from failed v1**:
- Output in solver indexing (860 dims max) instead of canonical 1326
- Masked HuberLoss — only compute on valid positions (860 for OOP, 502 for IP)
- Following train_huber.py conventions (Ranger21, OneCycleLR, EMA, HuberLoss)
- 7 layers × 500 hidden
- Input: boundary_onehot(25) + player(1) + cfreach_padded(860) = 886 dims
- Output: cfv_padded(860) dims

**Training**: RMSE dropped from 0.016 chips (canonical 1326) to **0.0006 chips** (solver indexing) — 27x improvement.

**Result: Working!**

| Metric | Canonical 1326 (failed) | Solver Indexing |
|--------|------------------------|-----------------|
| Training RMSE | 0.016 chips | 0.0006 chips |
| Root avg diff (Check) | ~35% | **0.60%** |
| Root avg diff (Bet18) | ~35% | **0.57%** |
| Root avg diff (AllIn) | ~35% | **0.04%** |
| Root max diff | 100% | **5.0%** (8cTc) |
| Full tree avg diff | 34.8% | **13.6%** |

Root strategy is excellent — worst hand (8cTc) at only 5.0% diff. Most hands within 2-3%.

**Key lesson**: The 1326-dim canonical output was the bottleneck, not model capacity or compounding error. When every output dimension carries real signal, even a modestly-sized model (7×500 = ~3M params) can memorize 9,000 samples and produce correct DCFR replay.

---

## Analysis: Where Does the 13.6% Full Tree Diff Come From? (2025-03-01)

Isolated the error sources by comparing three solutions:

| Comparison | Full Tree Avg Diff | What it measures |
|---|---|---|
| pairs2 vs standard | **0.000000%** | Iteration count (180 vs 200) |
| model-v1 vs pairs2 | **13.6%** | Pure model prediction error |
| model-v1 vs standard | **13.6%** | Total |

**Finding: 100% of the 13.6% diff comes from model prediction error compounding through the tree.** The iteration count mismatch (180 vs 200) contributes zero error — pairs2 (perfect CFVs, 180 iters) is byte-identical to standard (200 iters).

### Why 0.0006 RMSE compounds to 13.6%

The root node is shielded (only 0.6% avg diff) because it averages over many downstream paths. But deeper decision nodes closer to boundaries are hit harder:

1. Small CFV prediction error → slightly wrong regret update at a node
2. Wrong regret → wrong strategy → different cfreach next iteration
3. Different cfreach = out-of-distribution input for model → worse prediction
4. Error compounds over 180 iterations
5. Deep nodes have fewer paths averaging out errors → higher sensitivity

The model was trained on a **single trajectory** (9,000 samples). Once inference diverges even slightly, the model sees cfreach values it never trained on.

### Possible improvements

1. **Multi-trajectory training (DAgger-style)** — run multiple solves with perturbed strategies, collect diverse (cfreach, CFV) pairs so the model handles OOD inputs
2. **More training data** — run with more iterations (e.g. 500+) to get more samples near convergence
3. **Run more iterations at inference** — let DCFR self-correct beyond the 180 training iterations
4. **Better model fit** — reduce 0.0006 RMSE further

---

## Open Issue: Loss Curve Spike at Epoch ~360 (2025-03-01)

The training loss curve shows an unusual spike around epoch 360/500 — loss jumps ~50x (from 9.4e-6 to 4.8e-4) then slowly recovers.

**Suspected cause**: Ranger21 has a built-in warm-down that starts at `72% × 500 = epoch 360` (hardcoded `warmdown_start_pct=0.72`). This internal LR schedule likely conflicts with the external OneCycleLR, both fighting over `group["lr"]`.

**Attempted fixes** (both failed — solver output was worse despite smoother loss curves):

1. **Remove OneCycleLR, let Ranger21 handle scheduling**: Smoother loss curve, but solver result was completely wrong.
2. **Disable Ranger21's internal scheduling (`use_warmup=False, warmdown_active=False`), keep OneCycleLR**: Also produced wrong solver output.

**Current status**: Reverted to original code with both Ranger21 + OneCycleLR. Despite the ugly spike, this configuration produces the best solver results (0.6% root avg diff, 13.6% full tree). The spike remains unexplained in terms of why the conflicting schedulers produce better results than either one alone.

**Remains to investigate.**

---

**Next steps**: Bucketing (K-dim output) for multi-board generalization — same principle as solver indexing (no wasted zeros) but board-agnostic.
