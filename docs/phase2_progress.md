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

### Verification: bt1 data is correct

Created `solve_with_bt1` — replays bt1's CFVs directly (ignoring cfreach and the model), like pairs_v2 does with dpairs2. Result: **byte-identical to standard solve**. The bt1 recording pipeline is correct.

### Verification: cfreach matches perfectly between bt1 and inference

Created `debug_bt1_vs_inference` — stays on the training trajectory (using bt1's true CFVs for replay) while comparing:
1. cfreach from `collect_boundary_cfreaches` vs bt1's recorded cfreach
2. Model predictions vs bt1's true CFVs

**cfreach matches at ALL iterations (0.00000000 diff).** The input pipeline is correct.

### Finding: Model prediction error is non-uniform across iterations

The model's 0.0006 RMSE is an average. Per-iteration breakdown on the training trajectory:

| Iteration | Player | Mean Error | Max Error | RMSE |
|---|---|---|---|---|
| 0 | IP | 0.001143 | **0.024** chips | 0.002494 |
| 2 | IP | 0.001456 | **0.043** chips | 0.002693 |
| 10 | IP | 0.001218 | 0.006 chips | 0.001566 |
| 50 | IP | 0.000448 | 0.004 chips | 0.000668 |
| 179 | IP | 0.000306 | 0.002 chips | 0.000401 |

Early iterations (0-10) have 10-20x worse max errors than late iterations. These are exactly where DCFR strategies are most volatile — errors here compound through the rest of the solve.

### Why early iterations are harder to fit

Early DCFR iterations have wildly fluctuating CFV values because strategies are unstable (alpha ≈ 0, regrets barely carry over). Late iterations converge and have smooth, predictable CFVs. The model spreads its capacity uniformly across all 9,000 samples, under-fitting the noisy early iterations.

### Attempted fix: Iteration-weighted loss (failed)

Weighted the training loss by DCFR's gamma discount `t/(t+1)` — giving low weight to early iterations (which DCFR discounts anyway) and high weight to late iterations (where accuracy matters most). Result: worse solver output.

The idea was sound (DCFR does discount early iterations) but the model still needs reasonable predictions at early iterations to avoid trajectory divergence during inference. Under-fitting early iterations makes the compounding worse, not better.

### Remaining ideas

1. **Multi-trajectory training (DAgger-style)** — run multiple solves with perturbed strategies, collect diverse (cfreach, CFV) pairs so the model handles OOD inputs
2. **More training data** — run with more iterations (e.g. 500+) to get more samples near convergence
3. **Run more iterations at inference** — let DCFR self-correct beyond the 180 training iterations
4. **Better model fit** — reduce early-iteration errors specifically (e.g. larger model, curriculum learning)

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

## Experiment 4: Adding Iteration Number as Model Input (2025-03-01)

**Hypothesis**: The lookup table is indexed by `(iteration, boundary, player)` while the model only sees `(boundary, player, cfreach)`. Adding normalized iteration number (t / max_t → [0,1]) as an input feature should help the model distinguish early vs late iterations and predict iteration-appropriate CFVs.

**Changes**: Input becomes `[boundary_onehot(25), player(1), iter_norm(1), cfreach(860)] = 887 dims`.

**Result**: Made things worse.

**Why it failed**: Adding iteration number makes the model memorize the exact training trajectory even harder — it learns "at iteration 42, boundary 7, the CFV should be exactly X." During inference, the first tiny prediction error causes strategy drift, so by iteration 43 the cfreach is slightly different from training. The model now sees the "right" iteration but "wrong" cfreach — a contradiction it never encountered in training. This makes predictions less stable, not more.

Without iteration number, the model at least tries to learn a general `cfreach → CFV` mapping that has some robustness to drift. With iteration number, it anchors to the specific trajectory and becomes MORE brittle to any deviation.

**Key insight**: The problem is not missing features — it's that the model operates in a closed-loop feedback system where its own errors change its future inputs. Adding more features that overfit to the training trajectory makes the distribution shift problem worse, not better.

---

## Experiment 5: Removing Regularization for True Memorization (2025-03-01)

**Hypothesis**: The model's 0.0006 RMSE isn't zero — it still has max errors of 0.024-0.043 chips at early iterations. The training recipe (weight decay, EMA, Ranger21, OneCycleLR) actively prevents perfect memorization. If we remove all regularization, the model should achieve near-zero training error and produce solver results matching the lookup table.

**Changes** (train_bt1_overfit.py):
- Adam optimizer (no built-in regularization) instead of Ranger21
- Weight decay = 0 (was 1e-4)
- Raw weights saved (no EMA smoothing)
- MSE loss (was Huber — stronger gradient signal)
- ReduceLROnPlateau (was OneCycleLR — keeps LR high until stuck)
- Up to 2000 epochs with early stopping

**Training result**: RMSE dropped from 0.0006 to **0.000313 chips** — better training fit.

**Solver result**: Catastrophically worse.

| Metric | Regularized (EMA) | Overfit (no reg) |
|--------|-------------------|------------------|
| Training RMSE | 0.0006 chips | **0.000313 chips** |
| Root avg diff | **0.6%** | 42.2% |
| Full tree avg diff | **13.6%** | 19.1% |
| Root max diff | 5.0% | 99.3% |

The overfit model memorized training data better but produced completely wrong strategies — most JT combos at 95-99% diff.

**Root cause: NOT underfitting.** The regularized model's smoothness is actually essential:

- **Smooth model** (EMA + weight decay): small cfreach perturbation → small CFV change → trajectory stays close
- **Sharp model** (no regularization): small cfreach perturbation → large CFV change → trajectory diverges immediately

The lookup table avoids this entirely because it's indexed by iteration number — cfreach perturbations have zero effect on its output. The NN uses cfreach as input, so it's inherently sensitive to perturbations. Regularization limits that sensitivity.

**Key finding**: The 0.6% root / 13.6% full tree diff with the regularized model is likely near the best achievable for single-trajectory cfreach→CFV mapping. The remaining error comes from the fundamental sensitivity of the feedback loop to cfreach-dependent predictions, not from model capacity or training quality.

---

## Experiment 6: Replacing Boundary One-Hot with (pot, stack) for Generalization (2025-03-01)

**Goal**: Enable one model to handle different flop bet size configurations without retraining. Turn/river bet sizes are fixed (abstraction). If we change flop bet sizes, different boundaries exist — but boundaries sharing the same (pot, stack) should have identical turn/river subtrees.

**Hypothesis**: Replace boundary one-hot encoding (config-specific, 25 dims) with continuous (pot_norm, stack_norm) features (2 dims, generalizable). Normalization: `pot / max_pot`, `stack / max_pot` where `max_pot = starting_pot + 2 * effective_stack`.

**Changes**:
- `trainings/train_bt1.py`: Input changed from `[boundary_onehot(25), player(1), cfreach(860)] = 886 dims` to `[pot_norm(1), stack_norm(1), player(1), cfreach(860)] = 863 dims`
- `examples/solve_with_model_v1/main.rs`: Updated to read `max_pot` from meta.json, use `collect_boundary_pot_stack()` and build input with `(pot_norm, stack_norm)` instead of boundary one-hot
- `src/solver.rs`: Added `collect_boundary_pot_stack()` function
- `examples/dump_boundaries.rs`: Added stack info to JSON output
- `data/bt1/KcQh7s_boundaries.json`: Generated boundary pot/stack metadata

**Training result**: Identical to one-hot — RMSE 0.0006 chips, same training accuracy.

**Solver result**: Significantly worse.

| Metric | One-hot (working) | Pot/stack |
|--------|-------------------|-----------|
| Training RMSE | 0.0006 chips | 0.0006 chips |
| Root avg diff (Check) | 0.60% | **9.7%** |
| Root avg diff (Bet18) | 0.57% | **9.0%** |
| Root avg diff (AllIn) | 0.04% | **0.7%** |
| Full tree avg diff | 13.6% | **18.0%** |
| Full tree >10% | ~30% | **39.2%** |

**Root cause**: KcQh7s has 25 boundaries but only 10 unique (pot, stack) pairs. **13 boundaries share (pot=415, stack=0)** — all the all-in action paths. With one-hot, the model could learn distinct CFV mappings for each. With (pot, stack), these 13 boundaries are indistinguishable (same input features except cfreach), forcing the model to produce the same output for the same cfreach.

Theoretically, boundaries with the same (pot, stack) SHOULD have the same CFV for the same cfreach (identical turn/river subtree). But in practice, the reduced discriminative power of 2 continuous features vs 25 one-hot features compounds through the feedback loop, producing larger trajectory divergence.

**Conclusion**: The (pot, stack) encoding loses too much boundary-specific information for the feedback loop to stay stable. Generalization across flop configs needs a different approach — perhaps combining (pot, stack) with additional structural features, or training on multi-config data where the diversity helps the model learn the true underlying mapping.

---

---

## Experiment 7: Iter-Only Model — Testing NN Precision Limits (2025-03-01)

**Hypothesis**: The 13.6% full tree diff comes from cfreach feedback loop compounding. If we remove cfreach entirely and use only `(boundary, player, iteration)` as input, the model becomes a pure lookup table with no feedback loop — it should produce ~0% diff if it memorizes perfectly.

**Setup**: Input = `[boundary_onehot(25), player(1), iteration_onehot(180)] = 206 dims`. No cfreach input. Code: `trainings/train_bt1_iteronly.py`, `examples/solve_with_model_iteronly/`.

**Key discovery**: Iteration encoding matters hugely for memorization.

| Encoding | Model Size | RMSE (chips) | Full Tree Diff |
|----------|-----------|-------------|----------------|
| iter_norm (continuous), H=500 | 2M params | 0.0028 | 14.85% |
| iter_norm, H=1500 (Ranger+EMA) | 15M params | 0.0012 | 12.09% |
| iter_norm, H=1500 (Adam full-batch) | 15M params | 0.0042 | not tested |
| **iter_onehot, H=1500 (Ranger+EMA)** | **15M params** | **0.000195** | **10.56%** |
| dpairs2 (exact float32 lookup) | N/A | 0.000000 | 0.0002% |

Iteration one-hot gave 6x better RMSE than continuous iter_norm (0.000195 vs 0.0012). The network can distinguish 180 discrete iterations much better when they're categorically encoded.

**ONNX verification (best model)**:
- OOP: mean_err=0.000087, max_err=0.000554, p99=0.000310 chips
- IP: mean_err=0.000131, max_err=0.000829, p99=0.000411 chips

**Result**: Even with near-perfect memorization (0.000087 chips mean error), full tree diff is still 10.56%.

**Root cause: DCFR regret matching hypersensitivity.** Near equilibrium, many actions have cumulative regrets close to zero. Strategy is computed via regret matching: `strategy[a] = max(0, cumR[a]) / Σ max(0, cumR)`. Even a 0.0001 chip error per iteration, compounded over 180 iterations, can flip the sign of a cumulative regret (e.g., +0.001 → -0.001), changing that action from a small positive probability to exactly 0%.

The gap between NN (10.56%) and float32 lookup (0.0002%) is not closable — neural networks fundamentally cannot achieve float32-exact precision across 6.1M output values.

**Conclusion**: NN precision alone cannot eliminate the tree diff. The ~10-14% full tree diff is a **fundamental limit** of approximate boundary CFV methods with DCFR. Future work needs either:
1. Accept approximate strategies (the root strategy is already excellent at 0.2% diff)
2. Make the solver robust to CFV noise (modified regret matching, averaging, etc.)
3. Focus on multi-board generalization rather than single-board perfection

---

## Experiment 7b: Adding Iteration One-Hot to cfreach Model (2025-03-01)

**Hypothesis**: Since iteration one-hot dramatically improved the iter-only model, adding it to the cfreach model should help too — the model would know which iteration it's predicting for, making the cfreach→CFV mapping easier.

**Changes**: Input becomes `[boundary_onehot(25), player(1), iter_onehot(180), cfreach(860)] = 1066 dims`.

**Training result**: Same RMSE (0.0006 chips) — iteration info doesn't improve training fit when cfreach is already present.

**Solver result**: Catastrophically worse.

| Metric | cfreach only | cfreach + iter_onehot |
|--------|-------------|----------------------|
| Root avg diff | 0.6% | **56.2%** |
| Full tree avg diff | 13.6% | **24.6%** |
| Root max diff | 5.0% | **99.1%** |

**Root cause: Overfitting to the training trajectory.** With iteration one-hot, the model learns: "at iteration 37, with THIS exact cfreach → predict THIS exact CFV." During solving, cfreach drifts from training data (feedback loop), but the iteration index stays correct. The model sees the "right" iteration but "wrong" cfreach — a contradiction it never saw in training. Without iteration info, the model learns a more general cfreach→CFV mapping that tolerates cfreach drift better.

**Key insight**: Iteration info helps **memorization** (iter-only model) but hurts **generalization** (cfreach model in a feedback loop). This is consistent with Experiment 4 (iter_norm also hurt).

**Conclusion**: Reverted. The cfreach model must NOT include iteration information — it needs to generalize to drifted cfreaches, not memorize the training trajectory.

---

**Current best result (Experiment 3)**: 0.6% root avg diff, 13.6% full tree avg diff.

**Next steps**:
- Multi-trajectory training (DAgger-style) to improve cfreach model's robustness to distribution shift
- Accept ~10-14% tree diff as inherent to approximate CFV methods — focus on multi-board generalization
- Investigate modified DCFR variants that are less sensitive to boundary CFV noise
