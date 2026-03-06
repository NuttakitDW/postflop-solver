# DEVN Standard — Implementation Plan

**Paper:** "Don't Predict Counterfactual Values, Predict Expected Values Instead" (AAAI 2023)
**Codename:** DEVN Standard (separate from RL1 — independent scripts, independent model checkpoints)
**Goal:** Apply the DEVN insight (predict EV, not CFV) to our online training pipeline.
**Test config:** `config/test_small.json` (9s6d6c, 9 OOP hands, 11 IP hands, 500 iters)

---

## Naming Convention

| Approach | Training Script | Inference Script | Model Dir | Description |
|----------|----------------|-----------------|-----------|-------------|
| RL1 | `trainings/train_rl1.py` | `trainings/solve_with_rl1.py` | `models/rl1_<config>/` | Predict CFV directly |
| **DEVN Standard** | `trainings/train_devn.py` | `trainings/solve_with_devn.py` | `models/devn_<config>/` | Predict EV, multiply by matchup |

RL1 and DEVN Standard are **completely independent**. They share the same Rust solver
and PyO3 bridge but have separate scripts, separate model checkpoints, and separate output files.
This allows clean head-to-head comparison.

---

## Core Idea

The paper factors CFV into two components:

```
CFV[b][h] = EV[b][h] × matchup[b][h]
```

- **matchup[b][h]** = total opponent reach at boundary b for non-conflicting hands with hero hand h
  (can be computed exactly from opponent cfreach + card conflict matrix)
- **EV[b][h]** = expected value at boundary b assuming opponent enters the state
  (smooth, easier for NN to learn)

**Instead of training NN to predict CFV directly (RL1), train it to predict EV.**
At inference, multiply predicted EV by matchup to recover CFV for the solver.

---

## Why This Should Help

1. **Smoother targets:** EVs vary slowly across similar hands. CFVs are noisy because they're
   scaled by opponent reach (which varies wildly per hand).
2. **Better learning signal:** The NN doesn't waste capacity modeling the matchup multiplier
   that we can compute analytically.
3. **Paper results:** 9-16% better CFV prediction. 2-layer DEVN ≥ 7-layer DCVN.

---

## What Changes

### Files to Modify

| File | Change |
|------|--------|
| `pyo3-bridge/src/lib.rs` | Add `private_cards(player)` method to `GameWrapper` |
| `trainings/train_devn.py` | **New file** — DEVN Standard training script |
| `trainings/solve_with_devn.py` | **New file** — DEVN Standard inference script |

### Files NOT Modified

- `src/solver.rs` — no Rust solver changes needed
- `src/file.rs` — .flop format unchanged
- `trainings/train_rl1.py` — **untouched**, kept as independent baseline
- `trainings/solve_with_rl1.py` — **untouched**, kept as independent baseline

---

## Implementation Steps

### Step 1: Expose `private_cards` via PyO3 (Rust, ~10 min)

Add to `GameWrapper` in `pyo3-bridge/src/lib.rs`:

```rust
/// Return private cards for a player as list of (card1, card2) tuples.
/// Each card is a u8 (0-51). Used to build card conflict matrix in Python.
fn private_cards(&self, player: usize) -> Vec<(u8, u8)> {
    self.game.private_cards(player)
        .iter()
        .map(|&(c1, c2)| (c1, c2))
        .collect()
}
```

Rebuild: `cd pyo3-bridge && maturin develop --release`

### Step 2: Build Card Conflict Matrix (Python, ~10 min)

One-time computation at init. For each (hero_hand, opp_hand) pair, check if they share a card:

```python
def build_conflict_matrix(hero_cards, opp_cards):
    """Returns bool matrix [num_hero_hands, num_opp_hands].
    True = conflict (share at least one card)."""
    nh = len(hero_cards)
    no = len(opp_cards)
    conflict = np.zeros((nh, no), dtype=bool)
    for i, (h1, h2) in enumerate(hero_cards):
        for j, (o1, o2) in enumerate(opp_cards):
            if h1 == o1 or h1 == o2 or h2 == o1 or h2 == o2:
                conflict[i, j] = True
    return conflict
```

### Step 3: Compute Matchups from Cfreaches (Python, ~5 min)

Given opponent cfreach vector at a boundary and conflict matrix:

```python
def compute_matchups(cfreaches, conflict_matrix):
    """Compute matchup[b][h] = sum of non-conflicting opponent cfreaches.

    Args:
        cfreaches: list of np arrays, each [num_opp_hands]
        conflict_matrix: [num_hero_hands, num_opp_hands] bool

    Returns:
        matchups: np array [num_boundaries, num_hero_hands]
    """
    opp = np.stack(cfreaches)                     # [B, num_opp_hands]
    valid = ~conflict_matrix                       # [num_hero_hands, num_opp_hands]
    matchups = opp @ valid.T                       # [B, num_hero_hands]
    return matchups
```

### Step 4: Create `train_devn.py` (Python, ~30 min)

Standalone DEVN Standard training script (not a fork — clean separate file that shares no
imports with RL1). Key differences from RL1:

**Init (one-time):**
```python
# Build conflict matrices for both players
hero_cards = [game.private_cards(0), game.private_cards(1)]
opp_cards = [game.private_cards(1), game.private_cards(0)]
conflict = [
    build_conflict_matrix(hero_cards[0], opp_cards[0]),  # OOP hero vs IP opp
    build_conflict_matrix(hero_cards[1], opp_cards[1]),   # IP hero vs OOP opp
]
```

**Training loop — replace lines 224-243 of train_rl1.py:**
```python
for player in range(2):
    # 1. Collect opponent cfreaches at boundaries
    cfreaches = game.collect_boundary_cfreaches(player)

    # 2. Compute matchups for hero hands
    matchups = compute_matchups(cfreaches, conflict[player])  # [B, num_hero_hands]
    matchup_t = torch.tensor(matchups, device=dev)             # [B, num_hero_hands]

    # 3. Build model input (same as RL1)
    inputs = build_input_batch(cfreaches, player, num_boundaries, max_hands, dev)

    # 4. Model predicts EV (not CFV)
    predicted_ev = model(inputs)  # [B, max_hands]

    # 5. Convert predicted EV → predicted CFV for the solver
    nh = num_hands[player]
    matchup_padded = torch.zeros(len(cfreaches), max_hands, device=dev)
    matchup_padded[:, :nh] = matchup_t
    predicted_cfv = predicted_ev * matchup_padded
    model_cfvs = tensor_to_cfv_lists(predicted_cfv, player, num_hands)

    # 6. Solver returns true CFVs
    true_cfvs = game.solve_step_with_model(t, player, model_cfvs)

    # 7. Convert true CFV → true EV (training target)
    targets_cfv, mask = build_target_batch(true_cfvs, player, max_hands, num_hands, dev)
    eps = 1e-8
    safe_matchup = matchup_padded.clamp(min=eps)
    targets_ev = targets_cfv / safe_matchup

    # 8. Mask out hands with near-zero matchup (unreachable)
    matchup_mask = (matchup_padded > 1e-6).float()
    final_mask = mask * matchup_mask

    # 9. Loss on EV predictions
    loss = masked_mse(predicted_ev, targets_ev, final_mask)
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()
```

**Key insight:** When matchup ≈ 0 (opponent can't reach with hands conflicting hero's),
the EV is undefined. We mask these out — the NN doesn't need to learn them.

### Step 5: Create `solve_with_devn.py` (Python, ~15 min)

Standalone DEVN Standard inference script. Key difference from RL1 inference:

```python
with torch.no_grad():
    for t in range(max_iters):
        for player in range(2):
            cfreaches = game.collect_boundary_cfreaches(player)
            matchups = compute_matchups(cfreaches, conflict[player])
            matchup_t = torch.tensor(matchups, device=dev)

            inputs = build_input_batch(cfreaches, player, num_boundaries, max_hands, dev)
            predicted_ev = model(inputs)

            # EV → CFV
            nh = num_hands[player]
            matchup_padded = torch.zeros(len(cfreaches), max_hands, device=dev)
            matchup_padded[:, :nh] = matchup_t
            predicted_cfv = predicted_ev * matchup_padded

            model_cfvs = tensor_to_cfv_lists(predicted_cfv, player, num_hands)
            game.solve_step_replay(t, player, model_cfvs)
```

---

## Testing Plan

### Test 1: Matchup Sanity Check (~5 min)

Before training, verify matchups are correct:

```python
game = postflop_solver.GameWrapper("config/test_small.json")
cfreaches = game.collect_boundary_cfreaches(0)
hero_cards = game.private_cards(0)
opp_cards = game.private_cards(1)
conflict = build_conflict_matrix(hero_cards, opp_cards)
matchups = compute_matchups(cfreaches, conflict)

# Check: all matchups should be >= 0
# Check: matchups should be < sum(cfreach) (some hands blocked)
# Check: no NaN or Inf
for b in range(len(cfreaches)):
    total = sum(cfreaches[b])
    print(f"Boundary {b}: matchup range [{matchups[b].min():.4f}, {matchups[b].max():.4f}], "
          f"total cfreach={total:.4f}")
```

### Test 2: Oracle Validation (~10 min)

Run one episode of normal DCFR (via `solve_step_recording`) and verify:
- `true_cfv / matchup` gives reasonable EVs (no extreme values)
- `ev * matchup` recovers original CFV exactly

```python
game.reset()
for t in range(500):
    for player in range(2):
        cfreaches = game.collect_boundary_cfreaches(player)
        matchups = compute_matchups(cfreaches, conflict[player])
        true_cfvs = game.solve_step_recording(t, player)

        for b, cfv in enumerate(true_cfvs):
            cfv_arr = np.array(cfv)
            m = matchups[b]
            valid = m > 1e-8
            ev = np.where(valid, cfv_arr / m, 0.0)
            recovered = ev * m
            max_diff = np.max(np.abs(cfv_arr[valid] - recovered[valid]))
            assert max_diff < 1e-6, f"Recovery failed: {max_diff}"
print("Oracle validation passed!")
```

### Test 3: DEVN Training on test_small.json (~30 min)

```bash
/opt/anaconda3/bin/python trainings/train_devn.py config/test_small.json --episodes 50
```

**Success criteria:**
- Exploitability decreases over episodes
- Final exploitability comparable to or better than RL1 on same config
- Loss curve shows learning (not stuck)

### Test 4: Head-to-Head Comparison (~1 hr)

Run both RL1 and DEVN with identical settings, compare:

```bash
# Baseline
/opt/anaconda3/bin/python trainings/train_rl1.py config/test_small.json --episodes 50

# DEVN
/opt/anaconda3/bin/python trainings/train_devn.py config/test_small.json --episodes 50
```

Compare:
- Exploitability curves (DEVN should converge faster or to lower value)
- Per-episode loss (not directly comparable — different targets)
- Training time per episode (DEVN adds matchup computation overhead — should be negligible)

### Test 5: Inference + Save .flop (~5 min)

```bash
/opt/anaconda3/bin/python trainings/solve_with_devn.py config/test_small.json models/devn_test_small/best.pt
```

Verify:
- .flop file is created successfully
- File can be loaded by the viewer (if available)
- Exploitability of saved solution matches training-time measurement

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Division by zero (matchup=0) | Clamp matchup to eps=1e-8, mask out in loss |
| Large EV values when matchup tiny | Mask out hands with matchup < 1e-6 from loss |
| NN output scale differs (EVs vs CFVs) | EVs are generally larger magnitude than CFVs; may need to tune LR. Start with same LR and adjust if loss diverges. |
| Card conflict matrix wrong | Test 1 + Test 2 verify correctness before training |
| Online EV target noisy early on | Same issue as RL1 with CFVs — online training inherently handles this |

---

## Timeline (1 day)

| Time | Task |
|------|------|
| 30 min | Step 1-2: Rust `private_cards` + rebuild + conflict matrix |
| 1 hr | Step 3-5: Write `train_devn.py` + `solve_with_devn.py` |
| 15 min | Test 1-2: Sanity checks |
| 2 hr | Test 3: Training run on test_small.json (mostly waiting) |
| 30 min | Test 4-5: Comparison + inference |
| **~4.5 hr** | **Total** |

---

## What This Does NOT Include

- Card abstraction / bucketing (paper's main scenario — not needed for our hand-level approach)
- DeepStack-style depth-limited search (orthogonal to our DCFR approach)
- Multi-board generalization (future work)
- RL2 integration (DEVN insight could apply there too, but separate task)

---

## Iteration Relevance

Same as RL1. The iteration number `t`:
- **IS** passed to the Rust solver for DCFR discount weighting
- **IS NOT** a model input (proven failure #4 — adding it makes things worse)
- EVs at boundaries still evolve per-iteration as turn/river strategies change
- The model handles this implicitly through the changing cfreaches

DEVN Standard changes only the **target** (EV vs CFV), not the iteration structure.
The online training loop is identical in shape to RL1.
