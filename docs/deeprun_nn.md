# DeepRun NN

## Overview

DeepRun NN is a neural network approach that learns to predict poker flop strategy directly from solved games. Given a hand and game context, the NN outputs action probabilities — replacing the need to store or re-solve the game tree at the flop.

## Key Insight: Strategy is Decomposable Per Hand

Previous approaches (Phase 2 Experiments 1-8) tried to predict **boundary CFVs** during CFR iteration. This failed because:
- CFV depends on the entire range (not decomposable per hand)
- NN operates in a feedback loop where its errors change future inputs (distribution shift)
- Even 0.0001 chip error compounds over 200 iterations

DeepRun NN sidesteps all of this by predicting the **final converged strategy** — not CFV, not during iteration, but the end result:

| Property | Boundary CFV (Phase 2) | DeepRun NN |
|---|---|---|
| What NN predicts | CFV vector (range-coupled) | Action probabilities (per-hand) |
| When NN runs | During CFR iterations | After CFR (or standalone) |
| Feedback loop | Yes (errors compound) | No (one-shot prediction) |
| Distribution shift | Fatal problem | Does not exist |
| Per-hand decomposition | Invalid | Valid |

## Architecture

```
Input → Linear(H) → ReLU → Linear(H) → ReLU → Linear(H) → ReLU → [Linear(H) → ReLU →] Linear(A) → Softmax
```

- **Single board**: 3 hidden layers, H=256
- **Multi board**: 4 hidden layers, H=512

### Input Encoding

| Feature | Dims | Description |
|---------|------|-------------|
| Hand | 104 | One-hot card1 (52) + card2 (52) |
| Board | 156 | One-hot flop card1 (52) + card2 (52) + card3 (52). Multi-board only. |
| Player | 1 | 0=OOP, 1=IP |
| Action history | MAX_HISTORY × MAX_ACTIONS | One-hot action taken at each tree depth |
| Pot fraction | 1 | Current pot / max pot, normalized |

Single board: 130 dims. Multi board: 286 dims.

### Output

Softmax over actions (padded to max_actions across all nodes). Probabilities sum to 1.

Example: `[Check=0.514, Bet(18)=0.486, AllIn(180)=0.000]`

## Pipeline

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│ CFR Solve        │────▶│ Train NN     │────▶│ Inference    │
│ (slow, one-time) │     │ (extract     │     │ (instant,    │
│ → .flop file     │     │  strategies, │     │  per hand)   │
│                  │     │  fit model)  │     │              │
└─────────────────┘     │ → .pt file   │     └──────────────┘
                        └──────────────┘
```

### Step 1: Solve with CFR

Use the existing Rust solver to produce a `.flop` file:

```bash
make start CONFIG=config/2c3c4h_p2.json
```

### Step 2: Train DeepRun NN

**Single board:**
```bash
/opt/anaconda3/bin/python trainings/demo_nn_flop.py data/out/2c3c4h_p2.flop
```

Output:
- `models/demo/2c3c4h_p2_flop.pt` — trained model
- `models/demo/2c3c4h_p2_flop_loss.png` — loss curve

**Multi board (one model for multiple spots):**
```bash
/opt/anaconda3/bin/python trainings/demo_nn_flop_multi.py \
  data/out/2c3c4h_p2.flop \
  data/out/7s6s4c_p2.flop \
  data/out/Ad8s2c_p2.flop
```

Output:
- `models/demo/2c3c4h_7s6s4c_Ad8s2c_flop.pt`
- `models/demo/2c3c4h_7s6s4c_Ad8s2c_flop_loss.png`

### Step 3: Inference

**Query a specific hand:**
```bash
/opt/anaconda3/bin/python trainings/infer_nn_flop.py models/demo/test_small_flop.pt KcKd
```

**Build a .flop file from NN (flop from NN, turn/river uniform):**
```bash
/opt/anaconda3/bin/python trainings/build_flop_from_nn.py \
  config/2c3c4h_p2.json \
  models/demo/2c3c4h_7s6s4c_Ad8s2c_flop.pt
```

Output: `data/out/2c3c4h_p2-nn.flop`

## Results

### Single Board (test_small: 9s6d6c, 46 OOP hands, 53 IP hands)

| Metric | Value |
|--------|-------|
| Flop nodes | 16 |
| Training samples | 792 |
| Model size | 655 KB |
| Original .flop size | 229 MB |
| Compression ratio | **300x** |
| Max strategy diff | 0.002 |
| Training time | ~10 seconds |

### Multi Board (3 boards: 2c3c4h, 7s6s4c, Ad8s2c)

| Metric | Value |
|--------|-------|
| Flop nodes | 72 (24 per board) |
| Training samples | 51,300 |
| Model params | 937K |
| Max diff (2c3c4h) | 0.025 |
| Max diff (7s6s4c) | 0.035 |
| Max diff (Ad8s2c) | 0.027 |
| Training time | ~2 minutes |

## Comparison with Previous Approaches

| Approach | Root Diff | Full Tree Diff | Status |
|----------|-----------|----------------|--------|
| Phase 2 Exp 1: Static Oracle | 57% | 27% | Failed |
| Phase 2 Exp 2: Model v1 (1326-dim) | 35% | 35% | Failed |
| Phase 2 Exp 3: Model v1 (solver idx) | 0.6% | 13.6% | Best CFV approach |
| Phase 2 Exp 7: Iter-only (no cfreach) | - | 10.6% | NN precision limit |
| Phase 2 RL1: Online training | 1.77% exploitability | - | In progress |
| **DeepRun NN** | **< 0.1%** | **N/A (flop only)** | **Working** |

Note: DeepRun NN only covers flop strategy. Turn/river require separate handling (CFR, or future extension).

## Limitations

1. **Requires CFR solve first** — DeepRun NN is compression, not a solver replacement (yet)
2. **Fixed range** — the model is trained for a specific preflop range. Different ranges need retraining.
3. **Flop only** — turn/river strategies are not predicted (set to uniform in .flop output)
4. **Memorization vs generalization** — current results are on training boards. Generalization to unseen boards is the next step.

## Future Directions

1. **Generalization**: Train on 100+ boards, test on unseen boards
2. **Card embeddings**: Replace one-hot with learned embeddings for better generalization
3. **Turn/river extension**: Apply the same per-hand strategy prediction to turn and river nodes
4. **Combined with RL1**: Use DeepRun NN for fast flop initialization, RL1 for online refinement

## Files

| File | Purpose |
|------|---------|
| `trainings/demo_nn_strategy.py` | Train on single spot (root only) |
| `trainings/demo_nn_flop.py` | Train on all flop nodes (single board) |
| `trainings/demo_nn_flop_multi.py` | Train on all flop nodes (multi board) |
| `trainings/infer_nn_strategy.py` | Infer single spot |
| `trainings/infer_nn_flop.py` | Infer all flop nodes |
| `trainings/build_flop_from_nn.py` | Build .flop file from NN model |
| `pyo3-bridge/src/lib.rs` | PyO3 bridge with tree navigation + strategy lock |
