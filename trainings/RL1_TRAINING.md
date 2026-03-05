# RL1 Multi-Board Training Guide

## Goal

Train NN models to predict boundary CFVs (counterfactual values) for a postflop poker solver. Each model replaces the expensive turn/river subtree traversal during DCFR iterations, predicting the CFV values that would have been computed at flop-to-turn boundary nodes.

**Target: < 0.5% exploitability** (of starting pot) on each board.

## How It Works

The solver uses DCFR (Discounted Counterfactual Regret Minimization) to find Nash equilibrium strategies. The game tree has boundary nodes where the flop transitions to the turn. Normally, the solver must traverse the full turn+river subtree to compute CFVs at these boundaries.

RL1 trains a neural network to predict these boundary CFVs online:

1. **Each episode** = one full DCFR solve (2000 iterations)
2. **Each iteration**, for each player:
   - Collect counterfactual reach probabilities (cfreaches) at all boundary nodes
   - NN predicts CFVs from cfreaches
   - Rust solver uses predicted CFVs for flop regret updates, computes true CFVs from full subtree traversal
   - NN trains on (predicted vs true) CFV loss with immediate gradient update
3. **After each episode**, compute exploitability of the resulting strategy
4. Training stops when exploitability < 0.5% of pot

Key insight: online training avoids distribution shift — the model always trains on its own trajectory, so cfreaches at inference match training.

## Architecture

- 7-layer MLP: Linear → LayerNorm → GELU (x7) → Linear
- Hidden dim: 500
- Input: [boundary_one_hot | player_flag | cfreaches] (varies per config)
- Output: CFV vector (max_hands dimensions)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4, grad_clip=5.0)

## Config Files

All configs use `_p2` suffix. Shared settings across all 10 boards:

| Setting | Value |
|---------|-------|
| Starting pot | 55 |
| Effective stack | 180 |
| Bet sizes | 33%, 75% pot (flop/turn/river bets) |
| Raises | 33% pot |
| Donk bets | 33% pot |
| Max iterations | 2000 |
| Target exploitability | 0.5% |
| Max raises/street | 4 |

The 10 board configs:

```
config/9s6d6c_p2.json
config/KcQh7s_p2.json
config/2c3c4h_p2.json
config/7s6s4c_p2.json
config/9s8s2c_p2.json
config/Ad8s2c_p2.json
config/As3s4s_p2.json
config/AsKcTh_p2.json
config/Jc8c6s_p2.json
config/Kc8h7h_p2.json
```

All boards use the same OOP/IP ranges (standard BB vs BTN single-raised pot).

## Prerequisites

### Build the Rust PyO3 Bridge

```bash
cd pyo3-bridge
maturin develop --release
```

This builds the `postflop_solver` Python module that wraps the Rust solver.

### Python Dependencies

```
torch
numpy
matplotlib
```

Python 3.12+ required.

## Training Commands

Train each board individually:

```bash
# Basic — runs 50 episodes by default
python trainings/train_rl1.py config/9s6d6c_p2.json

# More episodes (recommended — some boards need 200+)
python trainings/train_rl1.py config/9s6d6c_p2.json --episodes 300

# With early stopping target (default 0.5%)
python trainings/train_rl1.py config/9s6d6c_p2.json --episodes 300 --target 0.5

# Resume from checkpoint
python trainings/train_rl1.py config/9s6d6c_p2.json --episodes 200 --resume models/rl1_9s6d6c_p2/latest.pt
```

### Train All 10 Boards

```bash
for config in config/*_p2.json; do
    echo "=== Training: $config ==="
    python trainings/train_rl1.py "$config" --episodes 300 --target 0.5
done
```

Or run in parallel (if enough memory — each job uses ~2-4 GB):

```bash
for config in config/*_p2.json; do
    python trainings/train_rl1.py "$config" --episodes 300 --target 0.5 &
done
wait
```

## Output

Each training run saves to `models/rl1_<config_name>/`:

```
models/rl1_9s6d6c_p2/
  latest.pt    — most recent checkpoint (resume-able)
  best.pt      — lowest exploitability checkpoint
  curves.png   — loss + exploitability plots (4 panels)
  log.csv      — per-episode: avg_loss, exploitability_chips, exploitability_pct
```

Checkpoint contents:
- `model` — state dict
- `optimizer` — optimizer state (for resuming)
- `episode` — episode number
- `losses` — list of per-episode avg losses
- `exploits` — list of per-episode exploitability values
- `config` — config path used

## Inference (Solve with Trained Model)

After training, use the model to solve and export a `.flop` file:

```bash
python trainings/solve_with_rl1.py config/9s6d6c_p2.json models/rl1_9s6d6c_p2/best.pt
```

Output goes to `data/out/<config_name>-rl1.flop` by default, or specify `--output <path>`.

## Expected Behavior

- Each episode takes minutes (varies by board complexity and hardware)
- Exploitability typically decreases monotonically across episodes
- Some boards converge faster than others
- The 0.5% target is achievable — exploitability keeps decreasing with more episodes (no fundamental limit for RL1)
- If a run stalls, try resuming with more episodes; the learning rate and architecture are tuned

## Monitoring

Check `log.csv` or `curves.png` in the model output directory to track progress. Key metric is `exploitability_pct` — training is done when it drops below 0.5.
