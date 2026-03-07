# Postflop Solver — NN-Accelerated

Neural network approach that learns to predict poker flop strategy from solved games.
Given a hand and game context, the NN outputs action probabilities — replacing the need to store or re-solve the game tree at the flop.

Full documentation: [docs/deeprun_nn.md](docs/deeprun_nn.md)

## NN1 Pipeline

### Step 1: Solve with CFR
```bash
make start CONFIG=config/2c3c4h_p2.json
```

### Step 2: Train NN
```bash
/opt/anaconda3/bin/python trainings/nn1_train.py data/out/2c3c4h_p2.flop
```
Saves model to `models/nn1/<board>_flop.pt`

### Step 3: Solve with NN model
```bash
/opt/anaconda3/bin/python trainings/nn1_solve.py \
  config/2c3c4h_p2.json \
  models/nn1/2c3c4h_p2_flop.pt
```
Outputs: `data/out/2c3c4h_p2-nn.flop` (flop from NN, turn/river uniform)

### Optional: Interactive inference
```bash
/opt/anaconda3/bin/python trainings/nn1_infer.py models/nn1/2c3c4h_p2_flop.pt KcKd
```

## Files

| File | Purpose |
|------|---------|
| `trainings/nn1_train.py` | Train NN on all flop nodes (single board) |
| `trainings/nn1_solve.py` | Build .flop file from trained NN model |
| `trainings/nn1_infer.py` | Interactive inference — query hand strategies |
