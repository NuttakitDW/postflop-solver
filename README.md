# DeepRun NN

Neural network approach that learns to predict poker flop strategy from solved games.
Given a hand and game context, the NN outputs action probabilities — replacing the need to store or re-solve the game tree at the flop.

Full documentation: [docs/deeprun_nn.md](docs/deeprun_nn.md)

## Pipeline

### Step 1: Solve with CFR
```bash
make start CONFIG=config/2c3c4h_p2.json
```

### Step 2: Train NN

**Single board:**
```bash
/opt/anaconda3/bin/python trainings/demo_nn_flop.py data/out/2c3c4h_p2.flop
```

**Multi board (one model for multiple spots):**
```bash
/opt/anaconda3/bin/python trainings/demo_nn_flop_multi.py \
  data/out/2c3c4h_p2.flop \
  data/out/7s6s4c_p2.flop \
  data/out/Ad8s2c_p2.flop
```

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
