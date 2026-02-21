#!/usr/bin/env python3
"""
print_full_feature_stats.py
----------------------------------------------------
Shows post-scaling statistics.

Usage
-----
$ python print_full_feature_stats.py          # only board features + range min/max
$ python print_full_feature_stats.py --range  # prints *all* 2 015 columns

Flags
-----
--range     also print stats for every probability column (15-2014).
"""

import os, argparse, numpy as np

# ------------------- CLI -------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--range", action="store_true",
    help="print stats for the 2,000 probability columns as well"
)
args = parser.parse_args()

# ------------------- paths -----------------
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "training_data_100k"
)
x_raw = np.load(os.path.join(DATA_DIR, "inputs.npy")).astype(np.float32)

# ------------------- constants -------------
BOARD_FEATURES = 15
K              = 1000
RNG_SEED       = 42
np.random.seed(RNG_SEED)

# ------------------- board-group split -----
def board_split(arr, frac=0.2, seed=42):
    keys = np.round(arr[:, :12], 4)
    ids  = {tuple(k): i for i, k in enumerate({tuple(r) for r in keys})}
    idx  = np.array([ids[tuple(k)] for k in keys])
    rng  = np.random.RandomState(seed)
    test_boards = set(rng.choice(len(ids), int(len(ids)*frac), replace=False))
    mask = np.isin(idx, list(test_boards))
    return np.where(~mask)[0], np.where(mask)[0]

tr_idx, _ = board_split(x_raw, 0.2, RNG_SEED)

# ------------------- range-safe scaling ----
mu  = x_raw[tr_idx, :BOARD_FEATURES].mean(0)
std = x_raw[tr_idx, :BOARD_FEATURES].std (0) + 1e-8
x   = x_raw.copy()
x[:, :BOARD_FEATURES] = (x[:, :BOARD_FEATURES] - mu) / std   # z-score only first 15

# ------------------- prints -----------------
print("\n=== Board-geometry columns (after scaling) ===")
print("col   mean        std")
print("---   ----------  ----------")
for col in range(BOARD_FEATURES):
    m, s = x[:, col].mean(), x[:, col].std()
    print(f"{col:3d}  {m:+.6f}  {s:.6f}")

rng_min, rng_max = x[:, BOARD_FEATURES:].min(), x[:, BOARD_FEATURES:].max()
print(f"\nRange-probability columns  min={rng_min:.4f}  max={rng_max:.4f}")

if args.range:
    print("\n=== FULL list of range columns ===")
    print("col   mean        std")
    print("---   ----------  ----------")
    for col in range(BOARD_FEATURES, x.shape[1]):
        m, s = x[:, col].mean(), x[:, col].std()
        print(f"{col:3d}  {m:+.6f}  {s:.6f}")