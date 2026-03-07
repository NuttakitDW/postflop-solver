# .bt1 Format — Boundary Training Data Standard

## Overview

A `.bt1` file stores per-iteration boundary data from a DCFR poker solver. For each iteration, at each turn boundary node, it records:

- **CFV** (counterfactual value) — the value returned from the turn/river subtree
- **cfreach** (counterfactual reach) — the opponent's reach probability at that boundary

This is the training data for a neural network that learns to predict boundary CFVs from cfreach, replacing the turn/river subtree computation.

## Why .bt1 Exists

Phase 1 proved that boundary CFVs are sufficient to solve the flop correctly (via `.dpairs2` files that store only CFVs). The key insight for Phase 2 is that **cfreach changes every iteration** alongside CFVs — making it a valid NN input. The `.bt1` format extends `.dpairs2` by also recording cfreach, giving us (input, output) training pairs.

## Relationship to .dpairs2

| | .dpairs2 | .bt1 |
|---|---|---|
| **Purpose** | Perfect replay (lookup table) | NN training data |
| **Stores** | CFV only | CFV + cfreach |
| **cfreach layout** | — | opponent's reach per hand |
| **convergence_mode** | yes (per-iteration flag) | no (not needed for NN) |
| **Use case** | Verify boundary approach works | Train NN to predict CFV from cfreach |

## Binary Format

### Header (32 bytes)

```
Offset  Size  Type      Field
0       8     bytes     Magic: b"BT1\0\0\0\0\0"
8       4     u32-LE    version (always 1)
12      4     u32-LE    num_oop — OOP hand count (e.g. 860)
16      4     u32-LE    num_ip — IP hand count (e.g. 502)
20      4     u32-LE    num_boundaries — turn boundary nodes (e.g. 25)
24      4     u32-LE    num_iterations — total iterations recorded
28      4     f32-LE    starting_pot (e.g. 55.0 chips)
```

### Per-Iteration Record (repeated `num_iterations` times)

```
Offset  Size  Type      Field
0       4     u32-LE    iteration — 0-based iteration index
4       4     f32-LE    exploitability — current exploitability in chips
8       4     u32-LE    reserved — always 0
```

Then for each of the `num_boundaries` boundaries (in DFS traversal order):

```
For each boundary b:
  Player 0 (OOP):
    cfv[num_oop]     — f32-LE × num_oop    (OOP's counterfactual values)
    cfreach[num_ip]  — f32-LE × num_ip     (IP's reach = opponent of OOP)

  Player 1 (IP):
    cfv[num_ip]      — f32-LE × num_ip     (IP's counterfactual values)
    cfreach[num_oop] — f32-LE × num_oop    (OOP's reach = opponent of IP)
```

### Size Calculation

Per boundary: `(num_oop + num_ip) × 2 × 4` bytes (cfv + cfreach for both players)

Example for KcQh7s (num_oop=860, num_ip=502):
- Per boundary: `(860 + 502) × 2 × 4 = 10,896` bytes
- Per iteration: `12 + 25 × 10,896 = 272,412` bytes
- Total (180 iterations): `32 + 180 × 272,412 ≈ 47 MB`

## Field Definitions

### CFV (Counterfactual Value)

A vector indexed by the **acting player's** hand index (solver-compressed, not canonical 1326).

- For player 0 (OOP): length = `num_oop`
- For player 1 (IP): length = `num_ip`
- Units: chips
- Meaning: expected chip gain/loss at this boundary, given current strategies and opponent reach
- Source: computed by the DCFR traversal through the full turn/river subtree

### cfreach (Counterfactual Reach)

A vector indexed by the **opponent's** hand index.

- For player 0 (OOP): cfreach has length = `num_ip` (IP is the opponent)
- For player 1 (IP): cfreach has length = `num_oop` (OOP is the opponent)
- Values: un-normalized reach probabilities (product of action probabilities from root to boundary)
- Meaning: how much weight the opponent places on each hand at this boundary
- Key property: **changes every iteration** as strategies evolve, making it a valid NN input

### Exploitability

Total exploitability of current strategies, in chips. Updated every 10 iterations; between updates it carries the previous value.

## Boundary Ordering

Boundaries are in **DFS traversal order** of the flop action tree. This matches the order used by `collect_boundary_cfreaches` during inference, ensuring correct alignment.

Example for KcQh7s (25 boundaries):

```
b=0:  X-X                pot=55    (check-check → turn)
b=1:  X-B18-C            pot=91    (check, bet 18, call)
b=2:  X-B18-R64-C        pot=183   (check, bet, raise, call)
...
b=24: A180-C             pot=415   (all-in, call)
```

Full boundary list with pot sizes is stored in `{board}_boundaries.json`.

## Sidecar Files

When building a `.bt1` file, the builder also writes:

- **`{board}_boundaries.json`** — boundary index → action path + pot size
- **`{board}_hands.json`** — solver hand index → card string (e.g. index 0 → "2c3c")

These are not needed by the training pipeline but are useful for debugging and visualization.

## How Data is Generated

```
examples/build_bt1/main.rs
```

1. Build the full game tree from config
2. For each DCFR iteration:
   - Call `solve_step_for_player_recording_with_cfreach` for player 0 and player 1
   - This runs normal DCFR but records (cfv, cfreach) at each turn boundary
   - Also computes exploitability every 10 iterations
3. Write binary `.bt1` file + sidecar JSONs

## How Data is Used for Training

```
trainings/train_bt1.py
```

Each (iteration, boundary, player) tuple becomes one training sample:

```
Input:  [boundary_onehot(25), player_flag(1), cfreach_padded(max_hands)]  = 886 dims
Output: [cfv_padded(max_hands)]                                           = 860 dims
Mask:   [1 for valid positions, 0 for padding]                            = 860 dims
```

- cfreach is the **opponent's** reach (the NN input)
- cfv is the **player's** value (the NN target)
- Masked HuberLoss ensures padding doesn't affect training
- Total samples for KcQh7s: `180 × 25 × 2 = 9,000`
