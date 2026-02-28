# dpairs2 — Boundary CFV Oracle (Phase 1: Proof of Concept)

## What Problem Does This Solve?

A postflop poker solver uses DCFR to converge on a Nash equilibrium strategy.
The game tree has three streets: **flop → turn → river**.
The turn/river subtree is massive — it accounts for most of the computation.

The long-term goal: replace the turn/river subtree with a neural network during
DCFR solving. But first, we need to **prove the concept** — can we skip the
turn/river subtree entirely and still get a perfect flop strategy?

**Phase 1 (this implementation)** answers that question. We run the full solver
once, record the boundary CFVs into a `.dpairs2` lookup table, then replay the
flop-only solver using those recorded values. If the replay produces
bit-identical flop strategies, it proves that a perfect boundary predictor
(lookup table today, neural network in the future) is sufficient.

## What Is a "Boundary"?

When flop action ends (players check or finish betting), the game reaches a
**chance node** where the turn card is dealt. This is a **turn boundary**.

Different flop action sequences create different boundaries:

```
                   ROOT (pot=55, OOP to act)
                  /           |            \
             Check          Bet 33        AllIn
            /    \            |              |
       IP Check  IP Bet   IP to act      TERMINAL
          |       ...    /    |     \
       CHANCE          Fold  Call  Raise50  ...
    [Boundary 0]        |     |       \
     pot = 55        TERM  CHANCE    IP to act
                        [Boundary 1]   /    \
                         pot = 121   Call  Raise100
                                      |       \
                                   CHANCE    ...
                                [Boundary 2]
                                 pot = 188
```

Each boundary is a different pot size reached through a different betting line.
Boundaries are numbered in **DFS order** (depth-first traversal of the flop tree).

## What Is a CFV Vector?

At each boundary, for each player, the solver computes a **CFV vector** —
one float per possible hole-card combination for that player.

```
Boundary 3, OOP CFV vector (863 floats):
  hand[0]   = -0.0415   ← "holding hand #0, expected to lose 0.04 chips"
  hand[1]   =  0.0280   ← "holding hand #1, expected to gain 0.03 chips"
  ...
  hand[862] =  0.0450
```

OOP and IP have different range sizes (e.g., 863 vs 526 hands on board 9s6d6c).

## Why Store Per-Iteration?

DCFR is **stateful** — each iteration builds on cumulative regrets from all
previous iterations. The CFV at boundary `b` at iteration `t=50` differs from
`t=5` because the strategies have evolved. You can't just store the final CFV.
**Every iteration needs its own snapshot.**

## How It Works

### Step 1: Build — Record Boundary CFVs

`build_pairs_v2` runs the full DCFR solver (flop + turn + river). At each
turn boundary, it captures the CFV that the turn/river subtree returned.

```
for t in 0..max_iterations:
    p0_cfvs = solve_step_for_player_recording(game, t, player=0, conv_mode)
    p1_cfvs = solve_step_for_player_recording(game, t, player=1, conv_mode)
    write_to_file(t, exploitability, conv_mode, p0_cfvs, p1_cfvs)
```

Inside `solve_step_for_player_recording` (src/solver.rs), the recursion does a
full tree traversal. When it hits a turn boundary:

```rust
// At a chance node where turn card has not been dealt:
if node.is_chance() && node.turn() == NOT_DEALT {
    // Already recursed into all 45 turn cards + river subtrees
    // Already summed CFVs across all outcomes
    // Now capture the result:
    boundaries.push(cfv.clone());   // raw CFV, no discount applied
}
```

The `boundaries` vector fills in DFS order — boundary 0 is the first turn
chance node encountered, boundary 1 is the second, etc.

### Step 2: Replay — Inject Recorded CFVs

`solve_with_pairs_v2` runs only the flop. At each turn boundary, it injects
the previously recorded CFV and skips the turn/river subtree entirely.

```
for t in 0..max_iterations:
    (conv_mode, p0_cfvs, p1_cfvs) = read_from_file(t)
    solve_step_for_player_replay(game, t, player=0, conv_mode, p0_cfvs)
    solve_step_for_player_replay(game, t, player=1, conv_mode, p1_cfvs)
```

Inside `solve_step_for_player_replay` (src/solver.rs), when the recursion hits
a turn boundary:

```rust
if node.is_chance() && node.turn() == NOT_DEALT {
    result.write(boundary_cfvs[counter]);   // inject recorded CFV
    counter += 1;
    return;                                 // skip turn/river entirely
}
```

The result: replay produces **bit-identical** flop strategies and regrets
as the full solve, while skipping all turn/river computation.

### Why `convergence_mode` Is Stored Per-Iteration

DCFR has two regimes with different discount parameters:

| Parameter | Normal Mode | Convergence Mode (exploit < 1% pot) |
|-----------|-------------|--------------------------------------|
| alpha     | `t^1.5 / (t^1.5 + 1)` | `max(above, 0.9)` |
| beta      | `0.5`       | `0.9`                                |
| gamma     | `t / (t+1)` | `max(above, 0.9)`                   |

The switch from normal to convergence mode changes how regrets are discounted.
If replay recomputed the flag (its exploitability may differ slightly), it could
switch at a different iteration, causing cascading regret differences. Recording
the flag ensures identical discount parameters.

### Where Discounts Are Applied

Discounting happens in the **parent node** after receiving CFVs from children:

```rust
// After recursion returns child CFVs:
cumulative_regret[i] = cumulative_regret[i] * coef + child_cfv[i]
//                      ^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
//                      discount old regrets         raw CFV (not discounted)
```

Boundary CFVs are **raw** values — the parent applies its own discounting.
This is why recording raw CFVs is correct.

## .dpairs2 Binary Format

```
HEADER (32 bytes):
  [0..8]   "DPAIRS2\0"     magic
  [8..12]  u32 LE          version (= 2)
  [12..16] u32 LE          num_oop       (OOP hand count)
  [16..20] u32 LE          num_ip        (IP hand count)
  [20..24] u32 LE          num_boundaries
  [24..28] u32 LE          num_iterations
  [28..32] f32 LE          starting_pot

PER ITERATION (repeated num_iterations times):
  [+0]     u32 LE          iteration number
  [+4]     f32 LE          exploitability
  [+8]     u32 LE          convergence_mode (0 or 1)
  [+12]    boundary data:
           for b in 0..num_boundaries:
             f32[num_oop]    OOP CFV vector for boundary b
             f32[num_ip]     IP CFV vector for boundary b
```

Think of the data as two 3D tensors:

```
OOP: shape (num_iterations, num_boundaries, num_oop)
IP:  shape (num_iterations, num_boundaries, num_ip)
```

## Project Structure

### Source Code

| File | What It Does |
|------|-------------|
| `src/solver.rs` | Core DCFR solver. Contains `solve_step_for_player_recording` and `solve_step_for_player_replay` |

### Example Binaries (Phase 1 — dpairs2)

| Binary | What It Does |
|--------|-------------|
| `build_pairs_v2` | Full solve, records boundary CFVs, writes `.dpairs2` |
| `solve_with_pairs_v2` | Flop-only solve, reads `.dpairs2`, injects CFVs at boundaries |

### Other Useful Binaries

| Binary | What It Does |
|--------|-------------|
| `backend_solver` | Standard full solve, writes `.flop` file (used as baseline) |

### Config and Data

| Path | Contents |
|------|----------|
| `config/*.json` | Game configs (ranges, bet sizes, pot, stack). One per board. |
| `data/oracles/*.dpairs2` | Pre-computed oracle files. 10 boards, ~20-27 MB each. |
| `notebooks/inspect_dpairs2.ipynb` | Interactive guide to the `.dpairs2` data structure |

## How to Run

### Prerequisites

```bash
# Rust toolchain (stable)
rustup update stable
```

### 1. Build a .dpairs2 oracle from a board config

```bash
cargo run --example build_pairs_v2 --release --features "bincode rayon" -- config/9s6d6c.json
```

Output: `data/oracles/9s6d6c.dpairs2`

### 2. Replay: solve flop-only using the oracle

```bash
cargo run --example solve_with_pairs_v2 --release --features "bincode rayon" -- config/9s6d6c.json
```

Output: `data/out/9s6d6c-pairs2.flop`

### 3. Run baseline (standard full solve) for comparison

```bash
cargo run --example backend_solver --release --features "bincode zstd" -- config/9s6d6c.json
```

Output: `data/out/9s6d6c-standard.flop`


## Available Boards

| Board | Config | Oracle |
|-------|--------|--------|
| 2c3c4h | `config/2c3c4h.json` | `data/oracles/2c3c4h.dpairs2` |
| 7s6s4c | `config/7s6s4c.json` | `data/oracles/7s6s4c.dpairs2` |
| 9s6d6c | `config/9s6d6c.json` | `data/oracles/9s6d6c.dpairs2` |
| 9s8s2c | `config/9s8s2c.json` | `data/oracles/9s8s2c.dpairs2` |
| Ad8s2c | `config/Ad8s2c.json` | `data/oracles/Ad8s2c.dpairs2` |
| As3s4s | `config/As3s4s.json` | `data/oracles/As3s4s.dpairs2` |
| AsKcTh | `config/AsKcTh.json` | `data/oracles/AsKcTh.dpairs2` |
| Jc8c6s | `config/Jc8c6s.json` | `data/oracles/Jc8c6s.dpairs2` |
| Kc8h7h | `config/Kc8h7h.json` | `data/oracles/Kc8h7h.dpairs2` |
| KcQh7s | `config/KcQh7s.json` | `data/oracles/KcQh7s.dpairs2` |

## Key Lessons from v1 to v2

Three bugs were found in v1 and fixed in v2:

1. **Custom DCFR missed convergence_mode** — v1 reimplemented DCFR outside the library and
   used `beta=0.5` always. The library switches to `beta=0.9` near convergence. Result: 70% diff.
   Fix: use the library's own `solve_step_for_player_recording`.

2. **Duplicate boundary keys** — v1 keyed boundaries by `(iter, pot_amount, player)`. Different
   action sequences can reach the same pot amount, causing key collisions. Result: 42% diff.
   Fix: use sequential DFS index instead of pot amount.

3. **Convergence mode timing mismatch** — v1 recomputed the convergence flag during replay,
   which could trigger at a different iteration. Result: 1.15% diff.
   Fix: record the flag per-iteration in the `.dpairs2` file.
