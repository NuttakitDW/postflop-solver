# RL1: Online RL Training with Exploitability Objective

## Problem

All Phase 2 experiments fail from **distribution shift**: model trains on one cfreach trajectory, drifts OOD at inference, errors compound over 180 iterations. We also never measured exploitability with an NN because the model-driven solve skips turn/river subtrees, leaving no strategies to evaluate.

## Objective

Train a model that produces boundary CFVs achieving **< 0.3% exploitability** (relative to pot) on a single board/config.

**MVP**: One board (KcQh7s), exploitability < 0.3%. That proves Phase 2 works.

## Architecture: Single Full Tree

One full game tree. The model drives flop regret updates, while turn/river subtrees are traversed normally. This gives us everything from a single pass:

- **Model's cfreaches** at boundaries (model input)
- **True CFVs** from turn/river traversal (dense training label)
- **Turn/river strategies** maintained normally (needed for exploitability)
- **Flop strategies** driven by model's predictions (what we're evaluating)
- **Exploitability** computable because all nodes have strategies

## The Key Solver Function

```
solve_step_with_model(game, iteration, player, model_cfvs):
    Traverse full tree (flop + turn + river):
    - At boundary nodes:
        * Record cfreaches → model input
        * Traverse turn/river subtree normally → get true CFVs (training label)
        * Update turn/river regrets normally
        * BUT use model_cfvs (not true CFVs) for flop parent regret update
    - Returns: (cfreaches, true_cfvs) for training
```

## Training Loop

```python
game = GameWrapper("config/KcQh7s.json")

for episode in range(N):
    game.reset()

    for t in range(180):
        for player in [0, 1]:
            cfreaches = game.collect_boundary_cfreaches(player)
            predicted = model.predict(cfreaches, player)

            # Full tree traversal: model CFVs for flop, normal for turn/river
            true_cfvs = game.solve_step_with_model(t, player, predicted)

            # Train on dense CFV loss per step
            loss = mse(predicted, true_cfvs)
            loss.backward()
            optimizer.step()

    # Evaluate episode
    exploit = game.compute_exploitability()
    print(f"Episode {episode}: exploit={exploit:.4f}%")
```

Exploitability is the **evaluation metric**. Dense CFV loss is the **training signal**. If CFV loss alone doesn't reach < 0.3% exploitability, we add exploitability as an auxiliary RL reward later.

## Inference

```python
eval_game = GameWrapper("config/KcQh7s.json", flop_only=True)

for t in range(180):
    for player in [0, 1]:
        cfreaches = eval_game.collect_boundary_cfreaches(player)
        predicted = model.predict(cfreaches, player)
        eval_game.solve_step_replay(t, player, predicted)
```

---

## Implementation Steps

### Step 1: `solve_step_with_model` (Rust)

Add a new function to `src/solver.rs`. It's a modified `solve_step_for_player_recording` where at boundary nodes the true CFVs are recorded but the model's CFVs are passed up to the flop parent instead.

**Inputs**: game, iteration, player, model_cfvs (Vec of CFV vectors, one per boundary)
**Outputs**: true_cfvs (Vec of CFV vectors from subtree traversal)

**Verify before moving on**:
- Build a Rust example that runs `solve_step_with_model` passing true CFVs (from `solve_step_for_player_recording`) as model_cfvs
- Compare final strategy against normal `solve` → must be identical
- This confirms the function is a no-op when model_cfvs = true_cfvs

### Step 2: PyO3 Bridge (Rust → Python)

Create a Python module (via PyO3 or subprocess/stdin pipe) exposing:

```python
class GameWrapper:
    def __init__(self, config_path: str)
    def reset(self)
    def num_private_hands(self, player: int) -> int
    def num_boundaries(self) -> int
    def collect_boundary_cfreaches(self, player: int) -> list[list[float]]
    def solve_step_with_model(self, iter: int, player: int,
                              model_cfvs: list[list[float]]) -> list[list[float]]
    def solve_step_replay(self, iter: int, player: int, cfvs: list[list[float]])
    def compute_exploitability(self) -> float
    def save(self, path: str)
```

**Verify before moving on**:
- From Python, create a game, run 180 iterations of `solve_step_with_model` with model_cfvs = true_cfvs (collect them from the same function)
- Compute exploitability → should match normal solver's exploitability
- This confirms the bridge passes data correctly

### Step 3: Training Loop with Dense CFV Loss

Write `trainings/train_rl1.py`. Model architecture same as bt1 (MLP, cfreach → CFV).

Run training on KcQh7s for a few episodes. After each episode, compute and print exploitability.

**Verify before moving on**:
- CFV loss decreases over episodes
- Exploitability is computable and printed per episode
- No crashes, memory stable across episodes

### Step 4: Evaluate

After training, run inference-only (model predicts, no oracle). Export .flop file. Compare against baseline using `compare_flop_files`.

**Verify**:
- Exploitability < 0.3% → **Phase 2 complete**
- If not, we iterate: tune training (learning rate, episodes, model size), or add exploitability as RL reward

---

## What We Reuse

| Component | Purpose |
|-----------|---------|
| `solve_step_for_player_recording` | Reference implementation for Step 1 |
| `solve_step_for_player_replay` | Inference-time solve |
| `collect_boundary_cfreaches` | Get cfreaches at boundaries |
| `.dpairs2` / `compare_flop_files` | Baseline comparison |

## What We Build

| Component | Purpose |
|-----------|---------|
| `solve_step_with_model` in `src/solver.rs` | Core solver function |
| PyO3 bridge or Rust-Python pipe | Rust callable from Python |
| `trainings/train_rl1.py` | Training loop |
| `trainings/eval_rl1.py` | Inference + export |

## Cost

- ~20 min per episode (same as one full solve)
- Estimated 10-50 episodes to overfit one board (3-17 hours)
- Inference: seconds

## Success Criteria

| Metric | Target |
|--------|--------|
| Exploitability (KcQh7s) | **< 0.3% of pot** |
