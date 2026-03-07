"""
Demo: Train a NN to predict OOP flop root strategy from a solved .flop file.

Usage: python demo_nn_strategy.py <path_to.flop>
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import postflop_solver

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_str(cid):
    return f"{RANKS[cid // 4]}{SUITS[cid % 4]}"

def hand_str(c1, c2):
    return f"{card_str(c1)}{card_str(c2)}"

if len(sys.argv) < 2:
    print("Usage: python demo_nn_strategy.py <path_to.flop>")
    sys.exit(1)

flop_path = sys.argv[1]
basename = os.path.basename(flop_path).replace(".flop", "")

# ── 1. Load solved game ──
game = postflop_solver.GameWrapper.load_from_file(flop_path)
n_act, n_h, strat_flat = game.get_root_strategy()
strategy = np.array(strat_flat).reshape(n_act, n_h)
cards = game.private_cards(0)
actions = game.root_actions()
print(f"Loaded: {flop_path}")
print(f"{n_h} hands, {n_act} actions: {actions}")

# ── 2. Encode hands (one-hot card1 + card2 = 104 dims) ──
X = np.zeros((n_h, 104), dtype=np.float32)
for i, (c1, c2) in enumerate(cards):
    X[i, c1] = 1.0
    X[i, 52 + c2] = 1.0
Y = strategy.T.astype(np.float32)  # [hands, actions]
X_t, Y_t = torch.from_numpy(X), torch.from_numpy(Y)

# ── 3. Train NN ──
class StrategyNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

model = StrategyNet(104, 256, n_act)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nTraining NN ({n_params} params)...")

losses = []
best_loss = float("inf")
best_state = None
n_epochs = 2000
for epoch in range(n_epochs):
    pred = model(X_t)
    loss = -(Y_t * torch.log(pred.clamp(min=1e-8))).sum(-1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    l = loss.item()
    losses.append(l)
    if l < best_loss:
        best_loss = l
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if (epoch + 1) % 500 == 0:
        print(f"  epoch {epoch+1}: loss={l:.6f}")

model.load_state_dict(best_state)
print(f"Best loss: {best_loss:.6f}")

# ── Save model ──
model_dir = "models/demo"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"{basename}.pt")
torch.save({
    "model": best_state,
    "actions": actions,
    "cards": cards,
    "n_act": n_act,
    "n_hands": n_h,
    "flop_path": flop_path,
}, model_path)
print(f"Model saved: {model_path}")

# ── Save loss plot ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (cross-entropy)")
ax.set_title(f"Training Loss — {basename} ({n_h} hands, {n_act} actions)")
ax.grid(True, alpha=0.3)
plot_path = os.path.join(model_dir, f"{basename}_loss.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Loss plot saved: {plot_path}")

# ── 4. Compare ──
with torch.no_grad():
    pred = model(X_t).numpy()

diff = np.abs(pred - Y)
max_diff = diff.max(axis=1)
print(f"\nMax diff: {max_diff.max():.6f}, Mean diff: {diff.mean():.6f}")

worst_idx = np.argsort(max_diff)[-10:][::-1]
print(f"\nWorst 10 hands:")
for i in worst_idx:
    c1, c2 = cards[i]
    cfr = " ".join(f"{Y[i,j]:.3f}" for j in range(n_act))
    nn_ = " ".join(f"{pred[i,j]:.3f}" for j in range(n_act))
    print(f"  {hand_str(c1,c2):5s}  CFR=[{cfr}]  NN=[{nn_}]  diff={max_diff[i]:.6f}")

print(f"\nAction frequency:")
for j in range(n_act):
    print(f"  {actions[j]:15s}: CFR={Y[:,j].mean():.4f}  NN={pred[:,j].mean():.4f}")
