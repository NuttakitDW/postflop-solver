"""
Train a NN to predict strategy at ALL flop decision nodes.

Usage: python nn1_train.py <path_to.flop>

The NN learns: (hand, node_context) -> action_probabilities
for every player node on the flop (both OOP and IP).
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
    print("Usage: python nn1_train.py <path_to.flop>")
    sys.exit(1)

flop_path = sys.argv[1]
basename = os.path.basename(flop_path).replace(".flop", "")

# ── 1. Load solved game and collect all flop nodes ──
game = postflop_solver.GameWrapper.load_from_file(flop_path)
oop_cards = game.private_cards(0)
ip_cards = game.private_cards(1)
n_oop = len(oop_cards)
n_ip = len(ip_cards)

# DFS to collect all flop player nodes
MAX_HISTORY = 6  # max action history depth on flop

nodes = []  # list of (history, player, actions, num_hands, strategy)

def collect_nodes():
    if game.is_terminal() or game.is_chance():
        return
    player = game.current_player()
    actions = game.current_actions()
    n_act, n_h, strat_flat = game.get_current_strategy()
    hist = game.history()
    bets = game.total_bet_amount()
    strategy = np.array(strat_flat).reshape(n_act, n_h)
    nodes.append({
        "history": list(hist),
        "player": player,
        "actions": actions,
        "n_act": n_act,
        "n_hands": n_h,
        "strategy": strategy,
        "bets": bets,
    })
    for i in range(len(actions)):
        game.play(i)
        collect_nodes()
        # navigate back to this node
        game.back_to_root()
        for a in hist:
            game.play(a)

game.back_to_root()
collect_nodes()

print(f"Loaded: {flop_path}")
print(f"OOP hands: {n_oop}, IP hands: {n_ip}")
print(f"Flop player nodes: {len(nodes)}")

# Find max actions across all nodes
max_actions = max(n["n_act"] for n in nodes)
print(f"Max actions at any node: {max_actions}")

# ── 2. Build training data ──
# For each node, for each hand, we have one training sample:
#   Input:  hand_onehot(104) + player(1) + history_onehot(MAX_HISTORY * max_actions) + pot_frac(1)
#   Output: action_probs (padded to max_actions)
#
# History encoding: at each step, one-hot over max_actions (which action was taken)
# Pad shorter histories with zeros.

history_dim = MAX_HISTORY * max_actions
input_dim = 104 + 1 + history_dim + 1  # cards + player + history + pot_fraction
starting_pot = 55  # from config

X_list = []
Y_list = []

for node in nodes:
    player = node["player"]
    cards = oop_cards if player == 0 else ip_cards
    n_h = node["n_hands"]
    n_act = node["n_act"]
    strategy = node["strategy"]  # [n_act, n_hands]
    hist = node["history"]
    bets = node["bets"]
    pot = starting_pot + bets[0] + bets[1]

    # Encode history
    hist_enc = np.zeros(history_dim, dtype=np.float32)
    for step, action_idx in enumerate(hist):
        if step < MAX_HISTORY and action_idx < max_actions:
            hist_enc[step * max_actions + action_idx] = 1.0

    pot_frac = pot / (starting_pot + 360)  # normalize by max pot

    for h in range(n_h):
        c1, c2 = cards[h]
        x = np.zeros(input_dim, dtype=np.float32)
        # Hand one-hot
        x[c1] = 1.0
        x[52 + c2] = 1.0
        # Player
        x[104] = float(player)
        # History
        x[105:105 + history_dim] = hist_enc
        # Pot fraction
        x[105 + history_dim] = pot_frac

        # Target: pad to max_actions
        y = np.zeros(max_actions, dtype=np.float32)
        y[:n_act] = strategy[:, h]

        X_list.append(x)
        Y_list.append(y)

X = np.array(X_list)
Y = np.array(Y_list)
print(f"\nTraining samples: {len(X)} (across all nodes)")
print(f"Input dim: {input_dim}, Output dim: {max_actions}")

X_t = torch.from_numpy(X)
Y_t = torch.from_numpy(Y)

# ── 3. Train NN ──
class FlopStrategyNet(nn.Module):
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

model = FlopStrategyNet(input_dim, 256, max_actions)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
print(f"Training NN ({n_params} params)...")

losses = []
best_loss = float("inf")
best_state = None
n_epochs = 3000

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
model_dir = "models/nn1"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"{basename}_flop.pt")
torch.save({
    "model": best_state,
    "input_dim": input_dim,
    "max_actions": max_actions,
    "max_history": MAX_HISTORY,
    "starting_pot": starting_pot,
    "n_oop": n_oop,
    "n_ip": n_ip,
    "oop_cards": oop_cards,
    "ip_cards": ip_cards,
    "nodes": [{
        "history": n["history"],
        "player": n["player"],
        "actions": n["actions"],
        "n_act": n["n_act"],
        "bets": n["bets"],
    } for n in nodes],
    "flop_path": flop_path,
}, model_path)
print(f"Model saved: {model_path}")

# ── Save loss plot ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (cross-entropy)")
ax.set_title(f"Full Flop Training — {basename} ({len(nodes)} nodes, {len(X)} samples)")
ax.grid(True, alpha=0.3)
plot_path = os.path.join(model_dir, f"{basename}_flop_loss.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Loss plot saved: {plot_path}")

# ── 4. Compare per node ──
with torch.no_grad():
    pred_all = model(X_t).numpy()

print(f"\n{'='*70}")
print(f"RESULTS: NN vs CFR at each flop node")
print(f"{'='*70}")

idx = 0
for node in nodes:
    n_h = node["n_hands"]
    n_act = node["n_act"]
    player = node["player"]
    pname = "OOP" if player == 0 else "IP"
    actions = node["actions"]
    hist = node["history"]

    cfr = node["strategy"].T  # [hands, n_act]
    nn_pred = pred_all[idx:idx + n_h, :n_act]
    idx += n_h

    diff = np.abs(nn_pred - cfr)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"\n  [{pname}] history={hist} actions={actions}")
    print(f"    hands={n_h}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    # Worst hand at this node
    worst = np.argmax(diff.max(axis=1))
    cards = oop_cards if player == 0 else ip_cards
    c1, c2 = cards[worst]
    cfr_s = " ".join(f"{cfr[worst, j]:.3f}" for j in range(n_act))
    nn_s = " ".join(f"{nn_pred[worst, j]:.3f}" for j in range(n_act))
    print(f"    worst: {hand_str(c1, c2)} CFR=[{cfr_s}] NN=[{nn_s}]")
