"""
Train one NN to predict flop strategy across multiple boards.

Usage: python demo_nn_flop_multi.py <board1.flop> <board2.flop> ...

Adds board encoding (3 cards one-hot = 156 dims) to input so the model
knows which board it's on.
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
    print("Usage: python demo_nn_flop_multi.py <board1.flop> <board2.flop> ...")
    sys.exit(1)

flop_paths = sys.argv[1:]

# ── 1. Load all boards and collect flop node data ──
MAX_HISTORY = 6

all_boards_meta = []  # per-board metadata for saving
X_list = []
Y_list = []
max_actions_global = 0

# First pass: find max_actions across all boards
print("Scanning boards...")
for flop_path in flop_paths:
    game = postflop_solver.GameWrapper.load_from_file(flop_path)

    def scan_max_actions():
        global max_actions_global
        if game.is_terminal() or game.is_chance():
            return
        n = game.current_num_actions()
        if n > max_actions_global:
            max_actions_global = n
        hist = list(game.history())
        for i in range(n):
            game.play(i)
            scan_max_actions()
            game.back_to_root()
            for a in hist:
                game.play(a)

    game.back_to_root()
    scan_max_actions()
    print(f"  {flop_path}: OOP={game.num_private_hands(0)}, IP={game.num_private_hands(1)}")
    del game

print(f"Max actions across all boards: {max_actions_global}")

# Input dims: hand(104) + board(156) + player(1) + history(MAX_HISTORY * max_actions) + pot(1)
history_dim = MAX_HISTORY * max_actions_global
input_dim = 104 + 156 + 1 + history_dim + 1
starting_pot = 55  # from config (same for all)

print(f"Input dim: {input_dim}, Output dim: {max_actions_global}")

# Second pass: collect training data
print("\nCollecting training data...")
total_nodes = 0

for flop_path in flop_paths:
    game = postflop_solver.GameWrapper.load_from_file(flop_path)
    oop_cards = game.private_cards(0)
    ip_cards = game.private_cards(1)

    # Get board cards from the first OOP hand's excluded cards
    # Actually, we need to parse board from filename
    board_name = os.path.basename(flop_path).split("_")[0].split("-")[0]
    # Parse 3 cards from board name (e.g. "2c3c4h")
    board_cards = []
    for i in range(0, 6, 2):
        r = RANKS.index(board_name[i].upper())
        s = SUITS.index(board_name[i + 1].lower())
        board_cards.append(r * 4 + s)

    # Board one-hot encoding (156 = 3 * 52)
    board_enc = np.zeros(156, dtype=np.float32)
    for slot, cid in enumerate(board_cards):
        board_enc[slot * 52 + cid] = 1.0

    board_nodes = []

    def collect_nodes():
        global total_nodes
        if game.is_terminal() or game.is_chance():
            return
        player = game.current_player()
        actions = game.current_actions()
        n_act, n_h, strat_flat = game.get_current_strategy()
        hist = list(game.history())
        bets = game.total_bet_amount()
        strategy = np.array(strat_flat).reshape(n_act, n_h)
        cards = oop_cards if player == 0 else ip_cards
        pot = starting_pot + bets[0] + bets[1]

        board_nodes.append({
            "history": hist,
            "player": player,
            "actions": actions,
            "n_act": n_act,
            "bets": bets,
        })

        # Encode history
        hist_enc = np.zeros(history_dim, dtype=np.float32)
        for step, action_idx in enumerate(hist):
            if step < MAX_HISTORY and action_idx < max_actions_global:
                hist_enc[step * max_actions_global + action_idx] = 1.0

        pot_frac = pot / (starting_pot + 360)

        for h in range(n_h):
            c1, c2 = cards[h]
            x = np.zeros(input_dim, dtype=np.float32)
            # Hand one-hot
            x[c1] = 1.0
            x[52 + c2] = 1.0
            # Board one-hot
            x[104:104 + 156] = board_enc
            # Player
            x[260] = float(player)
            # History
            x[261:261 + history_dim] = hist_enc
            # Pot fraction
            x[261 + history_dim] = pot_frac

            # Target
            y = np.zeros(max_actions_global, dtype=np.float32)
            y[:n_act] = strategy[:, h]

            X_list.append(x)
            Y_list.append(y)

        total_nodes += 1

        for i in range(len(actions)):
            game.play(i)
            collect_nodes()
            game.back_to_root()
            for a in hist:
                game.play(a)

    game.back_to_root()
    collect_nodes()

    all_boards_meta.append({
        "flop_path": flop_path,
        "board_name": board_name,
        "board_cards": board_cards,
        "n_oop": len(oop_cards),
        "n_ip": len(ip_cards),
        "oop_cards": oop_cards,
        "ip_cards": ip_cards,
        "nodes": board_nodes,
    })

    print(f"  {board_name}: {len(board_nodes)} nodes, {sum(n['n_act'] * (len(oop_cards) if n['player']==0 else len(ip_cards)) for n in board_nodes)} samples")
    del game

X = np.array(X_list)
Y = np.array(Y_list)
print(f"\nTotal: {total_nodes} nodes, {len(X)} training samples")

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
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

model = FlopStrategyNet(input_dim, 512, max_actions_global)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())
print(f"Training NN ({n_params} params, hidden=512)...")

losses = []
best_loss = float("inf")
best_state = None
n_epochs = 5000

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
board_names = [m["board_name"] for m in all_boards_meta]
model_name = "_".join(board_names) + "_flop.pt"
model_path = os.path.join(model_dir, model_name)
torch.save({
    "model": best_state,
    "input_dim": input_dim,
    "max_actions": max_actions_global,
    "max_history": MAX_HISTORY,
    "starting_pot": starting_pot,
    "boards": all_boards_meta,
}, model_path)
print(f"Model saved: {model_path}")

# ── Save loss plot ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (cross-entropy)")
ax.set_title(f"Multi-board Flop Training — {', '.join(board_names)} ({total_nodes} nodes, {len(X)} samples)")
ax.grid(True, alpha=0.3)
plot_path = os.path.join(model_dir, "_".join(board_names) + "_flop_loss.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Loss plot saved: {plot_path}")

# ── 4. Compare per board ──
with torch.no_grad():
    pred_all = model(X_t).numpy()

print(f"\n{'='*70}")
print(f"RESULTS: NN vs CFR per board")
print(f"{'='*70}")

idx = 0
for meta in all_boards_meta:
    board_name = meta["board_name"]
    oop_cards = meta["oop_cards"]
    ip_cards = meta["ip_cards"]
    board_max = 0.0
    board_samples = 0

    print(f"\n  Board: {board_name}")
    for node in meta["nodes"]:
        player = node["player"]
        n_act = node["n_act"]
        cards = oop_cards if player == 0 else ip_cards
        n_h = len(cards)

        cfr = Y[idx:idx + n_h, :n_act]
        nn_pred = pred_all[idx:idx + n_h, :n_act]
        diff = np.abs(nn_pred - cfr)
        node_max = diff.max()
        board_max = max(board_max, node_max)
        board_samples += n_h
        idx += n_h

    print(f"    nodes={len(meta['nodes'])}, samples={board_samples}, max_diff={board_max:.6f}")
