"""
Build a .flop file using NN for flop strategies, uniform for turn/river.

Usage: python build_flop_from_nn.py <config.json> <model_flop.pt> [output.flop]

1. Creates a fresh game from config (same tree structure)
2. Navigates all flop nodes → locks strategy from NN predictions
3. Turn/river nodes keep default uniform (zero regrets = uniform)
4. Finalizes and saves as .flop
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import postflop_solver

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_str(cid):
    return f"{RANKS[cid // 4]}{SUITS[cid % 4]}"

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

# Also support 4-layer models from multi-board training
class FlopStrategyNet4(nn.Module):
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

if len(sys.argv) < 3:
    print("Usage: python build_flop_from_nn.py <config.json> <model_flop.pt> [output.flop]")
    sys.exit(1)

config_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3] if len(sys.argv) > 3 else None

# ── Load model ──
ckpt = torch.load(model_path, weights_only=False)
input_dim = ckpt["input_dim"]
max_actions = ckpt["max_actions"]
max_history = ckpt["max_history"]
starting_pot = ckpt["starting_pot"]
has_board = input_dim > 200  # multi-board model has board encoding (286 vs 130)

# Detect hidden size and depth from state dict
first_weight = ckpt["model"]["net.0.weight"]
hidden_size = first_weight.shape[0]
n_layers = max(int(k.split(".")[1]) for k in ckpt["model"] if k.startswith("net.")) // 2 + 1

if n_layers == 5:
    model = FlopStrategyNet4(input_dim, hidden_size, max_actions)
else:
    model = FlopStrategyNet(input_dim, hidden_size, max_actions)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"Model: {model_path} (input={input_dim}, hidden={hidden_size}, layers={n_layers}, board={'yes' if has_board else 'no'})")

# ── Parse board from config ──
with open(config_path) as f:
    cfg = json.load(f)
board_str = cfg["board"]["flop"]

board_cards = []
for i in range(0, len(board_str), 2):
    r = RANKS.index(board_str[i].upper())
    s = SUITS.index(board_str[i + 1].lower())
    board_cards.append(r * 4 + s)

board_enc = np.zeros(156, dtype=np.float32)
for slot, cid in enumerate(board_cards):
    board_enc[slot * 52 + cid] = 1.0

# ── Create fresh game ──
game = postflop_solver.GameWrapper(config_path)
oop_cards = game.private_cards(0)
ip_cards = game.private_cards(1)
print(f"Config: {config_path} (board={board_str})")
print(f"OOP hands: {len(oop_cards)}, IP hands: {len(ip_cards)}")

def predict_strategy(cards, player, history, bets, n_act, n_hands):
    """Predict strategy for all hands at a node. Returns flat [n_act * n_hands]."""
    history_dim = max_history * max_actions
    pot = starting_pot + bets[0] + bets[1]
    pot_frac = pot / (starting_pot + 360)

    hist_enc = np.zeros(history_dim, dtype=np.float32)
    for step, action_idx in enumerate(history):
        if step < max_history and action_idx < max_actions:
            hist_enc[step * max_actions + action_idx] = 1.0

    X = np.zeros((n_hands, input_dim), dtype=np.float32)
    for h in range(n_hands):
        c1, c2 = cards[h]
        X[h, c1] = 1.0
        X[h, 52 + c2] = 1.0

        if has_board:
            # Multi-board: hand(104) + board(156) + player(1) + history + pot
            X[h, 104:104 + 156] = board_enc
            X[h, 260] = float(player)
            X[h, 261:261 + history_dim] = hist_enc
            X[h, 261 + history_dim] = pot_frac
        else:
            # Single-board: hand(104) + player(1) + history + pot
            X[h, 104] = float(player)
            X[h, 105:105 + history_dim] = hist_enc
            X[h, 105 + history_dim] = pot_frac

    with torch.no_grad():
        probs = model(torch.from_numpy(X)).numpy()

    strategy_flat = np.zeros(n_act * n_hands, dtype=np.float32)
    for a in range(n_act):
        for h in range(n_hands):
            strategy_flat[a * n_hands + h] = probs[h, a]

    return strategy_flat.tolist()

# ── DFS: lock all flop nodes with NN strategy ──
locked_count = 0

def lock_flop_nodes():
    global locked_count
    if game.is_terminal() or game.is_chance():
        return
    player = game.current_player()
    actions = game.current_actions()
    n_act = game.current_num_actions()
    n_hands = game.num_private_hands(player)
    hist = list(game.history())
    bets = game.total_bet_amount()
    cards = oop_cards if player == 0 else ip_cards
    pname = "OOP" if player == 0 else "IP"

    strategy_flat = predict_strategy(cards, player, hist, bets, n_act, n_hands)
    game.lock_current_strategy(strategy_flat)
    locked_count += 1
    print(f"  Locked [{pname}] history={hist} actions={actions}")

    for i in range(len(actions)):
        game.play(i)
        lock_flop_nodes()
        game.back_to_root()
        for a in hist:
            game.play(a)

game.back_to_root()
lock_flop_nodes()
print(f"\nLocked {locked_count} flop nodes with NN strategies")
print("Turn/river nodes: uniform (default)")

# ── Finalize and save ──
game.finalize()

if output_path is None:
    basename = os.path.basename(config_path).replace(".json", "")
    output_path = f"data/out/{basename}-nn.flop"

os.makedirs(os.path.dirname(output_path), exist_ok=True)
game.save_to_file(output_path, "nn-flop")
print(f"\nSaved: {output_path}")
