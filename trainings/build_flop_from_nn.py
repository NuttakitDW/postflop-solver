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
import numpy as np
import torch
import torch.nn as nn
import postflop_solver

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_str(cid):
    return f"{RANKS[cid // 4]}{SUITS[cid % 4]}"

def hand_str(c1, c2):
    return f"{card_str(c1)}{card_str(c2)}"

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

model = FlopStrategyNet(input_dim, 256, max_actions)
model.load_state_dict(ckpt["model"])
model.eval()

# ── Create fresh game ──
game = postflop_solver.GameWrapper(config_path)
oop_cards = game.private_cards(0)
ip_cards = game.private_cards(1)
n_oop = len(oop_cards)
n_ip = len(ip_cards)
print(f"Config: {config_path}")
print(f"OOP hands: {n_oop}, IP hands: {n_ip}")

def predict_strategy(cards, player, history, bets, n_act, n_hands):
    """Predict strategy for all hands at a node. Returns flat [n_act * n_hands]."""
    history_dim = max_history * max_actions
    pot = starting_pot + bets[0] + bets[1]
    pot_frac = pot / (starting_pot + 360)

    # Encode history once
    hist_enc = np.zeros(history_dim, dtype=np.float32)
    for step, action_idx in enumerate(history):
        if step < max_history and action_idx < max_actions:
            hist_enc[step * max_actions + action_idx] = 1.0

    # Build batch input for all hands
    X = np.zeros((n_hands, input_dim), dtype=np.float32)
    for h in range(n_hands):
        c1, c2 = cards[h]
        X[h, c1] = 1.0
        X[h, 52 + c2] = 1.0
        X[h, 104] = float(player)
        X[h, 105:105 + history_dim] = hist_enc
        X[h, 105 + history_dim] = pot_frac

    with torch.no_grad():
        probs = model(torch.from_numpy(X)).numpy()  # [n_hands, max_actions]

    # Build flat strategy: [n_act * n_hands] row-major (action, hand)
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

    # Predict and lock
    strategy_flat = predict_strategy(cards, player, hist, bets, n_act, n_hands)
    game.lock_current_strategy(strategy_flat)
    locked_count += 1
    print(f"  Locked [{pname}] history={hist} actions={actions}")

    # Recurse into children
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
