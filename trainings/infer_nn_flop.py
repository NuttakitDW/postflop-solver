"""
Inference: play through the full flop using a trained NN model.

Usage:
  python infer_nn_flop.py <model_flop.pt>              # interactive mode
  python infer_nn_flop.py <model_flop.pt> KcKd          # show strategy at all nodes for one hand
"""

import sys
import numpy as np
import torch
import torch.nn as nn

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_str(cid):
    return f"{RANKS[cid // 4]}{SUITS[cid % 4]}"

def hand_str(c1, c2):
    return f"{card_str(c1)}{card_str(c2)}"

def parse_card(s):
    r = RANKS.index(s[0].upper())
    su = SUITS.index(s[1].lower())
    return r * 4 + su

def parse_hand_input(s):
    s = s.strip()
    if len(s) == 4:
        return (parse_card(s[:2]), parse_card(s[2:]))
    elif len(s) == 2:
        # Pair — pick first combo
        r = RANKS.index(s[0].upper())
        return (r * 4, r * 4 + 1)
    else:
        return None

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

if len(sys.argv) < 2:
    print("Usage: python infer_nn_flop.py <model_flop.pt> [hand]")
    sys.exit(1)

# Load model
ckpt = torch.load(sys.argv[1], weights_only=False)
input_dim = ckpt["input_dim"]
max_actions = ckpt["max_actions"]
max_history = ckpt["max_history"]
starting_pot = ckpt["starting_pot"]
model_nodes = ckpt["nodes"]

model = FlopStrategyNet(input_dim, 256, max_actions)
model.load_state_dict(ckpt["model"])
model.eval()

# Build node lookup: history tuple -> node info
node_map = {}
for n in model_nodes:
    node_map[tuple(n["history"])] = n

def predict(c1, c2, player, history, bets):
    x = torch.zeros(1, input_dim)
    x[0, c1] = 1.0
    x[0, 52 + c2] = 1.0
    x[0, 104] = float(player)
    history_dim = max_history * max_actions
    for step, action_idx in enumerate(history):
        if step < max_history and action_idx < max_actions:
            x[0, 105 + step * max_actions + action_idx] = 1.0
    pot = starting_pot + bets[0] + bets[1]
    x[0, 105 + history_dim] = pot / (starting_pot + 360)
    with torch.no_grad():
        probs = model(x)[0].numpy()
    return probs

def show_all_nodes(c1, c2):
    """Show NN strategy at every flop node for a given hand."""
    print(f"\nHand: {hand_str(c1, c2)}")
    print(f"{'='*60}")
    for n in model_nodes:
        player = n["player"]
        pname = "OOP" if player == 0 else "IP"
        hist = n["history"]
        actions = n["actions"]
        bets = n["bets"]
        n_act = n["n_act"]
        probs = predict(c1, c2, player, hist, bets)
        parts = [f"{actions[j]}={probs[j]:.3f}" for j in range(n_act)]
        indent = "  " * len(hist)
        print(f"  {indent}[{pname}] {' -> '.join(str(a) for a in hist) or 'root'}: {', '.join(parts)}")

# ── Mode: show all nodes for a hand ──
if len(sys.argv) > 2:
    hand = parse_hand_input(sys.argv[2])
    if hand is None:
        print(f"Cannot parse hand: {sys.argv[2]}")
        sys.exit(1)
    show_all_nodes(hand[0], hand[1])
    sys.exit(0)

# ── Mode: interactive play ──
print(f"Loaded model with {len(model_nodes)} flop nodes")
print(f"Enter a hand to see all nodes, or 'q' to quit.\n")

while True:
    try:
        inp = input("Hand (e.g. KcKd): ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if inp.lower() in ("q", "quit", "exit"):
        break
    hand = parse_hand_input(inp)
    if hand is None:
        print("  Cannot parse. Use format like KcKd or QQ")
        continue
    show_all_nodes(hand[0], hand[1])
