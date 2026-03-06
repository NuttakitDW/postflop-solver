"""
Inference: predict OOP flop root strategy using a trained NN model.

Usage:
  python infer_nn_strategy.py <model.pt>              # print all hands
  python infer_nn_strategy.py <model.pt> KcKd          # single hand
  python infer_nn_strategy.py <model.pt> KcKd AcAh QQ  # multiple hands
"""

import sys
import torch
import torch.nn as nn
import numpy as np

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

def parse_hand(s):
    """Parse hand like 'KcKd' or 'QQ' (all combos)."""
    if len(s) == 4:
        return [(parse_card(s[:2]), parse_card(s[2:]))]
    elif len(s) == 2:
        # Pair like 'QQ' — return all combos
        r = RANKS.index(s[0].upper())
        combos = []
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                combos.append((r * 4 + s1, r * 4 + s2))
        return combos
    else:
        print(f"Cannot parse hand: {s}")
        sys.exit(1)

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

if len(sys.argv) < 2:
    print("Usage: python infer_nn_strategy.py <model.pt> [hand ...]")
    sys.exit(1)

# Load model
ckpt = torch.load(sys.argv[1], weights_only=False)
actions = ckpt["actions"]
cards = ckpt["cards"]
n_act = ckpt["n_act"]
n_h = ckpt["n_hands"]

model = StrategyNet(104, 256, n_act)
model.load_state_dict(ckpt["model"])
model.eval()

def predict(c1, c2):
    x = torch.zeros(1, 104)
    x[0, c1] = 1.0
    x[0, 52 + c2] = 1.0
    with torch.no_grad():
        probs = model(x)[0].numpy()
    return probs

# Determine which hands to show
if len(sys.argv) > 2:
    # Specific hands requested
    for arg in sys.argv[2:]:
        combos = parse_hand(arg)
        for c1, c2 in combos:
            probs = predict(c1, c2)
            parts = [f"{actions[j]}={probs[j]:.3f}" for j in range(n_act)]
            print(f"  {hand_str(c1, c2):5s}: {', '.join(parts)}")
else:
    # All hands
    print(f"Model: {sys.argv[1]}")
    print(f"{n_h} hands, actions: {actions}\n")
    for c1, c2 in cards:
        probs = predict(c1, c2)
        parts = [f"{actions[j]}={probs[j]:.3f}" for j in range(n_act)]
        print(f"  {hand_str(c1, c2):5s}: {', '.join(parts)}")
