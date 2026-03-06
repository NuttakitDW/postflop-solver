#!/usr/bin/env python3
"""
Solve a flop game using a trained DEVN model and save the result as a .flop file.

The model predicts boundary EVs at each DCFR iteration (flop-only traversal).
EVs are multiplied by matchups to recover CFVs for the solver.
This is the inference counterpart to train_devn.py.

Usage:
  python trainings/solve_with_devn.py config/9s6d6c.json models/devn_9s6d6c/best.pt
  python trainings/solve_with_devn.py config/9s6d6c.json models/devn_9s6d6c/best.pt --output data/out/9s6d6c-devn.flop
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import postflop_solver

# ─── model (must match train_devn.py) ───
HIDDEN = 500
LAYERS = 7


class Net(nn.Module):
    def __init__(self, in_dim, out_dim, h=HIDDEN, n_layers=LAYERS):
        super().__init__()
        seq, d = [], in_dim
        for _ in range(n_layers):
            seq += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        seq.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


# ─── DEVN helpers ───
def build_conflict_matrix(hero_cards, opp_cards):
    """Returns bool matrix [num_hero_hands, num_opp_hands].
    True = conflict (share at least one card)."""
    nh = len(hero_cards)
    no = len(opp_cards)
    conflict = np.zeros((nh, no), dtype=bool)
    for i, (h1, h2) in enumerate(hero_cards):
        for j, (o1, o2) in enumerate(opp_cards):
            if h1 == o1 or h1 == o2 or h2 == o1 or h2 == o2:
                conflict[i, j] = True
    return conflict


def compute_matchups(cfreaches, conflict_matrix):
    """Compute matchup[b][h] = sum of non-conflicting opponent cfreaches."""
    opp = np.stack([np.array(cr, dtype=np.float32) for cr in cfreaches])
    valid = (~conflict_matrix).astype(np.float32)
    return opp @ valid.T


def build_input_batch(cfreaches, player, num_boundaries, max_hands, device):
    n = len(cfreaches)
    in_dim = num_boundaries + 1 + max_hands
    inputs = torch.zeros(n, in_dim, device=device)
    for b, cr in enumerate(cfreaches):
        inputs[b, b] = 1.0
        inputs[b, num_boundaries] = float(player)
        inputs[b, num_boundaries + 1:num_boundaries + 1 + len(cr)] = torch.tensor(cr)
    return inputs


def tensor_to_cfv_lists(predicted, player, num_hands):
    nh = num_hands[player]
    pred_np = predicted.detach().cpu().numpy()
    return [pred_np[b, :nh].tolist() for b in range(pred_np.shape[0])]


def main():
    parser = argparse.ArgumentParser(description="Solve flop with DEVN model")
    parser.add_argument("config", help="Path to config JSON")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--output", default=None, help="Output .flop path")
    args = parser.parse_args()

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    output_path = args.output or f"data/out/{config_name}-devn.flop"

    dev = torch.device("cpu")  # CPU for inference (no grad needed)

    print(f"=== Solve with DEVN Model ===")
    print(f"Config: {args.config}")
    print(f"Model:  {args.checkpoint}")
    print(f"Output: {output_path}")
    print()

    # Setup game
    total_start = time.time()
    game = postflop_solver.GameWrapper(args.config)
    num_boundaries = game.num_boundaries()
    num_hands = [game.num_private_hands(0), game.num_private_hands(1)]
    max_hands = max(num_hands)
    max_iters = game.max_iterations()
    pot = game.starting_pot()

    in_dim = num_boundaries + 1 + max_hands
    out_dim = max_hands

    print(f"Boundaries: {num_boundaries}")
    print(f"OOP hands: {num_hands[0]}, IP hands: {num_hands[1]}")
    print(f"Iterations: {max_iters}")
    print(f"Pot: {pot}")
    print()

    # Build conflict matrices
    hero_cards = [game.private_cards(0), game.private_cards(1)]
    opp_cards = [game.private_cards(1), game.private_cards(0)]
    conflict = [
        build_conflict_matrix(hero_cards[0], opp_cards[0]),
        build_conflict_matrix(hero_cards[1], opp_cards[1]),
    ]

    # Load model
    model = Net(in_dim, out_dim).to(dev)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Trained episodes: {ckpt.get('episode', '?')}")
    print()

    # Solve (flop-only via replay)
    print(f"Solving ({max_iters} iterations)...")
    solve_start = time.time()

    with torch.no_grad():
        for t in range(max_iters):
            for player in range(2):
                cfreaches = game.collect_boundary_cfreaches(player)
                matchups = compute_matchups(cfreaches, conflict[player])
                matchup_t = torch.tensor(matchups, dtype=torch.float32, device=dev)

                inputs = build_input_batch(cfreaches, player, num_boundaries, max_hands, dev)
                predicted_ev = model(inputs)

                # EV → CFV
                nh = num_hands[player]
                matchup_padded = torch.zeros(len(cfreaches), max_hands, device=dev)
                matchup_padded[:, :nh] = matchup_t
                predicted_cfv = predicted_ev * matchup_padded

                model_cfvs = tensor_to_cfv_lists(predicted_cfv, player, num_hands)
                game.solve_step_replay(t, player, model_cfvs)

            if (t + 1) % 10 == 0 or t + 1 == max_iters:
                elapsed = time.time() - solve_start
                per_iter = elapsed / (t + 1)
                print(f"\r  iter {t+1}/{max_iters}  ({elapsed:.1f}s, {per_iter:.3f}s/iter)", end="", flush=True)

    solve_time = time.time() - solve_start
    print()

    # Finalize and save
    print("\nFinalizing...")
    game.finalize()

    print(f"Saving to {output_path}...")
    game.save_to_file(output_path, "devn")

    total_time = time.time() - total_start
    print()
    print(f"=== Done ===")
    print(f"Solve time: {solve_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
