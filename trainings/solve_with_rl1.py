#!/usr/bin/env python3
"""
Solve a flop game using a trained RL1 model and save the result as a .flop file.

The model predicts boundary CFVs at each DCFR iteration (flop-only traversal).
This is the inference counterpart to train_rl1.py.

Usage:
  python trainings/solve_with_rl1.py config/9s6d6c.json models/rl1_9s6d6c/best.pt
  python trainings/solve_with_rl1.py config/9s6d6c.json models/rl1_9s6d6c/best.pt --output data/out/9s6d6c-rl1.flop
"""

import argparse
import os
import time

import torch
import torch.nn as nn

import postflop_solver

# ─── model (must match train_rl1.py) ───
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
    parser = argparse.ArgumentParser(description="Solve flop with RL1 model")
    parser.add_argument("config", help="Path to config JSON")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--output", default=None, help="Output .flop path")
    args = parser.parse_args()

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    output_path = args.output or f"data/out/{config_name}-rl1.flop"

    dev = torch.device("cpu")  # CPU for inference (no grad needed)

    print(f"=== Solve with RL1 Model ===")
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
                inputs = build_input_batch(cfreaches, player, num_boundaries, max_hands, dev)
                predicted = model(inputs)
                model_cfvs = tensor_to_cfv_lists(predicted, player, num_hands)
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
    game.save_to_file(output_path, "rl1")

    total_time = time.time() - total_start
    print()
    print(f"=== Done ===")
    print(f"Solve time: {solve_time:.1f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
