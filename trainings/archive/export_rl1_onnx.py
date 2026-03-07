#!/usr/bin/env python3
"""
Export an RL1 checkpoint (.pt) to ONNX format + meta.json.

Usage:
  python trainings/export_rl1_onnx.py config/9s6d6c.json models/rl1_9s6d6c/best.pt
"""

import argparse
import json
import os

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


def main():
    parser = argparse.ArgumentParser(description="Export RL1 model to ONNX")
    parser.add_argument("config", help="Path to config JSON")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as checkpoint)")
    args = parser.parse_args()

    # Get game dimensions from config
    game = postflop_solver.GameWrapper(args.config)
    num_boundaries = game.num_boundaries()
    num_hands = [game.num_private_hands(0), game.num_private_hands(1)]
    max_hands = max(num_hands)
    max_iters = game.max_iterations()

    in_dim = num_boundaries + 1 + max_hands
    out_dim = max_hands

    # Load model
    dev = torch.device("cpu")
    model = Net(in_dim, out_dim).to(dev)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Output directory
    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    # Export ONNX
    onnx_path = os.path.join(out_dir, "model.onnx")
    dummy_input = torch.zeros(num_boundaries, in_dim)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX: {onnx_path}")

    # Save meta.json
    meta = {
        "num_boundaries": num_boundaries,
        "num_iterations": max_iters,
        "y_scale": 1.0,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "max_hands": max_hands,
        "num_oop": num_hands[0],
        "num_ip": num_hands[1],
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported meta: {meta_path}")

    # Summary
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {n_params:,}")
    print(f"Input dim: {in_dim}, Output dim: {out_dim}")
    print(f"Boundaries: {num_boundaries}")
    print(f"From episode: {ckpt.get('episode', '?')}")


if __name__ == "__main__":
    main()
