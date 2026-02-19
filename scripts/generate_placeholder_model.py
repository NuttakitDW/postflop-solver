#!/usr/bin/env python3
"""
Generate a placeholder ONNX model matching the bucketed value network spec.

Input:  [batch, 2015] = board(15) + range_oop(1000) + range_ip(1000)
Output: [batch, 2000] = cfv_oop(1000) + cfv_ip(1000)

Architecture matches train.py: MLP with PReLU + ZeroSumCorrectionLayer.
Weights are random (Kaiming default) — use for pipeline testing only.

Usage:
    python scripts/generate_placeholder_model.py -o models/model_placeholder.onnx
    python scripts/generate_placeholder_model.py -o models/model_placeholder.onnx --hidden-dim 128 --num-layers 3
"""

import argparse
import os
import sys

# Import model classes from train.py
sys.path.insert(0, os.path.dirname(__file__))
from train import TurnValueNetwork

import torch


def main():
    parser = argparse.ArgumentParser(description="Generate placeholder ONNX model")
    parser.add_argument("-o", "--output", default="models/model_placeholder.onnx", help="Output path")
    parser.add_argument("--hidden-dim", type=int, default=500)
    parser.add_argument("--num-layers", type=int, default=7)
    args = parser.parse_args()

    model = TurnValueNetwork(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_layers} layers x {args.hidden_dim} hidden, {num_params:,} params")

    # Dummy input matching spec: [batch, 2015]
    dummy_input = torch.randn(1, 2015)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,  # single .onnx file (no external .onnx.data)
    )

    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Exported to {args.output} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
