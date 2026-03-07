#!/usr/bin/env python3
"""Convert all .pt models to .onnx for benchmark."""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_presentation import TurnValueNetwork

MODELS = {
    "10k":  "models/10k/best_model.pt",
    "25k":  "models/25k/best_model.pt",
    "50k":  "models/50k/best_model.pt",
    "100k": "models/100k/best_model.pt",
}

OUTPUT_DIR = "models/benchmark_onnx"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, pt_path in MODELS.items():
        if not os.path.exists(pt_path):
            print(f"SKIP {name}: {pt_path} not found")
            continue

        print(f"Converting {name}: {pt_path}")
        model = TurnValueNetwork(hidden_dim=500, num_layers=7, dropout=0.0)
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        model.eval()

        onnx_path = os.path.join(OUTPUT_DIR, f"model_{name}.onnx")
        torch.onnx.export(
            model,
            torch.randn(1, 2015),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            dynamo=False,  # single .onnx file (no external .onnx.data)
        )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  -> {onnx_path} ({size_mb:.1f} MB)")

    print("\nDone. ONNX models ready for benchmark.")


if __name__ == "__main__":
    main()
