#!/usr/bin/env python3
"""
Export toy_20bb model to ONNX with scaling baked in.

The solver passes raw inputs (unscaled board features, raw bucketed ranges)
and expects raw CFV outputs. This script wraps the trained model with:
  - Input: normalize board features using saved mu/std
  - Output: denormalize CFVs by multiplying by y_scale

Usage:
  python scripts/export_toy_onnx.py
"""

import os, torch, torch.nn as nn

K, BOARD_FEATS = 1000, 15
IN_DIM, OUT_DIM = BOARD_FEATS + 2*K, 2*K

BASE = os.path.dirname(__file__)
CHECKPOINT = os.path.join(BASE, "..", "models", "toy_20bb", "best_ema.pt")
ONNX_OUT   = os.path.join(BASE, "..", "models", "toy_20bb", "model.onnx")

# -------- model (must match train_toy.py) ----------
class ZeroSum(nn.Module):
    def forward(self, raw, r_oop, r_ip):
        cfv_oop, cfv_ip = raw[:, :K], raw[:, K:]
        g = (r_oop*cfv_oop).sum(1, keepdim=True) + (r_ip*cfv_ip).sum(1, keepdim=True)
        corr = g/2
        return torch.cat([cfv_oop - corr/r_oop.sum(1, keepdim=True).clamp(1e-8),
                          cfv_ip  - corr/r_ip .sum(1, keepdim=True).clamp(1e-8)], 1)

class Net(nn.Module):
    def __init__(self, h=500, n_layers=7, drop=0.):
        super().__init__()
        seq, d = [], IN_DIM
        for _ in range(n_layers):
            seq += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            if drop: seq.append(nn.Dropout(drop))
            d = h
        seq.append(nn.Linear(h, OUT_DIM))
        self.backbone, self.zs = nn.Sequential(*seq), ZeroSum()
    def forward(self, x):
        raw = self.backbone(x)
        return self.zs(raw,
                       x[:, BOARD_FEATS:BOARD_FEATS+K],
                       x[:, BOARD_FEATS+K:])

# -------- wrapper with baked-in scaling ----------
class ScaledNet(nn.Module):
    def __init__(self, net, mu, std, y_scale):
        super().__init__()
        self.net = net
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.y_scale = float(y_scale)

    def forward(self, x):
        # Normalize board features (first 15 dims)
        x_scaled = x.clone()
        x_scaled[:, :BOARD_FEATS] = (x[:, :BOARD_FEATS] - self.mu) / self.std
        # Forward through model
        y_scaled = self.net(x_scaled)
        # Denormalize output
        return y_scaled * self.y_scale

def main():
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    # Extract scalers
    mu = ckpt.pop("_mu")
    std = ckpt.pop("_std")
    y_scale = ckpt.pop("_yscale")
    print(f"  mu shape: {mu.shape}, std shape: {std.shape}, y_scale: {y_scale:.6f}")

    # Load EMA weights into model
    net = Net(500, 7, 0.0)
    net.load_state_dict(ckpt)
    net.eval()

    # Wrap with scaling
    model = ScaledNet(net, mu, std, y_scale)
    model.eval()

    # Export to ONNX
    dummy = torch.randn(1, IN_DIM)
    torch.onnx.export(
        model,
        dummy,
        ONNX_OUT,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )

    size_mb = os.path.getsize(ONNX_OUT) / (1024 * 1024)
    print(f"Exported: {ONNX_OUT} ({size_mb:.1f} MB)")

    # Quick sanity check
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_OUT)
    out = sess.run(None, {"input": dummy.numpy()})
    print(f"Sanity check: input {dummy.shape} -> output {out[0].shape}")
    print(f"Output range: [{out[0].min():.4f}, {out[0].max():.4f}]")
    print("Done!")

if __name__ == "__main__":
    main()
