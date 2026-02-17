"""
CFV prediction model v2 — optimized for inference speed.

Same I/O as model_1 for training pipeline compatibility:
  Input:  combo_features (batch, 1326, 19) + global_features (batch, 20)
  Output: cfv (batch, 1326, 2) — cfv_oop and cfv_ip per combo

Changes from model_1:
  - Hidden dim: 256 → 128
  - Residual blocks: 7 → 3
  - Activation: GELU → ReLU
  - LayerNorm: everywhere → encoders only
  - RangeAggregation: removed (expensive mean-pool across 1326 combos)
  - Params: ~2M → ~236K

Zero-sum enforced: cfv_oop + cfv_ip = 0 for each combo.
"""
import torch
import torch.nn as nn
import numpy as np

NUM_COMBOS = 1326
PER_COMBO_FEATURES = 19
GLOBAL_FEATURES = 20
OUTPUT_DIM = 2


class ResidualBlock(nn.Module):
    """Lightweight residual block: Linear → ReLU → Linear + skip. No LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class CFVModel(nn.Module):
    """
    Fast CFV prediction model.

    Architecture:
        1. ComboEncoder:  19 → 128 (per-combo features)
        2. GlobalEncoder: 20 → 128 (board/game context)
        3. Merge:         256 → 128
        4. ResidualBlocks × 3
        5. OutputHead:    128 → 1 (raw value per combo)
        6. ZeroSum:       [raw, -raw]
    """

    def __init__(self, hidden=128, num_blocks=3):
        super().__init__()

        self.combo_encoder = nn.Sequential(
            nn.Linear(PER_COMBO_FEATURES, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.merge = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden) for _ in range(num_blocks)
        ])

        self.output_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, combo_features, global_features):
        # Encode per-combo: (batch, 1326, 19) → (batch, 1326, 128)
        combo_enc = self.combo_encoder(combo_features)

        # Encode global: (batch, 20) → (batch, 128) → broadcast (batch, 1326, 128)
        global_enc = self.global_encoder(global_features)
        global_enc = global_enc.unsqueeze(1).expand(-1, NUM_COMBOS, -1)

        # Merge: (batch, 1326, 256) → (batch, 1326, 128)
        x = self.merge(torch.cat([combo_enc, global_enc], dim=-1))

        # Residual tower
        for block in self.blocks:
            x = block(x)

        # Output: (batch, 1326, 128) → (batch, 1326, 1)
        raw = self.output_head(x)

        # Zero-sum enforcement
        cfv = torch.cat([raw, -raw], dim=-1)
        return cfv


FIXED_BATCH = 49  # max turn cards = 52 - 3 flop


def export_onnx(model, path="model_2.onnx"):
    import onnx

    model.eval()
    # Fixed batch=49 for CoreML compatibility (no dynamic axes)
    combo_features = torch.randn(FIXED_BATCH, NUM_COMBOS, PER_COMBO_FEATURES)
    global_features = torch.randn(FIXED_BATCH, GLOBAL_FEATURES)

    torch.onnx.export(
        model,
        (combo_features, global_features),
        path,
        input_names=["combo_features", "global_features"],
        output_names=["cfv"],
        opset_version=17,
    )

    onnx_model = onnx.load(path, load_external_data=True)
    onnx.save(onnx_model, path, save_as_external_data=False)

    import os
    data_path = path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)

    print(f"Exported ONNX model to: {path}")


if __name__ == "__main__":
    model = CFVModel()
    total_params = sum(p.numel() for p in model.parameters())

    print("=" * 60)
    print("CFV Model v2 — Fast Architecture (untrained)")
    print("=" * 60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Input:  combo_features (batch, {NUM_COMBOS}, {PER_COMBO_FEATURES})")
    print(f"        global_features (batch, {GLOBAL_FEATURES})")
    print(f"Output: cfv (batch, {NUM_COMBOS}, {OUTPUT_DIM})")
    print()
    print("Architecture:")
    print(f"  ComboEncoder:     {PER_COMBO_FEATURES} → 128 (LayerNorm + ReLU)")
    print(f"  GlobalEncoder:    {GLOBAL_FEATURES} → 128 (LayerNorm + ReLU)")
    print(f"  Merge:            256 → 128 (ReLU)")
    print(f"  ResidualBlocks:   3 × (128 → 256 → 128 + skip)")
    print(f"  OutputHead:       128 → 1")
    print(f"  ZeroSum:          [raw, -raw]")
    print()

    # Forward pass test (fixed batch=49)
    combo_features = torch.randn(FIXED_BATCH, NUM_COMBOS, PER_COMBO_FEATURES)
    global_features = torch.randn(FIXED_BATCH, GLOBAL_FEATURES)
    with torch.no_grad():
        cfv = model(combo_features, global_features)

    print(f"Forward pass OK: {cfv.shape}")

    # Zero-sum check
    zero_sum_error = (cfv[:, :, 0] + cfv[:, :, 1]).abs().max().item()
    print(f"Zero-sum check: max |cfv_oop + cfv_ip| = {zero_sum_error:.10f}")
    print()

    # Export
    export_onnx(model, "model_2.onnx")

    # Verify ONNX
    try:
        import onnxruntime as ort
        session = ort.InferenceSession("model_2.onnx")
        result = session.run(None, {
            "combo_features": combo_features.numpy(),
            "global_features": global_features.numpy(),
        })
        max_diff = np.max(np.abs(result[0] - cfv.numpy()))
        print(f"ONNX verification OK (max diff: {max_diff:.8f})")
        print(f"  ONNX output shape: {result[0].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping ONNX verification")
