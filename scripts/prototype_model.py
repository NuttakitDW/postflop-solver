"""
CFV prediction model for turn boundary (production architecture, untrained).

Input:  combo_features (batch, 1326, 19) + global_features (batch, 20)
Output: cfv (batch, 1326, 2) — cfv_oop and cfv_ip per combo

Zero-sum enforced: cfv_oop + cfv_ip = 0 for each combo.
"""
import torch
import torch.nn as nn
import numpy as np

# Feature dimensions
NUM_COMBOS = 1326
PER_COMBO_FEATURES = 19
GLOBAL_FEATURES = 20
OUTPUT_DIM = 2  # cfv_oop, cfv_ip


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout + skip."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class RangeAggregation(nn.Module):
    """
    Aggregate information across all combos to capture range-level context.
    Each combo's CFV depends on the ENTIRE opponent range, not just its own features.
    This computes reach-weighted stats across combos and feeds them back per-combo.
    """

    def __init__(self, dim, agg_dim=64):
        super().__init__()
        # Project to aggregation space
        self.proj = nn.Linear(dim, agg_dim)
        # Merge aggregated context back
        self.norm = nn.LayerNorm(dim + agg_dim)
        self.merge = nn.Linear(dim + agg_dim, dim)

    def forward(self, x):
        # x: (batch, 1326, dim)
        agg = self.proj(x)             # (batch, 1326, agg_dim)
        agg_mean = agg.mean(dim=1, keepdim=True)  # (batch, 1, agg_dim)
        agg_broadcast = agg_mean.expand_as(agg)    # (batch, 1326, agg_dim)
        merged = torch.cat([x, agg_broadcast], dim=-1)  # (batch, 1326, dim + agg_dim)
        return self.merge(self.norm(merged))  # (batch, 1326, dim)


class CFVModel(nn.Module):
    """
    Production CFV prediction model.

    Architecture:
        1. ComboEncoder:  19 -> 256 (per-combo features)
        2. GlobalEncoder: 20 -> 256 (board/game context)
        3. Merge:         512 -> 256 (project down)
        4. RangeAggregation: capture cross-combo interactions
        5. ResidualBlocks × 6: deep processing
        6. OutputHead:    256 -> 1 (raw value per combo)
        7. ZeroSum:       split into cfv_oop = raw, cfv_ip = -raw
    """

    def __init__(self, hidden=256, num_blocks=6, dropout=0.1):
        super().__init__()

        # --- Encoders ---
        self.combo_encoder = nn.Sequential(
            nn.Linear(PER_COMBO_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # --- Merge (concat 512 -> 256) ---
        self.merge = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        # --- Range aggregation (cross-combo context) ---
        self.range_agg = RangeAggregation(hidden, agg_dim=64)

        # --- Residual tower ---
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # --- Output head (predict single raw value per combo) ---
        self.output_norm = nn.LayerNorm(hidden)
        self.output_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for stable untrained outputs."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, combo_features, global_features):
        """
        Args:
            combo_features:  (batch, 1326, 19)
            global_features: (batch, 20)

        Returns:
            cfv: (batch, 1326, 2) where cfv[:,:,0] + cfv[:,:,1] = 0 (zero-sum)
        """
        # Encode per-combo: (batch, 1326, 19) -> (batch, 1326, 256)
        combo_enc = self.combo_encoder(combo_features)

        # Encode global: (batch, 20) -> (batch, 256) -> broadcast (batch, 1326, 256)
        global_enc = self.global_encoder(global_features)
        global_enc = global_enc.unsqueeze(1).expand(-1, NUM_COMBOS, -1)

        # Merge: (batch, 1326, 512) -> (batch, 1326, 256)
        x = self.merge(torch.cat([combo_enc, global_enc], dim=-1))

        # Range aggregation: cross-combo context
        x = self.range_agg(x)

        # Residual tower
        for block in self.blocks:
            x = block(x)

        # Output: (batch, 1326, 256) -> (batch, 1326, 1)
        raw = self.output_head(self.output_norm(x))  # (batch, 1326, 1)

        # Zero-sum enforcement: cfv_oop = raw, cfv_ip = -raw
        cfv = torch.cat([raw, -raw], dim=-1)  # (batch, 1326, 2)

        return cfv


def create_dummy_input(batch_size=1):
    """Create fake input matching the feature spec."""
    combo_features = torch.randn(batch_size, NUM_COMBOS, PER_COMBO_FEATURES)
    global_features = torch.randn(batch_size, GLOBAL_FEATURES)
    return combo_features, global_features


def export_onnx(model, path="prototype_cfv_model.onnx"):
    """Export model to single ONNX file (graph + weights together)."""
    import onnx

    model.eval()
    combo_features, global_features = create_dummy_input(batch_size=1)

    torch.onnx.export(
        model,
        (combo_features, global_features),
        path,
        input_names=["combo_features", "global_features"],
        output_names=["cfv"],
        dynamic_axes={
            "combo_features": {0: "batch"},
            "global_features": {0: "batch"},
            "cfv": {0: "batch"},
        },
        opset_version=17,
    )

    # Merge external data into single file
    onnx_model = onnx.load(path, load_external_data=True)
    onnx.save(onnx_model, path, save_as_external_data=False)

    # Clean up leftover .data file
    import os
    data_path = path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)

    print(f"Exported ONNX model to: {path}")


if __name__ == "__main__":
    model = CFVModel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("CFV Model — Production Architecture (untrained)")
    print("=" * 60)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input:  combo_features (batch, {NUM_COMBOS}, {PER_COMBO_FEATURES})")
    print(f"        global_features (batch, {GLOBAL_FEATURES})")
    print(f"Output: cfv (batch, {NUM_COMBOS}, {OUTPUT_DIM})")
    print()

    # Architecture summary
    print("Architecture:")
    print(f"  ComboEncoder:     {PER_COMBO_FEATURES} -> 256")
    print(f"  GlobalEncoder:    {GLOBAL_FEATURES} -> 256")
    print(f"  Merge:            512 -> 256")
    print(f"  RangeAggregation: cross-combo context")
    print(f"  ResidualBlocks:   6 x (LayerNorm -> 256 -> 512 -> 256 + skip)")
    print(f"  OutputHead:       256 -> 1 (raw value)")
    print(f"  ZeroSum:          raw -> [raw, -raw]")
    print()

    # Forward pass test
    combo_features, global_features = create_dummy_input(batch_size=1)
    with torch.no_grad():
        cfv = model(combo_features, global_features)

    print(f"Forward pass OK")
    print(f"  combo_features shape: {combo_features.shape}")
    print(f"  global_features shape: {global_features.shape}")
    print(f"  output cfv shape:      {cfv.shape}")
    print()

    # Verify zero-sum
    zero_sum_error = (cfv[:, :, 0] + cfv[:, :, 1]).abs().max().item()
    print(f"Zero-sum check: max |cfv_oop + cfv_ip| = {zero_sum_error:.10f}")
    print()

    # Sample outputs
    print(f"Sample outputs (first 5 combos):")
    print(f"  cfv_oop: {cfv[0, :5, 0].numpy()}")
    print(f"  cfv_ip:  {cfv[0, :5, 1].numpy()}")
    print()

    # Batch test
    combo_batch, global_batch = create_dummy_input(batch_size=4)
    with torch.no_grad():
        cfv_batch = model(combo_batch, global_batch)
    print(f"Batch forward pass OK: input batch=4 -> output {cfv_batch.shape}")
    print()

    # Export ONNX
    export_onnx(model, "prototype_cfv_model.onnx")

    # Verify ONNX
    try:
        import onnxruntime as ort
        session = ort.InferenceSession("prototype_cfv_model.onnx")
        combo_np, global_np = combo_features.numpy(), global_features.numpy()
        result = session.run(None, {
            "combo_features": combo_np,
            "global_features": global_np,
        })
        max_diff = np.max(np.abs(result[0] - cfv.numpy()))
        print(f"ONNX verification OK (max diff: {max_diff:.8f})")
        print(f"  ONNX output shape: {result[0].shape}")
    except ImportError:
        print("onnxruntime not installed, skipping ONNX verification")
