#!/usr/bin/env python3
"""
verify_bt2.py — Verify bt2 model predictions against ground truth .bt2 data.

Loads .bt2 files and the trained ONNX model, feeds the exact training inputs
through the model, and compares predicted CFVs vs recorded CFVs.

Plots per-boundary and per-iteration error to check if the model has overfit.

Usage:
  python trainings/verify_bt2.py models/bt2_KcQh7s data/bt2/KcQh7s_f*.bt2
"""

import struct, sys, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

# -------- .bt2 loader (same as train_bt2.py) --------
def load_bt2(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"BT2\0\0\0\0\0", f"Bad magic: {magic}"
        _version = struct.unpack("<I", f.read(4))[0]
        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_boundaries = struct.unpack("<I", f.read(4))[0]
        num_iterations = struct.unpack("<I", f.read(4))[0]
        starting_pot = struct.unpack("<f", f.read(4))[0]
        effective_stack = struct.unpack("<f", f.read(4))[0]

        boundary_pots, boundary_stacks = [], []
        for _ in range(num_boundaries):
            boundary_pots.append(struct.unpack("<f", f.read(4))[0])
            boundary_stacks.append(struct.unpack("<f", f.read(4))[0])

        num_hands = [num_oop, num_ip]
        records = []
        for _ in range(num_iterations):
            _iteration = struct.unpack("<I", f.read(4))[0]
            _exploitability = struct.unpack("<f", f.read(4))[0]
            _reserved = struct.unpack("<I", f.read(4))[0]
            boundaries = []
            for _ in range(num_boundaries):
                player_data = []
                for player in range(2):
                    opponent = player ^ 1
                    cfv = np.frombuffer(f.read(num_hands[player] * 4), dtype=np.float32).copy()
                    cfreach = np.frombuffer(f.read(num_hands[opponent] * 4), dtype=np.float32).copy()
                    player_data.append((cfv, cfreach))
                boundaries.append(player_data)
            records.append(boundaries)

    return {
        "records": records,
        "num_oop": num_oop, "num_ip": num_ip,
        "num_boundaries": num_boundaries,
        "starting_pot": starting_pot,
        "effective_stack": effective_stack,
        "boundary_pots": boundary_pots,
        "boundary_stacks": boundary_stacks,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python trainings/verify_bt2.py <model_dir> <file1.bt2> [file2.bt2] ...")
        sys.exit(1)

    model_dir = sys.argv[1]
    bt2_paths = []
    for arg in sys.argv[2:]:
        expanded = sorted(glob.glob(arg))
        bt2_paths.extend(expanded if expanded else [arg])
    seen = set()
    bt2_paths = [p for p in bt2_paths if not (os.path.realpath(p) in seen or seen.add(os.path.realpath(p)))]

    # Load model
    meta_path = os.path.join(model_dir, "meta.json")
    onnx_path = os.path.join(model_dir, "model.onnx")
    with open(meta_path) as f:
        meta = json.load(f)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)

    y_scale = meta["y_scale"]
    max_pot = meta["max_pot"]
    max_hands = meta["max_hands"]
    num_oop = meta["num_oop"]
    num_ip = meta["num_ip"]
    in_dim = meta["in_dim"]
    num_hands = [num_oop, num_ip]

    print(f"Model: {model_dir}")
    print(f"  in_dim={in_dim}, max_hands={max_hands}, max_pot={max_pot}, y_scale={y_scale:.6f}")
    print(f"  num_oop={num_oop}, num_ip={num_ip}")
    print()

    # Load all bt2 files
    print(f"Loading {len(bt2_paths)} bt2 file(s)...")
    all_data = []
    for path in bt2_paths:
        d = load_bt2(path)
        all_data.append((os.path.basename(path), d))
        print(f"  {os.path.basename(path)}: {d['num_boundaries']} boundaries, {len(d['records'])} iterations")

    out_dir = os.path.join(model_dir, "verify")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Per-file analysis ----
    for filename, d in all_data:
        label = filename.replace(".bt2", "")
        records = d["records"]
        num_boundaries = d["num_boundaries"]
        boundary_pots = d["boundary_pots"]
        boundary_stacks = d["boundary_stacks"]
        n_iters = len(records)

        print(f"\n{'='*60}")
        print(f"  File: {filename}  ({num_boundaries} boundaries, {n_iters} iterations)")
        print(f"{'='*60}")

        # Build unique (pot, stack) labels for this file
        ps_labels = []
        for b in range(num_boundaries):
            ps_labels.append(f"({int(boundary_pots[b])},{int(boundary_stacks[b])})")

        # Collect errors: [iteration][boundary][player] = mean_abs_error
        iter_errors = np.zeros((n_iters, num_boundaries, 2))
        # Also collect per-hand errors for scatter plot
        all_true = []
        all_pred = []

        for t, boundaries in enumerate(records):
            for b in range(num_boundaries):
                pot_norm = boundary_pots[b] / max_pot
                stack_norm = boundary_stacks[b] / max_pot

                for player in range(2):
                    cfv_true, cfreach = boundaries[b][player]
                    n_p = num_hands[player]

                    # Build model input
                    inp = np.zeros((1, in_dim), dtype=np.float32)
                    inp[0, 0] = pot_norm
                    inp[0, 1] = stack_norm
                    inp[0, 2] = float(player)
                    inp[0, 3:3 + len(cfreach)] = cfreach

                    # Predict
                    pred = sess.run(None, {"input": inp})[0][0] * y_scale
                    cfv_pred = pred[:n_p]

                    err = np.abs(cfv_pred - cfv_true)
                    iter_errors[t, b, player] = err.mean()

                    all_true.append(cfv_true)
                    all_pred.append(cfv_pred)

        # ---- Summary stats ----
        mean_err = iter_errors.mean()
        max_err = iter_errors.max()
        p99_err = np.percentile(iter_errors, 99)
        print(f"  Mean abs error: {mean_err:.6f} chips")
        print(f"  Max abs error:  {max_err:.6f} chips")
        print(f"  P99 abs error:  {p99_err:.6f} chips")

        for player in range(2):
            plabel = "OOP" if player == 0 else "IP"
            pe = iter_errors[:, :, player]
            print(f"  {plabel}: mean={pe.mean():.6f}  max={pe.max():.6f}  p99={np.percentile(pe, 99):.6f}")

        # ---- Plot 1: Error over iterations (per boundary, averaged over players) ----
        fig, ax = plt.subplots(figsize=(12, 5))
        err_per_iter = iter_errors.mean(axis=(1, 2))  # avg over boundaries and players
        ax.plot(range(n_iters), err_per_iter, 'b-', alpha=0.7, label="Avg all boundaries")

        # Also plot per unique (pot,stack)
        unique_ps = sorted(set(zip([int(p) for p in boundary_pots], [int(s) for s in boundary_stacks])))
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_ps), 10)))
        for idx, (pot, stack) in enumerate(unique_ps):
            b_indices = [b for b in range(num_boundaries) if int(boundary_pots[b]) == pot and int(boundary_stacks[b]) == stack]
            err_this = iter_errors[:, b_indices, :].mean(axis=(1, 2))
            ax.plot(range(n_iters), err_this, '-', alpha=0.4, color=colors[idx % len(colors)],
                    label=f"({pot},{stack}) x{len(b_indices)}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Abs Error (chips)")
        ax.set_title(f"{label}: Model Error Over Iterations")
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path1 = os.path.join(out_dir, f"{label}_error_by_iter.png")
        plt.savefig(path1, dpi=150)
        plt.close()
        print(f"  Plot: {path1}")

        # ---- Plot 2: Predicted vs True CFV scatter (sampled) ----
        all_true_cat = np.concatenate(all_true)
        all_pred_cat = np.concatenate(all_pred)

        # Sample if too many points
        n_total = len(all_true_cat)
        max_points = 50000
        if n_total > max_points:
            idx = np.random.choice(n_total, max_points, replace=False)
            plot_true = all_true_cat[idx]
            plot_pred = all_pred_cat[idx]
        else:
            plot_true = all_true_cat
            plot_pred = all_pred_cat

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(plot_true, plot_pred, s=0.5, alpha=0.3)
        vmin = min(plot_true.min(), plot_pred.min())
        vmax = max(plot_true.max(), plot_pred.max())
        ax1.plot([vmin, vmax], [vmin, vmax], 'r-', linewidth=1, label="y=x")
        ax1.set_xlabel("True CFV (chips)")
        ax1.set_ylabel("Predicted CFV (chips)")
        ax1.set_title(f"{label}: Predicted vs True")
        ax1.legend()
        ax1.grid(alpha=0.3)

        residuals = plot_pred - plot_true
        ax2.hist(residuals, bins=100, alpha=0.7, edgecolor='none')
        ax2.set_xlabel("Residual (pred - true) chips")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{label}: Residual Distribution (mean={residuals.mean():.6f}, std={residuals.std():.6f})")
        ax2.axvline(0, color='r', linewidth=1)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(out_dir, f"{label}_scatter.png")
        plt.savefig(path2, dpi=150)
        plt.close()
        print(f"  Plot: {path2}")

        # ---- Plot 3: Error heatmap (boundary x iteration) ----
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

        for player, (ax, plabel) in enumerate(zip([ax1, ax2], ["OOP", "IP"])):
            data = iter_errors[:, :, player].T  # shape: (boundaries, iterations)
            im = ax.imshow(data, aspect='auto', cmap='hot', interpolation='nearest')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Boundary")
            ax.set_title(f"{label} {plabel}: Error by Boundary x Iteration")
            ax.set_yticks(range(num_boundaries))
            ax.set_yticklabels([f"B{b} {ps_labels[b]}" for b in range(num_boundaries)], fontsize=6)
            plt.colorbar(im, ax=ax, label="Mean Abs Error (chips)")

        plt.tight_layout()
        path3 = os.path.join(out_dir, f"{label}_heatmap.png")
        plt.savefig(path3, dpi=150)
        plt.close()
        print(f"  Plot: {path3}")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
