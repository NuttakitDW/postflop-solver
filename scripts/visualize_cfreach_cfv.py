"""Visualize the relationship between cfreach (input) and CFV (output).

Shows what the NN needs to learn: given opponent's reach → predict player's values.

Usage:
    python scripts/visualize_cfreach_cfv.py data/bt1/KcQh7s.bt1 [boundary_idx]
"""

import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_bt1_boundary(path, target_boundary=0):
    with open(path, "rb") as f:
        f.read(8)  # magic
        f.read(4)  # version
        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_boundaries = struct.unpack("<I", f.read(4))[0]
        num_iterations = struct.unpack("<I", f.read(4))[0]
        starting_pot = struct.unpack("<f", f.read(4))[0]

        per_boundary_bytes = (num_oop + num_ip + num_ip + num_oop) * 4

        cfreach_ip_all = np.zeros((num_iterations, num_ip), dtype=np.float32)
        cfreach_oop_all = np.zeros((num_iterations, num_oop), dtype=np.float32)
        cfv_oop_all = np.zeros((num_iterations, num_oop), dtype=np.float32)
        cfv_ip_all = np.zeros((num_iterations, num_ip), dtype=np.float32)
        exploitabilities = np.zeros(num_iterations, dtype=np.float32)

        for t in range(num_iterations):
            f.read(4)  # iter
            exploitabilities[t] = struct.unpack("<f", f.read(4))[0]
            f.read(4)  # conv

            f.read(per_boundary_bytes * target_boundary)

            cfv_oop_all[t] = np.frombuffer(f.read(num_oop * 4), dtype=np.float32)
            cfreach_ip_all[t] = np.frombuffer(f.read(num_ip * 4), dtype=np.float32)
            cfv_ip_all[t] = np.frombuffer(f.read(num_ip * 4), dtype=np.float32)
            cfreach_oop_all[t] = np.frombuffer(f.read(num_oop * 4), dtype=np.float32)

            f.read(per_boundary_bytes * (num_boundaries - target_boundary - 1))

    return {
        "num_oop": num_oop, "num_ip": num_ip,
        "num_iterations": num_iterations, "starting_pot": starting_pot,
        "boundary_idx": target_boundary,
        "cfreach_ip": cfreach_ip_all, "cfreach_oop": cfreach_oop_all,
        "cfv_oop": cfv_oop_all, "cfv_ip": cfv_ip_all,
        "exploitabilities": exploitabilities,
    }


def plot_relationship(data, output_path, board_name):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"{board_name} — Boundary {data['boundary_idx']} — cfreach → CFV relationship",
                 fontsize=14, fontweight="bold")

    iters = np.arange(data["num_iterations"])
    pot = data["starting_pot"]

    # =========================================================================
    # Row 1: Heatmaps showing evolution (iterations x hands)
    # =========================================================================

    # Sort IP hands by final cfreach (descending)
    sort_ip = np.argsort(-data["cfreach_ip"][-1])
    # Sort OOP hands by final cfv (descending)
    sort_oop = np.argsort(-data["cfv_oop"][-1])

    ax1 = fig.add_subplot(3, 2, 1)
    im1 = ax1.imshow(data["cfreach_ip"][:, sort_ip], aspect="auto", cmap="Blues",
                     interpolation="nearest")
    ax1.set_title("IP cfreach across iterations (NN input for OOP)")
    ax1.set_ylabel("iteration")
    ax1.set_xlabel(f"IP hand (sorted by final reach)")
    plt.colorbar(im1, ax=ax1, label="reach prob")

    ax2 = fig.add_subplot(3, 2, 2)
    vmax = max(abs(data["cfv_oop"].min()), abs(data["cfv_oop"].max()))
    im2 = ax2.imshow(data["cfv_oop"][:, sort_oop], aspect="auto", cmap="RdBu_r",
                     interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax2.set_title("OOP CFV across iterations (NN output)")
    ax2.set_ylabel("iteration")
    ax2.set_xlabel(f"OOP hand (sorted by final CFV)")
    plt.colorbar(im2, ax=ax2, label="cfv")

    # =========================================================================
    # Row 2: Per-hand trajectories — pick interesting hands
    # =========================================================================

    # Pick 8 OOP hands spread across CFV range
    final_cfv = data["cfv_oop"][-1]
    percentiles = [5, 15, 30, 45, 55, 70, 85, 95]
    hand_indices = []
    for p in percentiles:
        target = np.percentile(final_cfv, p)
        idx = np.argmin(np.abs(final_cfv - target))
        hand_indices.append(idx)

    ax3 = fig.add_subplot(3, 2, 3)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(hand_indices)))
    for i, h in enumerate(hand_indices):
        ax3.plot(iters, data["cfv_oop"][:, h], color=colors[i],
                 label=f"hand {h} (final={final_cfv[h]:.4f})", alpha=0.8)
    ax3.set_title("OOP hand CFV trajectories (what NN must predict)")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("cfv")
    ax3.legend(fontsize=7, ncol=2)
    ax3.axhline(y=0, color="gray", linewidth=0.5)

    # cfreach_ip summary over iterations
    ax4 = fig.add_subplot(3, 2, 4)
    cfreach_sum = data["cfreach_ip"].sum(axis=1)
    cfreach_nonzero = (data["cfreach_ip"] > 0.01).sum(axis=1)
    ax4_twin = ax4.twinx()
    l1 = ax4.plot(iters, cfreach_sum, "b-", label="cfreach_ip sum", alpha=0.8)
    l2 = ax4_twin.plot(iters, cfreach_nonzero, "g--", label="active IP hands", alpha=0.8)
    ax4.set_title("IP cfreach summary (NN input changes)")
    ax4.set_xlabel("iteration")
    ax4.set_ylabel("cfreach sum", color="b")
    ax4_twin.set_ylabel("active hands", color="g")
    lines = l1 + l2
    ax4.legend(lines, [l.get_label() for l in lines], fontsize=8)

    # =========================================================================
    # Row 3: Direct scatter — does cfreach predict CFV?
    # =========================================================================

    # For each iteration, scatter: cfreach_ip_sum vs each OOP hand's CFV
    # Use last 50 iterations (converged) vs first 10 (random)
    ax5 = fig.add_subplot(3, 2, 5)
    # Show 3 snapshots: early, mid, late
    snapshots = [0, data["num_iterations"] // 2, data["num_iterations"] - 1]
    snapshot_colors = ["red", "orange", "blue"]
    snapshot_labels = ["iter 0 (random)", f"iter {snapshots[1]} (mid)", f"iter {snapshots[2]} (converged)"]

    for snap_t, c, label in zip(snapshots, snapshot_colors, snapshot_labels):
        # For each OOP hand, what determines its CFV?
        # Plot: hand's final equity rank vs CFV at this iteration
        cfv_at_t = data["cfv_oop"][snap_t]
        ax5.scatter(np.arange(data["num_oop"]), cfv_at_t[sort_oop],
                    s=1, alpha=0.4, color=c, label=label)

    ax5.set_title("OOP CFV distribution at different iterations")
    ax5.set_xlabel("OOP hand rank (by final CFV)")
    ax5.set_ylabel("cfv")
    ax5.legend(fontsize=8, markerscale=5)
    ax5.axhline(y=0, color="gray", linewidth=0.5)

    # Scatter: how correlated is cfreach change with CFV change?
    ax6 = fig.add_subplot(3, 2, 6)
    # For each iteration pair (t, t+1), compute delta_cfreach_norm and delta_cfv_norm
    deltas_cfreach = []
    deltas_cfv = []
    for t in range(data["num_iterations"] - 1):
        dc = np.linalg.norm(data["cfreach_ip"][t+1] - data["cfreach_ip"][t])
        dv = np.linalg.norm(data["cfv_oop"][t+1] - data["cfv_oop"][t])
        deltas_cfreach.append(dc)
        deltas_cfv.append(dv)

    ax6.scatter(deltas_cfreach, deltas_cfv, s=8, alpha=0.6, c=iters[:-1], cmap="viridis")
    ax6.set_title("Change correlation: |Δcfreach| vs |Δcfv|")
    ax6.set_xlabel("|Δcfreach_ip| (L2 norm)")
    ax6.set_ylabel("|Δcfv_oop| (L2 norm)")
    cb = plt.colorbar(ax6.collections[0], ax=ax6, label="iteration")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_cfreach_cfv.py <file.bt1> [boundary_idx]")
        sys.exit(1)

    bt1_path = sys.argv[1]
    boundary = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    board = Path(bt1_path).stem
    data = load_bt1_boundary(bt1_path, boundary)

    out_path = f"data/out/cfreach_cfv_{board}_b{boundary}.png"
    Path("data/out").mkdir(parents=True, exist_ok=True)

    print(f"Board: {board}, Boundary: {boundary}")
    print(f"OOP: {data['num_oop']} hands, IP: {data['num_ip']} hands, Iters: {data['num_iterations']}")

    plot_relationship(data, out_path, board)


if __name__ == "__main__":
    main()
