"""Animate mean CFV evolution for ALL boundaries in a single grid GIF.

Each boundary gets its own subplot (5x5 grid) + shared exploitability panel.
Labels show pot amount at each boundary (from _boundaries.json).

Usage:
    python scripts/animate_evolution_grid.py data/bt1/KcQh7s.bt1
"""

import json
import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def load_bt1_all_boundaries(path):
    with open(path, "rb") as f:
        f.read(8); f.read(4)
        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_boundaries = struct.unpack("<I", f.read(4))[0]
        num_iterations = struct.unpack("<I", f.read(4))[0]
        starting_pot = struct.unpack("<f", f.read(4))[0]

        cfreach_ip = np.zeros((num_boundaries, num_iterations, num_ip), dtype=np.float32)
        cfreach_oop = np.zeros((num_boundaries, num_iterations, num_oop), dtype=np.float32)
        cfv_oop = np.zeros((num_boundaries, num_iterations, num_oop), dtype=np.float32)
        cfv_ip = np.zeros((num_boundaries, num_iterations, num_ip), dtype=np.float32)
        exploitabilities = np.zeros(num_iterations, dtype=np.float32)

        for t in range(num_iterations):
            f.read(4)
            exploitabilities[t] = struct.unpack("<f", f.read(4))[0]
            f.read(4)
            for b in range(num_boundaries):
                cfv_oop[b, t] = np.frombuffer(f.read(num_oop * 4), dtype=np.float32)
                cfreach_ip[b, t] = np.frombuffer(f.read(num_ip * 4), dtype=np.float32)
                cfv_ip[b, t] = np.frombuffer(f.read(num_ip * 4), dtype=np.float32)
                cfreach_oop[b, t] = np.frombuffer(f.read(num_oop * 4), dtype=np.float32)

    return {
        "num_oop": num_oop, "num_ip": num_ip,
        "num_boundaries": num_boundaries,
        "num_iterations": num_iterations, "starting_pot": starting_pot,
        "cfreach_ip": cfreach_ip, "cfreach_oop": cfreach_oop,
        "cfv_oop": cfv_oop, "cfv_ip": cfv_ip,
        "exploitabilities": exploitabilities,
    }


def animate(data, boundary_meta, output_path, board_name):
    nb = data["num_boundaries"]
    n = data["num_iterations"]
    pot = data["starting_pot"]

    # Per-boundary mean across hands
    mean_cfv_oop = data["cfv_oop"].mean(axis=2)   # [nb, n]
    mean_cfv_ip = data["cfv_ip"].mean(axis=2)
    mean_cr_ip = data["cfreach_ip"].mean(axis=2)
    mean_cr_oop = data["cfreach_oop"].mean(axis=2)
    exploit_pct = data["exploitabilities"] / pot * 100
    iters = np.arange(n)

    cols = 5
    rows = (nb + cols - 1) // cols

    fig = plt.figure(figsize=(24, 4 * rows + 3))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1]*rows + [0.8],
                          hspace=0.35, wspace=0.3)

    # Create boundary subplots
    boundary_axes = []
    lines_cfv_oop = []
    lines_cfv_ip = []
    lines_cr_ip = []
    lines_cr_oop = []

    for b in range(nb):
        r, c = divmod(b, cols)
        ax = fig.add_subplot(gs[r, c])
        boundary_axes.append(ax)

        # CFV lines
        l_oop, = ax.plot([], [], "b-", linewidth=1, alpha=0.8)
        l_ip, = ax.plot([], [], "g-", linewidth=1, alpha=0.8)
        lines_cfv_oop.append(l_oop)
        lines_cfv_ip.append(l_ip)

        # cfreach on twin axis
        ax2 = ax.twinx()
        l_cr_ip, = ax2.plot([], [], "b--", linewidth=0.7, alpha=0.4)
        l_cr_oop, = ax2.plot([], [], "g--", linewidth=0.7, alpha=0.4)
        lines_cr_ip.append(l_cr_ip)
        lines_cr_oop.append(l_cr_oop)

        # Set limits
        cfv_min = min(mean_cfv_oop[b].min(), mean_cfv_ip[b].min())
        cfv_max = max(mean_cfv_oop[b].max(), mean_cfv_ip[b].max())
        margin = (cfv_max - cfv_min) * 0.15 if cfv_max > cfv_min else 0.01
        ax.set_xlim(0, n - 1)
        ax.set_ylim(cfv_min - margin, cfv_max + margin)
        ax.axhline(y=0, color="gray", linewidth=0.3)
        ax.grid(True, alpha=0.2)
        if b < len(boundary_meta):
            label = f"{boundary_meta[b]['path']} (pot {boundary_meta[b]['pot']})"
        else:
            label = f"B{b}"
        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)

        cr_max = max(mean_cr_ip[b].max(), mean_cr_oop[b].max()) * 1.1
        ax2.set_ylim(0, max(cr_max, 0.01))
        ax2.tick_params(labelsize=5, colors="gray")

        if b == 0:
            ax.set_ylabel("CFV", fontsize=7)

    # Hide unused cells
    for b in range(nb, rows * cols):
        r, c = divmod(b, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_visible(False)

    # Exploitability panel spanning full width
    ax_exploit = fig.add_subplot(gs[rows, :])
    line_exploit, = ax_exploit.plot([], [], "r-", linewidth=2, label="exploitability")
    dot_exploit, = ax_exploit.plot([], [], "ro", markersize=6)
    ax_exploit.set_xlim(0, n - 1)
    ax_exploit.set_ylim(0, exploit_pct.max() * 1.1)
    ax_exploit.set_xlabel("iteration")
    ax_exploit.set_ylabel("exploitability (% pot)")
    ax_exploit.legend(loc="upper right", fontsize=9)
    ax_exploit.grid(True, alpha=0.3)

    # Legend in first cell
    boundary_axes[0].plot([], [], "b-", linewidth=1, label="OOP CFV")
    boundary_axes[0].plot([], [], "g-", linewidth=1, label="IP CFV")
    boundary_axes[0].plot([], [], "b--", linewidth=0.7, alpha=0.4, label="IP reach")
    boundary_axes[0].plot([], [], "g--", linewidth=0.7, alpha=0.4, label="OOP reach")
    boundary_axes[0].legend(fontsize=5, loc="upper left")

    fig.suptitle("", fontsize=15)

    def update(frame):
        t = frame
        x = iters[:t + 1]
        ep = exploit_pct[t]

        fig.suptitle(
            f"{board_name} — Iteration {t}/{n-1} — Exploitability: {ep:.2f}%",
            fontsize=15, fontweight="bold",
        )

        for b in range(nb):
            lines_cfv_oop[b].set_data(x, mean_cfv_oop[b, :t + 1])
            lines_cfv_ip[b].set_data(x, mean_cfv_ip[b, :t + 1])
            lines_cr_ip[b].set_data(x, mean_cr_ip[b, :t + 1])
            lines_cr_oop[b].set_data(x, mean_cr_oop[b, :t + 1])

        line_exploit.set_data(x, exploit_pct[:t + 1])
        dot_exploit.set_data([t], [exploit_pct[t]])

        return ()

    anim = FuncAnimation(fig, update, frames=n, interval=80, blit=False)
    anim.save(output_path, writer="pillow", fps=12)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/animate_evolution_grid.py <file.bt1>")
        sys.exit(1)

    bt1_path = sys.argv[1]
    board = Path(bt1_path).stem
    data = load_bt1_all_boundaries(bt1_path)

    # Load boundary metadata
    boundaries_path = Path(bt1_path).parent / f"{board}_boundaries.json"
    if boundaries_path.exists():
        with open(boundaries_path) as f:
            boundary_meta = json.load(f)["boundaries"]
    else:
        print(f"Warning: {boundaries_path} not found, using B0..BN labels")
        boundary_meta = []

    out_path = f"data/out/evolution_grid_{board}.gif"
    Path("data/out").mkdir(parents=True, exist_ok=True)

    print(f"Board: {board}")
    print(f"OOP: {data['num_oop']}, IP: {data['num_ip']}")
    print(f"Boundaries: {data['num_boundaries']}, Iters: {data['num_iterations']}")
    print("Generating grid animation...")

    animate(data, boundary_meta, out_path, board)


if __name__ == "__main__":
    main()
