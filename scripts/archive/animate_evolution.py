"""Animate mean cfreach and mean CFV evolution across iterations.

Usage:
    python scripts/animate_evolution.py data/bt1/KcQh7s.bt1           # all boundaries
    python scripts/animate_evolution.py data/bt1/KcQh7s.bt1 2         # single boundary
"""

import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def load_bt1_all_boundaries(path):
    """Load ALL boundaries from a .bt1 file."""
    with open(path, "rb") as f:
        f.read(8)  # magic
        f.read(4)  # version
        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_boundaries = struct.unpack("<I", f.read(4))[0]
        num_iterations = struct.unpack("<I", f.read(4))[0]
        starting_pot = struct.unpack("<f", f.read(4))[0]

        # [boundary][iteration][hands]
        cfreach_ip = np.zeros((num_boundaries, num_iterations, num_ip), dtype=np.float32)
        cfreach_oop = np.zeros((num_boundaries, num_iterations, num_oop), dtype=np.float32)
        cfv_oop = np.zeros((num_boundaries, num_iterations, num_oop), dtype=np.float32)
        cfv_ip = np.zeros((num_boundaries, num_iterations, num_ip), dtype=np.float32)
        exploitabilities = np.zeros(num_iterations, dtype=np.float32)

        for t in range(num_iterations):
            f.read(4)  # iter
            exploitabilities[t] = struct.unpack("<f", f.read(4))[0]
            f.read(4)  # conv

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


def animate_single(data, boundary, output_path, board_name):
    """Animate a single boundary's mean cfreach and CFV."""
    n = data["num_iterations"]

    mean_cr_ip = data["cfreach_ip"][boundary].mean(axis=1)
    mean_cr_oop = data["cfreach_oop"][boundary].mean(axis=1)
    mean_cfv_oop = data["cfv_oop"][boundary].mean(axis=1)
    mean_cfv_ip = data["cfv_ip"][boundary].mean(axis=1)
    exploit_pct = data["exploitabilities"] / data["starting_pot"] * 100

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_cfreach, ax_cfv, ax_exploit = axes
    iters = np.arange(n)

    line_cr_ip, = ax_cfreach.plot([], [], "b-", linewidth=1.5, label=f"IP cfreach mean ({data['num_ip']} hands)")
    line_cr_oop, = ax_cfreach.plot([], [], "g-", linewidth=1.5, label=f"OOP cfreach mean ({data['num_oop']} hands)")
    dot_cr_ip, = ax_cfreach.plot([], [], "bo", markersize=6)
    dot_cr_oop, = ax_cfreach.plot([], [], "go", markersize=6)
    ax_cfreach.set_xlim(0, n - 1)
    ax_cfreach.set_ylim(0, max(mean_cr_ip.max(), mean_cr_oop.max()) * 1.1)
    ax_cfreach.set_ylabel("mean cfreach")
    ax_cfreach.set_title("Mean cfreach (opponent reach probability)")
    ax_cfreach.legend(loc="upper left", fontsize=9)
    ax_cfreach.grid(True, alpha=0.3)

    line_cfv_oop, = ax_cfv.plot([], [], "b-", linewidth=1.5, label="OOP mean CFV")
    line_cfv_ip, = ax_cfv.plot([], [], "g-", linewidth=1.5, label="IP mean CFV")
    dot_cfv_oop, = ax_cfv.plot([], [], "bo", markersize=6)
    dot_cfv_ip, = ax_cfv.plot([], [], "go", markersize=6)
    cfv_min = min(mean_cfv_oop.min(), mean_cfv_ip.min())
    cfv_max = max(mean_cfv_oop.max(), mean_cfv_ip.max())
    margin = (cfv_max - cfv_min) * 0.15
    ax_cfv.set_xlim(0, n - 1)
    ax_cfv.set_ylim(cfv_min - margin, cfv_max + margin)
    ax_cfv.set_ylabel("mean CFV")
    ax_cfv.set_title("Mean CFV (counterfactual value)")
    ax_cfv.axhline(y=0, color="gray", linewidth=0.5)
    ax_cfv.legend(loc="upper left", fontsize=9)
    ax_cfv.grid(True, alpha=0.3)

    line_exploit, = ax_exploit.plot([], [], "r-", linewidth=1.5, label="exploitability")
    dot_exploit, = ax_exploit.plot([], [], "ro", markersize=6)
    ax_exploit.set_xlim(0, n - 1)
    ax_exploit.set_ylim(0, exploit_pct.max() * 1.1)
    ax_exploit.set_xlabel("iteration")
    ax_exploit.set_ylabel("exploitability (% pot)")
    ax_exploit.set_title("Exploitability")
    ax_exploit.legend(loc="upper right", fontsize=9)
    ax_exploit.grid(True, alpha=0.3)

    fig.suptitle("", fontsize=13)
    plt.tight_layout()

    def update(frame):
        t = frame
        x = iters[:t + 1]
        fig.suptitle(
            f"{board_name} — Boundary {boundary} — Iteration {t}",
            fontsize=13, fontweight="bold",
        )
        line_cr_ip.set_data(x, mean_cr_ip[:t + 1])
        line_cr_oop.set_data(x, mean_cr_oop[:t + 1])
        dot_cr_ip.set_data([t], [mean_cr_ip[t]])
        dot_cr_oop.set_data([t], [mean_cr_oop[t]])
        line_cfv_oop.set_data(x, mean_cfv_oop[:t + 1])
        line_cfv_ip.set_data(x, mean_cfv_ip[:t + 1])
        dot_cfv_oop.set_data([t], [mean_cfv_oop[t]])
        dot_cfv_ip.set_data([t], [mean_cfv_ip[t]])
        line_exploit.set_data(x, exploit_pct[:t + 1])
        dot_exploit.set_data([t], [exploit_pct[t]])
        return ()

    anim = FuncAnimation(fig, update, frames=n, interval=80, blit=False)
    anim.save(output_path, writer="pillow", fps=12)
    print(f"Saved: {output_path}")
    plt.close()


def animate_all(data, output_path, board_name):
    """Animate all boundaries as thin lines + bold mean."""
    n = data["num_iterations"]
    nb = data["num_boundaries"]

    mean_cr_ip = data["cfreach_ip"].mean(axis=2)
    mean_cr_oop = data["cfreach_oop"].mean(axis=2)
    mean_cfv_oop = data["cfv_oop"].mean(axis=2)
    mean_cfv_ip = data["cfv_ip"].mean(axis=2)

    grand_cr_ip = mean_cr_ip.mean(axis=0)
    grand_cr_oop = mean_cr_oop.mean(axis=0)
    grand_cfv_oop = mean_cfv_oop.mean(axis=0)
    grand_cfv_ip = mean_cfv_ip.mean(axis=0)

    exploit_pct = data["exploitabilities"] / data["starting_pot"] * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax_cfreach, ax_cfv, ax_exploit = axes
    iters = np.arange(n)

    cmap_blue = plt.cm.Blues(np.linspace(0.3, 0.8, nb))
    cmap_green = plt.cm.Greens(np.linspace(0.3, 0.8, nb))

    boundary_cr_ip_lines = []
    boundary_cr_oop_lines = []
    for b in range(nb):
        l_ip, = ax_cfreach.plot([], [], color=cmap_blue[b], linewidth=0.5, alpha=0.3)
        l_oop, = ax_cfreach.plot([], [], color=cmap_green[b], linewidth=0.5, alpha=0.3)
        boundary_cr_ip_lines.append(l_ip)
        boundary_cr_oop_lines.append(l_oop)

    line_cr_ip, = ax_cfreach.plot([], [], "b-", linewidth=2.5, label=f"IP mean (all {nb} boundaries)")
    line_cr_oop, = ax_cfreach.plot([], [], "g-", linewidth=2.5, label=f"OOP mean (all {nb} boundaries)")
    dot_cr_ip, = ax_cfreach.plot([], [], "bo", markersize=7)
    dot_cr_oop, = ax_cfreach.plot([], [], "go", markersize=7)

    cr_max = max(mean_cr_ip.max(), mean_cr_oop.max()) * 1.1
    ax_cfreach.set_xlim(0, n - 1)
    ax_cfreach.set_ylim(0, cr_max)
    ax_cfreach.set_ylabel("mean cfreach")
    ax_cfreach.set_title(f"Mean cfreach — {nb} boundaries (thin) + grand mean (bold)")
    ax_cfreach.legend(loc="upper left", fontsize=9)
    ax_cfreach.grid(True, alpha=0.3)

    boundary_cfv_oop_lines = []
    boundary_cfv_ip_lines = []
    for b in range(nb):
        l_oop, = ax_cfv.plot([], [], color=cmap_blue[b], linewidth=0.5, alpha=0.3)
        l_ip, = ax_cfv.plot([], [], color=cmap_green[b], linewidth=0.5, alpha=0.3)
        boundary_cfv_oop_lines.append(l_oop)
        boundary_cfv_ip_lines.append(l_ip)

    line_cfv_oop, = ax_cfv.plot([], [], "b-", linewidth=2.5, label="OOP mean CFV")
    line_cfv_ip, = ax_cfv.plot([], [], "g-", linewidth=2.5, label="IP mean CFV")
    dot_cfv_oop, = ax_cfv.plot([], [], "bo", markersize=7)
    dot_cfv_ip, = ax_cfv.plot([], [], "go", markersize=7)

    cfv_min = min(mean_cfv_oop.min(), mean_cfv_ip.min())
    cfv_max = max(mean_cfv_oop.max(), mean_cfv_ip.max())
    margin = (cfv_max - cfv_min) * 0.15
    ax_cfv.set_xlim(0, n - 1)
    ax_cfv.set_ylim(cfv_min - margin, cfv_max + margin)
    ax_cfv.set_ylabel("mean CFV")
    ax_cfv.set_title(f"Mean CFV — {nb} boundaries (thin) + grand mean (bold)")
    ax_cfv.axhline(y=0, color="gray", linewidth=0.5)
    ax_cfv.legend(loc="upper left", fontsize=9)
    ax_cfv.grid(True, alpha=0.3)

    line_exploit, = ax_exploit.plot([], [], "r-", linewidth=2, label="exploitability")
    dot_exploit, = ax_exploit.plot([], [], "ro", markersize=7)
    ax_exploit.set_xlim(0, n - 1)
    ax_exploit.set_ylim(0, exploit_pct.max() * 1.1)
    ax_exploit.set_xlabel("iteration")
    ax_exploit.set_ylabel("exploitability (% pot)")
    ax_exploit.set_title("Exploitability")
    ax_exploit.legend(loc="upper right", fontsize=9)
    ax_exploit.grid(True, alpha=0.3)

    fig.suptitle("", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    def update(frame):
        t = frame
        x = iters[:t + 1]

        fig.suptitle(
            f"{board_name} — All {nb} boundaries — Iteration {t}",
            fontsize=14, fontweight="bold",
        )

        for b in range(nb):
            boundary_cr_ip_lines[b].set_data(x, mean_cr_ip[b, :t + 1])
            boundary_cr_oop_lines[b].set_data(x, mean_cr_oop[b, :t + 1])
            boundary_cfv_oop_lines[b].set_data(x, mean_cfv_oop[b, :t + 1])
            boundary_cfv_ip_lines[b].set_data(x, mean_cfv_ip[b, :t + 1])

        line_cr_ip.set_data(x, grand_cr_ip[:t + 1])
        line_cr_oop.set_data(x, grand_cr_oop[:t + 1])
        dot_cr_ip.set_data([t], [grand_cr_ip[t]])
        dot_cr_oop.set_data([t], [grand_cr_oop[t]])

        line_cfv_oop.set_data(x, grand_cfv_oop[:t + 1])
        line_cfv_ip.set_data(x, grand_cfv_ip[:t + 1])
        dot_cfv_oop.set_data([t], [grand_cfv_oop[t]])
        dot_cfv_ip.set_data([t], [grand_cfv_ip[t]])

        line_exploit.set_data(x, exploit_pct[:t + 1])
        dot_exploit.set_data([t], [exploit_pct[t]])

        return ()

    anim = FuncAnimation(fig, update, frames=n, interval=80, blit=False)
    anim.save(output_path, writer="pillow", fps=12)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/animate_evolution.py <file.bt1> [boundary_idx]")
        sys.exit(1)

    bt1_path = sys.argv[1]
    boundary = int(sys.argv[2]) if len(sys.argv) > 2 else None
    board = Path(bt1_path).stem

    data = load_bt1_all_boundaries(bt1_path)
    Path("data/out").mkdir(parents=True, exist_ok=True)

    print(f"Board: {board}")
    print(f"OOP: {data['num_oop']}, IP: {data['num_ip']}")
    print(f"Boundaries: {data['num_boundaries']}, Iters: {data['num_iterations']}")

    if boundary is not None:
        out_path = f"data/out/evolution_{board}_b{boundary}.gif"
        print(f"Generating animation for boundary {boundary}...")
        animate_single(data, boundary, out_path, board)
    else:
        out_path = f"data/out/evolution_{board}_all.gif"
        print("Generating animation for all boundaries...")
        animate_all(data, out_path, board)


if __name__ == "__main__":
    main()
