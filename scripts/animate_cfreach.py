"""Animate cfreach evolution across DCFR iterations from a .bt1 file.

Usage:
    python scripts/animate_cfreach.py data/bt1/KcQh7s.bt1 [boundary_idx]

Default boundary_idx = 0. Saves GIF to data/out/cfreach_anim_<board>_b<idx>.gif
"""

import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def load_bt1(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"BT1\x00\x00\x00\x00\x00", f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_boundaries = struct.unpack("<I", f.read(4))[0]
        num_iterations = struct.unpack("<I", f.read(4))[0]
        starting_pot = struct.unpack("<f", f.read(4))[0]

        per_boundary_floats = num_oop + num_ip + num_ip + num_oop
        per_boundary_bytes = per_boundary_floats * 4

        # Arrays: [iterations, hands]
        cfreach_ip_all = np.zeros((num_iterations, num_ip), dtype=np.float32)
        cfreach_oop_all = np.zeros((num_iterations, num_oop), dtype=np.float32)
        cfv_oop_all = np.zeros((num_iterations, num_oop), dtype=np.float32)
        cfv_ip_all = np.zeros((num_iterations, num_ip), dtype=np.float32)
        exploitabilities = np.zeros(num_iterations, dtype=np.float32)
        conv_modes = np.zeros(num_iterations, dtype=np.int32)

        target_boundary = int(sys.argv[2]) if len(sys.argv) > 2 else 0

        for t in range(num_iterations):
            iter_num = struct.unpack("<I", f.read(4))[0]
            exploitabilities[t] = struct.unpack("<f", f.read(4))[0]
            conv_modes[t] = struct.unpack("<I", f.read(4))[0]

            # Skip boundaries before target
            f.read(per_boundary_bytes * target_boundary)

            # Read target boundary
            cfv_oop_all[t] = np.frombuffer(f.read(num_oop * 4), dtype=np.float32)
            cfreach_ip_all[t] = np.frombuffer(f.read(num_ip * 4), dtype=np.float32)
            cfv_ip_all[t] = np.frombuffer(f.read(num_ip * 4), dtype=np.float32)
            cfreach_oop_all[t] = np.frombuffer(f.read(num_oop * 4), dtype=np.float32)

            # Skip remaining boundaries
            f.read(per_boundary_bytes * (num_boundaries - target_boundary - 1))

    return {
        "num_oop": num_oop,
        "num_ip": num_ip,
        "num_boundaries": num_boundaries,
        "num_iterations": num_iterations,
        "starting_pot": starting_pot,
        "boundary_idx": target_boundary,
        "cfreach_ip": cfreach_ip_all,
        "cfreach_oop": cfreach_oop_all,
        "cfv_oop": cfv_oop_all,
        "cfv_ip": cfv_ip_all,
        "exploitabilities": exploitabilities,
        "conv_modes": conv_modes,
    }


def animate(data, output_path):
    # Sort hands by final iteration's cfreach (descending)
    final_ip = data["cfreach_ip"][-1]
    final_oop = data["cfreach_oop"][-1]
    sort_ip = np.argsort(-final_ip)
    sort_oop = np.argsort(-final_oop)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [3, 1]})
    ax_ip, ax_ip_cfv = axes[0]
    ax_oop, ax_oop_cfv = axes[1]

    fig.suptitle("", fontsize=14)

    # IP cfreach
    (bar_ip,) = ax_ip.plot([], [], "b-", linewidth=0.5, alpha=0.8)
    ax_ip.set_xlim(0, data["num_ip"])
    ax_ip.set_ylim(-0.02, 1.05)
    ax_ip.set_ylabel("cfreach")
    ax_ip.set_title(f"IP cfreach (input for OOP model) — {data['num_ip']} hands")
    ax_ip.axhline(y=0, color="gray", linewidth=0.5)

    # IP cfv
    (bar_ip_cfv,) = ax_ip_cfv.plot([], [], "r-", linewidth=0.5, alpha=0.8)
    ax_ip_cfv.set_xlim(0, data["num_ip"])
    cfv_ip_min = data["cfv_ip"].min()
    cfv_ip_max = data["cfv_ip"].max()
    margin = (cfv_ip_max - cfv_ip_min) * 0.1
    ax_ip_cfv.set_ylim(cfv_ip_min - margin, cfv_ip_max + margin)
    ax_ip_cfv.set_ylabel("cfv")
    ax_ip_cfv.set_title("IP CFV (label)")
    ax_ip_cfv.axhline(y=0, color="gray", linewidth=0.5)

    # OOP cfreach
    (bar_oop,) = ax_oop.plot([], [], "g-", linewidth=0.5, alpha=0.8)
    ax_oop.set_xlim(0, data["num_oop"])
    ax_oop.set_ylim(-0.02, 1.05)
    ax_oop.set_xlabel("hand index (sorted by final cfreach)")
    ax_oop.set_ylabel("cfreach")
    ax_oop.set_title(f"OOP cfreach (input for IP model) — {data['num_oop']} hands")
    ax_oop.axhline(y=0, color="gray", linewidth=0.5)

    # OOP cfv
    (bar_oop_cfv,) = ax_oop_cfv.plot([], [], "r-", linewidth=0.5, alpha=0.8)
    ax_oop_cfv.set_xlim(0, data["num_oop"])
    cfv_oop_min = data["cfv_oop"].min()
    cfv_oop_max = data["cfv_oop"].max()
    margin = (cfv_oop_max - cfv_oop_min) * 0.1
    ax_oop_cfv.set_ylim(cfv_oop_min - margin, cfv_oop_max + margin)
    ax_oop_cfv.set_xlabel("hand index (sorted by final cfreach)")
    ax_oop_cfv.set_ylabel("cfv")
    ax_oop_cfv.set_title("OOP CFV (label)")
    ax_oop_cfv.axhline(y=0, color="gray", linewidth=0.5)

    plt.tight_layout()

    def update(frame):
        t = frame
        exploit_pct = data["exploitabilities"][t] / data["starting_pot"] * 100
        conv = "ON" if data["conv_modes"][t] else "OFF"
        fig.suptitle(
            f"Boundary {data['boundary_idx']} — Iteration {t}/{data['num_iterations']-1}"
            f"  |  Exploitability: {exploit_pct:.2f}%  |  Convergence: {conv}",
            fontsize=13,
        )

        ip_data = data["cfreach_ip"][t][sort_ip]
        oop_data = data["cfreach_oop"][t][sort_oop]
        ip_cfv = data["cfv_ip"][t][sort_ip]
        oop_cfv = data["cfv_oop"][t][sort_oop]

        bar_ip.set_data(np.arange(len(ip_data)), ip_data)
        bar_oop.set_data(np.arange(len(oop_data)), oop_data)
        bar_ip_cfv.set_data(np.arange(len(ip_cfv)), ip_cfv)
        bar_oop_cfv.set_data(np.arange(len(oop_cfv)), oop_cfv)

        return bar_ip, bar_oop, bar_ip_cfv, bar_oop_cfv

    anim = FuncAnimation(
        fig,
        update,
        frames=data["num_iterations"],
        interval=80,
        blit=False,
    )

    anim.save(output_path, writer="pillow", fps=12)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/animate_cfreach.py <file.bt1> [boundary_idx]")
        sys.exit(1)

    bt1_path = sys.argv[1]
    data = load_bt1(bt1_path)

    board = Path(bt1_path).stem
    bidx = data["boundary_idx"]
    out_dir = Path("data/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"cfreach_anim_{board}_b{bidx}.gif"

    print(f"Board: {board}")
    print(f"Boundary: {bidx}")
    print(f"Iterations: {data['num_iterations']}")
    print(f"OOP hands: {data['num_oop']}, IP hands: {data['num_ip']}")
    print(f"Generating animation...")

    animate(data, str(output_path))


if __name__ == "__main__":
    main()
