"""Plot per-boundary CFV evolution with stabilization markers.

Shows each boundary in its own subplot (5x5 grid for 25 boundaries).
Marks the iteration where CFV stabilizes for each boundary.

Usage:
    python scripts/plot_cfv_stability.py data/bt1/KcQh7s.bt1
"""

import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_bt1_all_boundaries(path):
    with open(path, "rb") as f:
        f.read(8)
        f.read(4)
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


def find_stable_iter(mean_cfv, window=10, threshold_ratio=0.01):
    """Find the first iteration where CFV is stable.

    Stable = the max absolute change over a rolling window is less than
    threshold_ratio * the total range of the series.
    """
    n = len(mean_cfv)
    total_range = mean_cfv.max() - mean_cfv.min()
    if total_range < 1e-10:
        return 0  # already flat

    threshold = total_range * threshold_ratio

    for t in range(window, n):
        segment = mean_cfv[t - window:t + 1]
        if segment.max() - segment.min() < threshold:
            return t - window
    return n - 1  # never stabilized


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_cfv_stability.py <file.bt1>")
        sys.exit(1)

    bt1_path = sys.argv[1]
    board = Path(bt1_path).stem
    data = load_bt1_all_boundaries(bt1_path)

    nb = data["num_boundaries"]
    n = data["num_iterations"]
    pot = data["starting_pot"]
    exploit_pct = data["exploitabilities"] / pot * 100

    # Per-boundary mean CFV across hands: [boundary, iteration]
    mean_cfv_oop = data["cfv_oop"].mean(axis=2)
    mean_cfv_ip = data["cfv_ip"].mean(axis=2)

    # Grid layout
    cols = 5
    rows = (nb + cols - 1) // cols
    # Extra row for exploitability
    fig, all_axes = plt.subplots(rows + 1, cols, figsize=(22, 4 * (rows + 1)))

    iters = np.arange(n)

    stable_iters_oop = []
    stable_iters_ip = []

    for b in range(nb):
        r, c = divmod(b, cols)
        ax = all_axes[r, c]

        oop_mean = mean_cfv_oop[b]
        ip_mean = mean_cfv_ip[b]

        # Find stabilization
        s_oop = find_stable_iter(oop_mean)
        s_ip = find_stable_iter(ip_mean)
        stable_iters_oop.append(s_oop)
        stable_iters_ip.append(s_ip)

        # Exploitability at stabilization point
        s_combined = max(s_oop, s_ip)
        exploit_at_stable = exploit_pct[s_combined] if s_combined < n else exploit_pct[-1]

        ax.plot(iters, oop_mean, "b-", linewidth=1, alpha=0.8, label="OOP")
        ax.plot(iters, ip_mean, "g-", linewidth=1, alpha=0.8, label="IP")

        # Mark stabilization
        ax.axvline(x=s_oop, color="blue", linewidth=1.5, linestyle="--", alpha=0.6)
        ax.axvline(x=s_ip, color="green", linewidth=1.5, linestyle="--", alpha=0.6)

        ax.set_title(f"B{b}  stable@{s_combined} ({exploit_at_stable:.1f}%)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.axhline(y=0, color="gray", linewidth=0.3)
        ax.grid(True, alpha=0.2)

        if b == 0:
            ax.legend(fontsize=7)

    # Hide unused boundary subplots
    for b in range(nb, rows * cols):
        r, c = divmod(b, cols)
        all_axes[r, c].set_visible(False)

    # Bottom row: exploitability with all stability markers
    ax_exploit = fig.add_subplot(rows + 1, 1, rows + 1)
    # Hide the individual axes in the last row
    for c in range(cols):
        all_axes[rows, c].set_visible(False)

    ax_exploit.plot(iters, exploit_pct, "r-", linewidth=2, label="exploitability")
    ax_exploit.set_xlabel("iteration")
    ax_exploit.set_ylabel("exploitability (% pot)")
    ax_exploit.set_title("Exploitability with per-boundary stabilization points")
    ax_exploit.grid(True, alpha=0.3)

    # Mark each boundary's stabilization point on exploitability curve
    all_stable = [max(s_oop, s_ip) for s_oop, s_ip in zip(stable_iters_oop, stable_iters_ip)]
    for b, s in enumerate(all_stable):
        ax_exploit.axvline(x=s, color="gray", linewidth=0.5, alpha=0.4)
        ax_exploit.plot(s, exploit_pct[s], "k.", markersize=4)

    # Mark the latest stabilization point (when ALL boundaries are stable)
    latest = max(all_stable)
    ax_exploit.axvline(x=latest, color="black", linewidth=2, linestyle="--",
                       label=f"all stable @ iter {latest} ({exploit_pct[latest]:.2f}%)")
    ax_exploit.plot(latest, exploit_pct[latest], "ko", markersize=8)

    # Mark the earliest stabilization point
    earliest = min(all_stable)
    ax_exploit.axvline(x=earliest, color="green", linewidth=1.5, linestyle="--",
                       label=f"first stable @ iter {earliest} ({exploit_pct[earliest]:.2f}%)")

    ax_exploit.legend(fontsize=9)
    ax_exploit.set_xlim(0, n - 1)

    fig.suptitle(f"{board} — Per-boundary CFV stabilization ({nb} boundaries)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = f"data/out/cfv_stability_{board}.png"
    Path("data/out").mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # Print summary
    print(f"\nBoard: {board}, Boundaries: {nb}, Iterations: {n}")
    print(f"{'Boundary':>10} {'OOP stable':>12} {'IP stable':>12} {'Combined':>12} {'Exploit%':>10}")
    print("-" * 60)
    for b in range(nb):
        s = all_stable[b]
        print(f"{'B' + str(b):>10} {stable_iters_oop[b]:>12} {stable_iters_ip[b]:>12} {s:>12} {exploit_pct[s]:>10.2f}")
    print("-" * 60)
    print(f"{'Earliest':>10} {'':<12} {'':<12} {earliest:>12} {exploit_pct[earliest]:>10.2f}")
    print(f"{'Latest':>10} {'':<12} {'':<12} {latest:>12} {exploit_pct[latest]:>10.2f}")


if __name__ == "__main__":
    main()
