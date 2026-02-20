#!/usr/bin/env python3
"""Plot benchmark results from benchmark_models CSV output."""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "benchmark_onnx", "benchmark_results.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "benchmark_onnx")


def main():
    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Separate standard baseline vs deepstack models
    baseline = None
    models = []
    for r in rows:
        dp = int(r["data_points"])
        if dp == 0:
            baseline = float(r["exploitability_pct"])
        else:
            models.append({
                "data_points": dp,
                "exploit_pct": float(r["exploitability_pct"]),
                "time": float(r["solve_time_secs"]),
            })

    models.sort(key=lambda x: x["data_points"])
    dps = [m["data_points"] for m in models]
    exploits = [m["exploit_pct"] for m in models]
    times = [m["time"] for m in models]
    labels = [f"{d // 1000}k" for d in dps]

    # --- Plot 1: Exploitability vs Data Points ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dps, exploits, "o-", color="#4c72b0", linewidth=2, markersize=10, label="Deepstack (locked-flop)")
    if baseline is not None:
        ax.axhline(y=baseline, color="#55a868", linewidth=2, linestyle="--", label=f"Standard solver ({baseline:.2f}%)")
    ax.set_xlabel("Training Data Points", fontsize=12)
    ax.set_ylabel("Exploitability (% of pot)", fontsize=12)
    ax.set_title("Exploitability vs Training Data Size", fontsize=14)
    ax.set_xticks(dps)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for d, e, label in zip(dps, exploits, labels):
        ax.annotate(f"{e:.2f}%", (d, e), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=10)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "exploitability_vs_data.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved: {path1}")

    # --- Plot 2: Combined (exploitability + solve time) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: exploitability
    ax1.bar(labels, exploits, color="#4c72b0", alpha=0.85)
    if baseline is not None:
        ax1.axhline(y=baseline, color="#55a868", linewidth=2, linestyle="--", label=f"Standard ({baseline:.2f}%)")
        ax1.legend(fontsize=10)
    ax1.set_xlabel("Training Data Size")
    ax1.set_ylabel("Exploitability (% of pot)")
    ax1.set_title("Exploitability by Model")
    ax1.grid(True, alpha=0.3, axis="y")
    for i, e in enumerate(exploits):
        ax1.text(i, e + 0.3, f"{e:.2f}%", ha="center", fontsize=10)

    # Right: solve time
    ax2.bar(labels, times, color="#dd8452", alpha=0.85)
    ax2.set_xlabel("Training Data Size")
    ax2.set_ylabel("Solve Time (seconds)")
    ax2.set_title("Solve Time by Model")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, t in enumerate(times):
        ax2.text(i, t + 0.2, f"{t:.1f}s", ha="center", fontsize=10)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "benchmark_summary.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
