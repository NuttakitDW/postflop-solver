"""
Inspect oracle .oracle file — visualize matrix M structure.

Usage:
    python scripts/inspect_oracle.py data/oracles/A-oracle.oracle
"""

import struct
import sys
import numpy as np

def load_oracle(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"TORACLE\0", f"Bad magic: {magic}"

        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_amounts = struct.unpack("<I", f.read(4))[0]

        print(f"OOP hands: {num_oop}, IP hands: {num_ip}")
        print(f"Boundary amounts: {num_amounts}")
        print()

        matrices = {}
        for _ in range(num_amounts):
            amount = struct.unpack("<i", f.read(4))[0]

            # OOP matrix: [num_oop × num_ip]
            oop_size = num_oop * num_ip
            oop_data = np.frombuffer(f.read(oop_size * 4), dtype=np.float32)
            oop_matrix = oop_data.reshape(num_oop, num_ip)

            # IP matrix: [num_ip × num_oop]
            ip_size = num_ip * num_oop
            ip_data = np.frombuffer(f.read(ip_size * 4), dtype=np.float32)
            ip_matrix = ip_data.reshape(num_ip, num_oop)

            matrices[amount] = {"oop": oop_matrix, "ip": ip_matrix}

        return num_oop, num_ip, matrices


def analyze_matrix(name, M):
    """Print statistics about a matrix."""
    print(f"  Shape: {M.shape}")
    print(f"  Non-zero: {np.count_nonzero(M):,} / {M.size:,} ({np.count_nonzero(M)/M.size*100:.1f}%)")
    print(f"  Range: [{M.min():.6f}, {M.max():.6f}]")
    print(f"  Mean: {M.mean():.6f}, Std: {M.std():.6f}")
    print(f"  Abs mean: {np.abs(M).mean():.6f}")

    # Sparsity at various thresholds
    for thresh in [1e-6, 1e-4, 1e-2]:
        sparse = (np.abs(M) < thresh).sum()
        print(f"  |M[i,j]| < {thresh}: {sparse:,} ({sparse/M.size*100:.1f}%)")

    # Singular value decomposition — rank structure
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    total_energy = (S ** 2).sum()
    cum_energy = np.cumsum(S ** 2) / total_energy

    print(f"  Top singular values: {S[:10].tolist()}")
    for target in [0.90, 0.95, 0.99, 0.999]:
        rank = np.searchsorted(cum_energy, target) + 1
        print(f"  Rank for {target*100:.1f}% energy: {rank}")

    return S


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_oracle.py <oracle_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}")
    print()

    num_oop, num_ip, matrices = load_oracle(path)

    amounts = sorted(matrices.keys())
    print(f"Amounts: {amounts}")
    print()

    # Analyze each amount
    all_svs = {}
    for amount in amounts:
        print(f"=== Amount = {amount} ===")
        print(f"  (pot at turn boundary = {amount})")
        print()

        print(f"  OOP matrix (player=0, {num_oop}×{num_ip}):")
        oop_svs = analyze_matrix("OOP", matrices[amount]["oop"])
        print()

        print(f"  IP matrix (player=1, {num_ip}×{num_oop}):")
        ip_svs = analyze_matrix("IP", matrices[amount]["ip"])
        print()

        all_svs[amount] = {"oop": oop_svs, "ip": ip_svs}

    # Summary: rank structure across all amounts
    print("=" * 60)
    print("RANK SUMMARY (99% energy)")
    print("=" * 60)
    for amount in amounts:
        oop_S = all_svs[amount]["oop"]
        ip_S = all_svs[amount]["ip"]
        oop_cum = np.cumsum(oop_S ** 2) / (oop_S ** 2).sum()
        ip_cum = np.cumsum(ip_S ** 2) / (ip_S ** 2).sum()
        oop_rank99 = np.searchsorted(oop_cum, 0.99) + 1
        ip_rank99 = np.searchsorted(ip_cum, 0.99) + 1
        print(f"  amount={amount:>4}: OOP rank={oop_rank99:>3}, IP rank={ip_rank99:>3}  (of min({num_oop},{num_ip})={min(num_oop,num_ip)})")

    # Try to save heatmap if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Pick the most interesting amount (largest matrix values = deepest play)
        mid_amount = amounts[len(amounts) // 2]
        M = matrices[mid_amount]["oop"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Heatmap
        im = axes[0].imshow(M, aspect="auto", cmap="RdBu_r",
                           vmin=-np.percentile(np.abs(M), 99),
                           vmax=np.percentile(np.abs(M), 99))
        axes[0].set_title(f"OOP Matrix M (amount={mid_amount})")
        axes[0].set_xlabel("IP hand index")
        axes[0].set_ylabel("OOP hand index")
        plt.colorbar(im, ax=axes[0])

        # Singular value spectrum
        S = all_svs[mid_amount]["oop"]
        axes[1].semilogy(S, "b-")
        axes[1].set_title("Singular Value Spectrum")
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Singular Value")
        axes[1].grid(True, alpha=0.3)

        # Cumulative energy
        cum = np.cumsum(S ** 2) / (S ** 2).sum()
        axes[2].plot(cum, "r-")
        axes[2].axhline(0.99, color="gray", linestyle="--", label="99%")
        axes[2].axhline(0.999, color="gray", linestyle=":", label="99.9%")
        axes[2].set_title("Cumulative Energy (SVD)")
        axes[2].set_xlabel("Rank")
        axes[2].set_ylabel("Fraction of total energy")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = path.replace(".oracle", "_inspect.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nSaved visualization to: {out_path}")

    except ImportError:
        print("\n(matplotlib not available — skipping heatmap)")


if __name__ == "__main__":
    main()
