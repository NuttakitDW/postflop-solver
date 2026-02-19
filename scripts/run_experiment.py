#!/usr/bin/env python3
"""
Scaling experiment: Train 10 models with different dataset sizes, measure CFV accuracy
and exploitability when used in deepstack DCFR.

Usage:
    python scripts/run_experiment.py

Expects:
    - data/training_data/inputs.npy  [N, 2015]
    - data/training_data/targets.npy [N, 2000]
    - Rust binary: target/release/examples/backend_solver (with onnx feature)
    - config/template.json

Produces:
    - models/experiment/model_{size}.onnx    (10 models)
    - data/experiment/results.csv            (metrics CSV)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "training_data")
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "data", "experiment")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "experiment")
CONFIG_TEMPLATE = os.path.join(PROJECT_ROOT, "config", "template.json")
BACKEND_SOLVER = os.path.join(PROJECT_ROOT, "target", "release", "examples", "backend_solver")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train.py")

# Test boards for exploitability evaluation (diverse flops)
TEST_BOARDS = [
    "Td9d6h",  # template default - two-tone connected
    "As7h2c",  # dry Ace-high
]

# Dataset sizes to test (10 models) - adjusted for ~1000 total samples
DATASET_SIZES = [10, 25, 50, 100, 200, 350, 500, 650, 800, 900]


def get_available_sample_count():
    """Check how many samples are available in the training data."""
    inputs_path = os.path.join(TRAINING_DATA_DIR, "inputs.npy")
    if not os.path.exists(inputs_path):
        return 0
    inputs = np.load(inputs_path)
    return inputs.shape[0]


def adjust_dataset_sizes(total_available):
    """Adjust dataset sizes based on available data."""
    sizes = [s for s in DATASET_SIZES if s <= total_available]
    # Always include the max available if not already there
    if total_available not in sizes and total_available > 0:
        sizes.append(total_available)
    # Ensure we have ~10 sizes
    if len(sizes) < 3:
        # Very few samples - create linear splits
        step = max(1, total_available // 10)
        sizes = list(range(step, total_available + 1, step))
        if total_available not in sizes:
            sizes.append(total_available)
    return sorted(sizes)


def create_subset(size, subset_dir, train_indices=None):
    """Create a subset of training data.

    If train_indices is provided, selects from those indices only (ensuring
    test-set boards are never included). Otherwise uses first `size` samples.
    """
    os.makedirs(subset_dir, exist_ok=True)

    inputs = np.load(os.path.join(TRAINING_DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(TRAINING_DATA_DIR, "targets.npy"))

    if train_indices is not None:
        # Select from training partition only
        actual = min(size, len(train_indices))
        idx = train_indices[:actual]
    else:
        actual = min(size, len(inputs))
        idx = np.arange(actual)

    np.save(os.path.join(subset_dir, "inputs.npy"), inputs[idx])
    np.save(os.path.join(subset_dir, "targets.npy"), targets[idx])

    return len(idx)


def compute_epochs(size):
    """Adaptive epoch count: more epochs for small datasets, fewer for large."""
    if size <= 100:
        return 300
    elif size <= 500:
        return 200
    elif size <= 1000:
        return 150
    else:
        return 100


def train_model(size, subset_dir, model_path):
    """Train a model on a subset of data. Returns (train_loss, val_loss, train_time)."""
    epochs = compute_epochs(size)
    batch_size = min(512, max(16, size // 4))

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--input-dir", subset_dir,
        "--output", model_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", "0.001",
        "--hidden-dim", "500",
        "--num-layers", "7",
        "--huber-delta", "1.0",
    ]

    print(f"  Training: {size} samples, {epochs} epochs, batch={batch_size}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    train_time = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: Training failed!")
        print(result.stderr[-500:] if result.stderr else "No stderr")
        return None, None, train_time

    # Parse best val loss from output
    val_loss = None
    train_loss = None
    for line in result.stdout.split("\n"):
        if "Best val loss:" in line:
            match = re.search(r"Best val loss:\s*([\d.]+)", line)
            if match:
                val_loss = float(match.group(1))
        # Capture last epoch's losses
        if "train=" in line and "val=" in line:
            match_t = re.search(r"train=([\d.]+)", line)
            match_v = re.search(r"val=([\d.]+)", line)
            if match_t:
                train_loss = float(match_t.group(1))

    print(f"  Done in {train_time:.1f}s | best_val_loss={val_loss}")
    return train_loss, val_loss, train_time


def compute_cfv_accuracy(model_path, test_inputs, test_targets, test_range_oop, test_range_ip):
    """Compute reach-weighted MSE on a held-out test set using the ONNX model.

    Returns (mse_oop, mse_ip, mse_total) - pot-normalized MSE values.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  WARNING: onnxruntime not installed, skipping CFV accuracy")
        return None, None, None

    session = ort.InferenceSession(model_path)
    pred = session.run(None, {"input": test_inputs.astype(np.float32)})[0]

    K = 1000
    pred_oop = pred[:, :K]
    pred_ip = pred[:, K:]
    tgt_oop = test_targets[:, :K]
    tgt_ip = test_targets[:, K:]

    # Reach-weighted MSE (opponent's reach weights the error)
    # OOP CFV weighted by IP range, IP CFV weighted by OOP range
    err_oop = (pred_oop - tgt_oop) ** 2
    err_ip = (pred_ip - tgt_ip) ** 2

    # Weight by opponent reach
    w_oop = test_range_ip / (test_range_ip.sum(axis=1, keepdims=True) + 1e-8)
    w_ip = test_range_oop / (test_range_oop.sum(axis=1, keepdims=True) + 1e-8)

    mse_oop = (err_oop * w_oop).sum(axis=1).mean()
    mse_ip = (err_ip * w_ip).sum(axis=1).mean()
    mse_total = (mse_oop + mse_ip) / 2

    return float(mse_oop), float(mse_ip), float(mse_total)


def evaluate_exploitability(model_path, flop, config_template, locked_flop=True):
    """Run deepstack solver with the model, return exploitability percent."""
    # Create a temp config for this evaluation
    config = json.loads(json.dumps(config_template))
    config["board"]["flop"] = flop
    config["solver"]["maxIterations"] = 300
    config["solver"]["targetExploitabilityPercent"] = 0.5
    config["output"]["filename"] = os.path.join(
        EXPERIMENT_DIR, f"eval_{flop}.flop"
    )

    config_path = os.path.join(EXPERIMENT_DIR, f"eval_config_{flop}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [
        BACKEND_SOLVER,
        config_path,
        "--deepstack", model_path,
    ]
    if locked_flop:
        cmd += ["--locked-flop", "--flop-iters", "300", "--turnriver-iters", "300"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT on {flop}")
        return None, None

    # Parse JSON result even on non-zero exit code (file save may fail but solve succeeds)
    output = result.stdout
    json_match = re.search(r'\{[^}]*"exploitabilityPercent"[^}]*\}', output, re.DOTALL)
    if json_match:
        try:
            res = json.loads(json_match.group())
            exploit_pct = res.get("exploitabilityPercent")
            solve_time = res.get("solveTimeSeconds")
            return exploit_pct, solve_time
        except json.JSONDecodeError:
            pass

    # Fallback: try to find in full output
    match = re.search(r'"exploitabilityPercent"\s*:\s*([\d.]+)', output)
    if match:
        return float(match.group(1)), None

    print(f"    Could not parse result for {flop}")
    return None, None


def evaluate_exploitability_standard(flop, config_template):
    """Run standard (non-deepstack) solver for baseline exploitability."""
    config = json.loads(json.dumps(config_template))
    config["board"]["flop"] = flop
    config["solver"]["maxIterations"] = 300
    config["solver"]["targetExploitabilityPercent"] = 0.5
    config["output"]["filename"] = os.path.join(
        EXPERIMENT_DIR, f"eval_standard_{flop}.flop"
    )

    config_path = os.path.join(EXPERIMENT_DIR, f"eval_config_standard_{flop}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [BACKEND_SOLVER, config_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT on standard {flop}")
        return None, None

    # Parse JSON result even on non-zero exit code (file save may fail but solve succeeds)
    output = result.stdout
    match = re.search(r'"exploitabilityPercent"\s*:\s*([\d.]+)', output)
    if match:
        return float(match.group(1)), None

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Run scaling experiment")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only evaluate existing models")
    parser.add_argument("--skip-exploit", action="store_true",
                        help="Skip exploitability evaluation (faster)")
    args = parser.parse_args()

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Check available data
    total = get_available_sample_count()
    if total == 0:
        print("ERROR: No training data found. Run generate_raw_data + project_data first.")
        sys.exit(1)

    print(f"Available training samples: {total}")
    sizes = adjust_dataset_sizes(total)
    print(f"Dataset sizes to test: {sizes}")

    # Load config template for eval
    with open(CONFIG_TEMPLATE) as f:
        config_template = json.load(f)

    # Prepare held-out test set for CFV accuracy
    # Split by unique board to prevent data leakage
    inputs_all = np.load(os.path.join(TRAINING_DATA_DIR, "inputs.npy"))
    targets_all = np.load(os.path.join(TRAINING_DATA_DIR, "targets.npy"))

    # Import board-grouped split from train.py
    sys.path.insert(0, os.path.dirname(TRAIN_SCRIPT))
    from train import board_grouped_split

    # Use a different seed (999) to avoid correlation with train.py's internal split (seed=42)
    experiment_train_idx, experiment_test_idx = board_grouped_split(
        inputs_all, val_fraction=0.1, seed=999
    )

    test_inputs = inputs_all[experiment_test_idx]
    test_targets = targets_all[experiment_test_idx]
    test_size = len(experiment_test_idx)
    K = 1000
    BOARD_FEATURES = 15
    test_range_oop = test_inputs[:, BOARD_FEATURES:BOARD_FEATURES + K]
    test_range_ip = test_inputs[:, BOARD_FEATURES + K:]
    print(f"Test set: {test_size} samples (board-grouped, no leakage)")

    # Evaluate standard solver baseline (once)
    print("\n=== Standard Solver Baseline ===")
    baseline_exploits = {}
    if not args.skip_exploit:
        for flop in TEST_BOARDS:
            print(f"  Evaluating standard solver on {flop}...")
            exploit, stime = evaluate_exploitability_standard(flop, config_template)
            baseline_exploits[flop] = exploit
            print(f"    Exploitability: {exploit}% of pot" if exploit else "    Failed")

    # Results storage
    results = []

    for i, size in enumerate(sizes):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(sizes)}] Dataset size: {size}")
        print(f"{'='*60}")

        model_path = os.path.join(MODELS_DIR, f"model_{size}.onnx")
        subset_dir = os.path.join(EXPERIMENT_DIR, f"subset_{size}")

        # Cap at available training samples (test boards excluded)
        max_train = len(experiment_train_idx)
        actual_size = min(size, max_train)
        if actual_size != size:
            print(f"  Adjusted to {actual_size} (reserving {test_size} for test)")

        # Train
        if not args.skip_training:
            create_subset(actual_size, subset_dir, train_indices=experiment_train_idx)
            train_loss, val_loss, train_time = train_model(actual_size, subset_dir, model_path)
        else:
            train_loss, val_loss, train_time = None, None, 0

        if not os.path.exists(model_path):
            print(f"  SKIP: model not found at {model_path}")
            continue

        # CFV accuracy on held-out test set
        print(f"  Computing CFV accuracy on test set...")
        mse_oop, mse_ip, mse_total = compute_cfv_accuracy(
            model_path, test_inputs, test_targets, test_range_oop, test_range_ip
        )
        if mse_total is not None:
            print(f"  CFV MSE: oop={mse_oop:.6f} ip={mse_ip:.6f} total={mse_total:.6f}")

        # Exploitability evaluation
        exploit_results = {}
        avg_exploit = None
        avg_solve_time = None
        if not args.skip_exploit:
            print(f"  Evaluating exploitability...")
            exploits = []
            solve_times = []
            for flop in TEST_BOARDS:
                print(f"    Board: {flop}")
                exploit, stime = evaluate_exploitability(model_path, flop, config_template)
                exploit_results[flop] = exploit
                if exploit is not None:
                    exploits.append(exploit)
                if stime is not None:
                    solve_times.append(stime)
                print(f"      Exploit: {exploit}% of pot" if exploit else "      Failed")

            if exploits:
                avg_exploit = sum(exploits) / len(exploits)
                print(f"    Average exploitability: {avg_exploit:.4f}% of pot")
            if solve_times:
                avg_solve_time = sum(solve_times) / len(solve_times)

        row = {
            "dataset_size": actual_size,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_time_s": round(train_time, 1) if train_time else None,
            "test_mse_oop": round(mse_oop, 6) if mse_oop is not None else None,
            "test_mse_ip": round(mse_ip, 6) if mse_ip is not None else None,
            "test_mse_total": round(mse_total, 6) if mse_total is not None else None,
            "avg_exploit_pct": round(avg_exploit, 4) if avg_exploit is not None else None,
            "avg_solve_time_s": round(avg_solve_time, 1) if avg_solve_time is not None else None,
        }
        # Add per-board exploitability
        for flop in TEST_BOARDS:
            key = f"exploit_{flop}"
            row[key] = round(exploit_results.get(flop, None), 4) if exploit_results.get(flop) is not None else None

        # Add baseline info
        for flop in TEST_BOARDS:
            key = f"baseline_{flop}"
            row[key] = round(baseline_exploits.get(flop, None), 4) if baseline_exploits.get(flop) is not None else None

        results.append(row)

        # Write intermediate CSV after each model (for safety)
        write_csv(results)

    # Final report
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    write_csv(results)
    print(f"\nResults saved to: {os.path.join(EXPERIMENT_DIR, 'results.csv')}")

    # Print summary table
    print(f"\n{'Size':>8} | {'Val Loss':>10} | {'Test MSE':>10} | {'Avg Exploit%':>12} | {'Train Time':>10}")
    print("-" * 65)
    for r in results:
        size = r["dataset_size"]
        vl = f"{r['val_loss']:.6f}" if r['val_loss'] else "N/A"
        mse = f"{r['test_mse_total']:.6f}" if r['test_mse_total'] else "N/A"
        exp = f"{r['avg_exploit_pct']:.4f}" if r['avg_exploit_pct'] is not None else "N/A"
        tt = f"{r['train_time_s']:.0f}s" if r['train_time_s'] else "N/A"
        print(f"{size:>8} | {vl:>10} | {mse:>10} | {exp:>12} | {tt:>10}")


def write_csv(results):
    """Write results to CSV."""
    csv_path = os.path.join(EXPERIMENT_DIR, "results.csv")
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
