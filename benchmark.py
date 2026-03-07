"""
Benchmark: how fast can this machine solve CFR + train NN?

Run on a DigitalOcean GPU droplet (or any machine) to get:
  - CFR solve time (all CPU cores)
  - NN training throughput (GPU if available, else CPU)
  - Projected total time for full production run (3.5M solves)

Usage:
  python benchmark.py

Prerequisites:
  - Rust toolchain (cargo)
  - Python 3 with PyTorch
"""

import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER_BIN = os.path.join(PROJECT_DIR, "target/release/examples/backend_solver")
CONFIG = os.path.join(PROJECT_DIR, "config/2c3c4h_p2.json")

# Production scale
TOTAL_SOLVES = 21 * 95 * 1755  # 3,508,725


def build_solver():
    if os.path.exists(SOLVER_BIN):
        return
    print("Building solver...")
    r = subprocess.run(
        ["cargo", "build", "--example", "backend_solver",
         "--release", "--features", "bincode rayon zstd jemalloc"],
        cwd=PROJECT_DIR, capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"Build failed:\n{r.stderr}")
        sys.exit(1)


def run_solve(threads, max_iter):
    with open(CONFIG) as f:
        cfg = json.load(f)
    cfg["output"]["filename"] = os.path.join(PROJECT_DIR, "_benchmark_tmp.flop")
    cfg["solver"]["maxIterations"] = max_iter

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=PROJECT_DIR) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(threads)
        r = subprocess.run([SOLVER_BIN, tmp_path], capture_output=True, text=True, env=env, timeout=600)
        if r.returncode != 0:
            print(f"  Solver failed (exit {r.returncode}):")
            print(f"  stderr: {r.stderr[:500]}")
            print(f"  stdout: {r.stdout[:500]}")
            return None
        out = r.stdout
        j_start, j_end = out.find("{"), out.rfind("}") + 1
        if j_start >= 0 and j_end > j_start:
            return json.loads(out[j_start:j_end])
        print(f"  Could not parse output: {out[:500]}")
        return None
    finally:
        os.unlink(tmp_path)
        flop_tmp = os.path.join(PROJECT_DIR, "_benchmark_tmp.flop")
        if os.path.exists(flop_tmp):
            os.unlink(flop_tmp)


def benchmark_cfr():
    n_cpu = multiprocessing.cpu_count()
    print(f"\n[ CFR SOLVE ] config: 2c3c4h_p2.json, {n_cpu} threads, 200 iterations\n")

    r = run_solve(n_cpu, 200)
    if not r:
        print("  FAILED")
        return None

    # Handle both camelCase and snake_case keys
    def get(key):
        snake = key
        camel = key.split("_")
        camel = camel[0] + "".join(w.capitalize() for w in camel[1:])
        return r.get(snake) or r.get(camel)

    if not get("solve_time_seconds"):
        print(f"  Solver error: {json.dumps(r, indent=2)}")
        return None

    t = get("solve_time_seconds")
    mem = get("memory_mb")
    expl = get("exploitability_percent")
    print(f"  Solve time:   {t:.2f}s")
    print(f"  Memory:       {mem:.0f} MB")
    print(f"  Exploitability: {expl:.3f}%")

    # Project to ~500 iters average
    per_solve = t * (500 / 200)
    total_h = per_solve * TOTAL_SOLVES / 3600
    print(f"\n  Projected per solve (~500 iters): {per_solve:.1f}s")
    print(f"  Full production ({TOTAL_SOLVES:,} solves):")
    print(f"    This machine alone: {total_h:,.0f} hours ({total_h/24:,.0f} days)")
    for n in [4, 8, 16, 32]:
        print(f"    {n:>2} machines:        {total_h/n:,.0f} hours ({total_h/n/24:,.0f} days)")
    print(f"\n  RAM for parallel solves: {mem * 4 / 1024:.1f} GB (4 parallel)")

    return {"solve_time": t, "memory_mb": mem, "per_solve_est": per_solve}


def benchmark_nn():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        print(f"\n[ NN TRAINING ] PyTorch import failed: {e}")
        return None

    # Pick best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dev_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dev_name = "Apple MPS"
    else:
        device = torch.device("cpu")
        dev_name = "CPU"

    print(f"\n[ NN TRAINING ] device: {dev_name}\n")

    # Test two model sizes
    tests = [
        ("nn1 (current)",  256, 3, 150,  5,  100_000),
        ("nn2 (production)", 1024, 5, 500, 10, 1_000_000),
    ]

    for name, hidden, n_layers, in_dim, out_dim, n_samples in tests:
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        model = nn.Sequential(*layers).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        X = torch.randn(n_samples, in_dim, device=device)
        Y = torch.softmax(torch.randn(n_samples, out_dim, device=device), dim=-1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Warmup
        for _ in range(3):
            pred = torch.softmax(model(X), dim=-1)
            loss = -(Y * torch.log(pred.clamp(min=1e-8))).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed run
        t0 = time.perf_counter()
        for _ in range(20):
            pred = torch.softmax(model(X), dim=-1)
            loss = -(Y * torch.log(pred.clamp(min=1e-8))).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_epoch = elapsed / 20 * 1000
        throughput = n_samples / (elapsed / 20)
        print(f"  {name}: {n_params:,} params, {n_samples:,} samples")
        print(f"    {ms_epoch:.0f} ms/epoch, {throughput:,.0f} samples/s")

        # Production projection (2.8B samples, 50 epochs)
        total_samples = 2_800_000_000 * 50
        train_hours = total_samples / throughput / 3600
        print(f"    Full training (50 epochs × 2.8B): ~{train_hours:,.0f} hours\n")

        del X, Y, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    n_cpu = multiprocessing.cpu_count()
    print("=" * 50)
    print("POSTFLOP SOLVER BENCHMARK")
    print("=" * 50)
    print(f"Machine:   {os.uname().nodename}")
    print(f"CPU cores: {n_cpu}")
    try:
        import torch
        print(f"PyTorch:   {torch.__version__}")
        if torch.cuda.is_available():
            print(f"GPU:       {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    build_solver()
    cfr = benchmark_cfr()
    benchmark_nn()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    main()
