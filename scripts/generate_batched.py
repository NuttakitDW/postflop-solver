#!/usr/bin/env python3
"""
Batched data generation wrapper.
Runs generate_raw_data in small batches to avoid losing progress on slow boards.
Concatenates all batch outputs into a single dataset.
"""

import os
import sys
import subprocess
import numpy as np
import time
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "solver_output")

BATCH_SIZE = 50
TARGET_TOTAL = 1000
TARGET_EXPLOIT = 1.5
TIMEOUT_PER_BATCH = 180  # 3 min max per batch of 50


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    batch_dir = os.path.join(OUTPUT_DIR, "_batches")
    os.makedirs(batch_dir, exist_ok=True)

    total_collected = 0
    batch_num = 0
    all_meta = []
    all_ranges = []
    all_values = []

    start_time = time.time()
    seed = 42

    while total_collected < TARGET_TOTAL:
        batch_out = os.path.join(batch_dir, f"batch_{batch_num:04d}")
        os.makedirs(batch_out, exist_ok=True)

        remaining = TARGET_TOTAL - total_collected
        n = min(BATCH_SIZE, remaining)

        cmd = [
            os.path.join(PROJECT_ROOT, "target", "release", "examples", "generate_raw_data"),
            "--output-dir", batch_out,
            "--num-samples", str(n),
            "--target-exploit", str(TARGET_EXPLOIT),
            "--seed", str(seed),
        ]

        print(f"Batch {batch_num}: generating {n} samples (seed={seed})...", end=" ", flush=True)
        batch_start = time.time()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_PER_BATCH)

            if result.returncode != 0:
                print(f"FAILED (exit={result.returncode})")
                seed += 1
                batch_num += 1
                continue

            # Load batch output
            meta = np.load(os.path.join(batch_out, "meta.npy"))
            ranges = np.load(os.path.join(batch_out, "ranges.npy"))
            values = np.load(os.path.join(batch_out, "values.npy"))

            batch_time = time.time() - batch_start
            total_collected += meta.shape[0]
            all_meta.append(meta)
            all_ranges.append(ranges)
            all_values.append(values)

            elapsed = time.time() - start_time
            rate = total_collected / elapsed
            eta = (TARGET_TOTAL - total_collected) / rate if rate > 0 else 0

            print(f"OK ({meta.shape[0]} samples, {batch_time:.1f}s) | total={total_collected}/{TARGET_TOTAL} | rate={rate:.1f}/s | ETA={eta:.0f}s")

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT ({TIMEOUT_PER_BATCH}s)")

        seed += 1
        batch_num += 1

    # Concatenate all batches
    if all_meta:
        meta_all = np.concatenate(all_meta, axis=0)
        ranges_all = np.concatenate(all_ranges, axis=0)
        values_all = np.concatenate(all_values, axis=0)

        np.save(os.path.join(OUTPUT_DIR, "meta.npy"), meta_all)
        np.save(os.path.join(OUTPUT_DIR, "ranges.npy"), ranges_all)
        np.save(os.path.join(OUTPUT_DIR, "values.npy"), values_all)

        elapsed = time.time() - start_time
        print(f"\nDone! {meta_all.shape[0]} total samples in {elapsed:.0f}s ({meta_all.shape[0]/elapsed:.1f} samples/s)")
        print(f"Saved: meta={meta_all.shape}, ranges={ranges_all.shape}, values={values_all.shape}")

        # Cleanup batch dirs
        shutil.rmtree(batch_dir, ignore_errors=True)
    else:
        print("No data generated!")
        sys.exit(1)


if __name__ == "__main__":
    main()
