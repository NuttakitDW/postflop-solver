"""Benchmark ONNX model: CPU vs CoreML, batch vs loop."""
import sys
import time
import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else "model_1.onnx"
BATCH = 49  # all turn cards

print(f"Available providers: {ort.get_available_providers()}")
print()


def make_session(providers=None):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress ORT warnings
    if providers:
        return ort.InferenceSession(MODEL, opts, providers=providers)
    return ort.InferenceSession(MODEL, opts)


def bench(sess, combo, global_f, n_runs=5):
    input_names = [inp.name for inp in sess.get_inputs()]
    feed = {input_names[0]: combo, input_names[1]: global_f}
    # warmup
    for _ in range(3):
        sess.run(None, feed)
    # timed
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return min(times), np.median(times), np.mean(times)


def bench_loop(sess, n_runs=3):
    """Run BATCH individual batch=1 calls and time the total."""
    input_names = [inp.name for inp in sess.get_inputs()]
    combos = [np.random.randn(1, 1326, 19).astype(np.float32) for _ in range(BATCH)]
    globals_ = [np.random.randn(1, 20).astype(np.float32) for _ in range(BATCH)]
    # warmup
    for c, g in zip(combos, globals_):
        sess.run(None, {input_names[0]: c, input_names[1]: g})
    # timed
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for c, g in zip(combos, globals_):
            sess.run(None, {input_names[0]: c, input_names[1]: g})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return min(times), np.median(times), np.mean(times)


# Print model info
sess = make_session()
print("=== Model Info ===")
for inp in sess.get_inputs():
    print(f"  Input: {inp.name}  shape={inp.shape}  dtype={inp.type}")
for out in sess.get_outputs():
    print(f"  Output: {out.name}  shape={out.shape}  dtype={out.type}")
print()

# Prepare inputs
combo1 = np.random.randn(1, 1326, 19).astype(np.float32)
global1 = np.random.randn(1, 20).astype(np.float32)
combo49 = np.random.randn(BATCH, 1326, 19).astype(np.float32)
global49 = np.random.randn(BATCH, 20).astype(np.float32)

# Test configurations
configs = [
    ("CPU", ['CPUExecutionProvider'], None),
    ("CoreML-ML", [('CoreMLExecutionProvider', {
        'ModelFormat': 'MLProgram',
        'MLComputeUnits': 'ALL',
    }), 'CPUExecutionProvider'], None),
    ("CoreML-GPU", [('CoreMLExecutionProvider', {
        'ModelFormat': 'MLProgram',
        'MLComputeUnits': 'CPUAndGPU',
    }), 'CPUExecutionProvider'], None),
]

print(f"=== Benchmarks (best of 5 runs, batch={BATCH}) ===\n")
print(f"{'provider':<12} {'batch=1':>10} {'batch=49':>10} {'49×loop':>10} {'ratio b49/b1':>14} {'ratio loop/b1':>14}")
print("-" * 80)

for name, providers, _ in configs:
    try:
        sess = make_session(providers)
        b1_min, _, _ = bench(sess, combo1, global1)
        b49_min, _, _ = bench(sess, combo49, global49)
        loop_min, _, _ = bench_loop(sess)

        print(
            f"{name:<12} {b1_min:>8.1f}ms {b49_min:>8.1f}ms {loop_min:>8.1f}ms"
            f" {b49_min/b1_min:>12.1f}x {loop_min/b1_min:>12.1f}x"
        )
    except Exception as e:
        print(f"{name:<12} FAILED: {e}")

print()
print("If CoreML batch=49 ratio << 49x → GPU parallelism is helping")
