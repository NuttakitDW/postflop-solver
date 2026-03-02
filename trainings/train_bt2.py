#!/usr/bin/env python3
"""
train_bt2.py — Train NN on .bt2 boundary data with (pot, stack) features.

Unlike train_bt1.py which uses boundary one-hot encoding (tied to a specific
tree structure), this uses continuous (pot_norm, stack_norm) features.
This makes the model generalizable across different flop bet sizes.

Loads multiple .bt2 files (from different flop bet configs) and pools all
training data together.

Input:  [pot_norm(1), stack_norm(1), player(1), cfreach_padded(max_hands)]
Output: [cfv_padded(max_hands)]  — masked loss on valid positions only.

Usage:
  python trainings/train_bt2.py data/bt2/KcQh7s_f1.bt2 data/bt2/KcQh7s_f2.bt2 ...
  python trainings/train_bt2.py data/bt2/KcQh7s_f*.bt2
"""

import struct, json, os, csv, sys, time, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- constants --------
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")

# -------- hyper-parameters (following train_bt1.py) --------
RNG_SEED = 42
EPOCHS = 500
BATCH = 512
LR_MAX = 3e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 500
LAYERS = 7
CLIP = 5.0
EMA_DECAY = 0.999
HUBER_DELTA = 1.0

# -------- .bt2 loader --------
def load_bt2(path):
    """Parse .bt2 binary → iteration records + header info + boundary metadata."""
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"BT2\0\0\0\0\0", f"Bad magic: {magic}"

        _version = struct.unpack("<I", f.read(4))[0]
        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_boundaries = struct.unpack("<I", f.read(4))[0]
        num_iterations = struct.unpack("<I", f.read(4))[0]
        starting_pot = struct.unpack("<f", f.read(4))[0]
        effective_stack = struct.unpack("<f", f.read(4))[0]

        # Read boundary metadata table
        boundary_pots = []
        boundary_stacks = []
        for _ in range(num_boundaries):
            pot = struct.unpack("<f", f.read(4))[0]
            stack = struct.unpack("<f", f.read(4))[0]
            boundary_pots.append(pot)
            boundary_stacks.append(stack)

        num_hands = [num_oop, num_ip]
        records = []
        for _ in range(num_iterations):
            _iteration = struct.unpack("<I", f.read(4))[0]
            _exploitability = struct.unpack("<f", f.read(4))[0]
            _reserved = struct.unpack("<I", f.read(4))[0]

            boundaries = []
            for _ in range(num_boundaries):
                player_data = []
                for player in range(2):
                    opponent = player ^ 1
                    cfv = np.frombuffer(f.read(num_hands[player] * 4), dtype=np.float32).copy()
                    cfreach = np.frombuffer(f.read(num_hands[opponent] * 4), dtype=np.float32).copy()
                    player_data.append((cfv, cfreach))
                boundaries.append(player_data)

            records.append(boundaries)

    return {
        "records": records,
        "num_oop": num_oop,
        "num_ip": num_ip,
        "num_boundaries": num_boundaries,
        "starting_pot": starting_pot,
        "effective_stack": effective_stack,
        "boundary_pots": boundary_pots,
        "boundary_stacks": boundary_stacks,
    }

# -------- build dataset --------
def build_dataset(all_data):
    """Build dataset from multiple .bt2 files using (pot, stack) features."""
    # Verify all files share the same board (same num_oop, num_ip, starting_pot, effective_stack)
    ref = all_data[0]
    num_oop = ref["num_oop"]
    num_ip = ref["num_ip"]
    starting_pot = ref["starting_pot"]
    effective_stack = ref["effective_stack"]
    max_pot = starting_pot + 2 * effective_stack  # normalization constant

    for i, d in enumerate(all_data):
        assert d["num_oop"] == num_oop, f"File {i}: num_oop mismatch {d['num_oop']} vs {num_oop}"
        assert d["num_ip"] == num_ip, f"File {i}: num_ip mismatch {d['num_ip']} vs {num_ip}"
        assert d["starting_pot"] == starting_pot, f"File {i}: starting_pot mismatch"
        assert d["effective_stack"] == effective_stack, f"File {i}: effective_stack mismatch"

    max_hands = max(num_oop, num_ip)
    num_hands = [num_oop, num_ip]
    in_dim = 3 + max_hands   # pot_norm + stack_norm + player + cfreach
    out_dim = max_hands

    inputs, targets, masks = [], [], []

    for d in all_data:
        boundary_pots = d["boundary_pots"]
        boundary_stacks = d["boundary_stacks"]
        num_boundaries = d["num_boundaries"]

        for boundaries in d["records"]:
            for b in range(num_boundaries):
                pot_norm = boundary_pots[b] / max_pot
                stack_norm = boundary_stacks[b] / max_pot

                for player in range(2):
                    cfv_solver, cfreach_solver = boundaries[b][player]
                    n_player = num_hands[player]

                    # Input: pot_norm + stack_norm + player + cfreach (padded)
                    inp = np.zeros(in_dim, dtype=np.float32)
                    inp[0] = pot_norm
                    inp[1] = stack_norm
                    inp[2] = float(player)
                    inp[3:3 + len(cfreach_solver)] = cfreach_solver

                    # Target: cfv (padded)
                    tgt = np.zeros(out_dim, dtype=np.float32)
                    tgt[:n_player] = cfv_solver

                    # Mask: 1 for valid positions
                    msk = np.zeros(out_dim, dtype=np.float32)
                    msk[:n_player] = 1.0

                    inputs.append(inp)
                    targets.append(tgt)
                    masks.append(msk)

    return np.array(inputs), np.array(targets), np.array(masks), {
        "num_oop": num_oop,
        "num_ip": num_ip,
        "max_hands": max_hands,
        "starting_pot": starting_pot,
        "effective_stack": effective_stack,
        "max_pot": max_pot,
        "in_dim": in_dim,
        "out_dim": out_dim,
    }

# -------- model (following train_bt1.py) --------
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, h=500, n_layers=7):
        super().__init__()
        seq, d = [], in_dim
        for _ in range(n_layers):
            seq += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        seq.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)

# -------- masked huber loss --------
def masked_huber_loss(pred, target, mask, delta=1.0):
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.clamp(abs_diff, max=delta)
    loss = 0.5 * quad.pow(2) + delta * (abs_diff - quad)
    return (loss * mask).sum() / mask.sum()

# -------- plotting --------
def save_loss_plot(train_hist, path):
    skip = 5
    if len(train_hist) <= skip:
        return
    ep = np.arange(skip + 1, len(train_hist) + 1)
    tr = train_hist[skip:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, tr, label="Train")
    ax1.set_yscale("log")
    ax1.set_title("Full (log-y)")
    ax1.grid(alpha=0.3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    mid = len(tr) // 2
    if mid > 0:
        ax2.plot(ep[mid:], tr[mid:], label="Train")
        ax2.set_title("Zoom (last 50%)")
        ax2.set_xlabel("Epoch")
        ax2.grid(alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# -------- main --------
def main():
    if len(sys.argv) < 2:
        print("Usage: python trainings/train_bt2.py <file1.bt2> [file2.bt2] ...")
        print("       python trainings/train_bt2.py data/bt2/KcQh7s_f*.bt2")
        sys.exit(1)

    # Expand glob patterns
    bt2_paths = []
    for arg in sys.argv[1:]:
        expanded = sorted(glob.glob(arg))
        if expanded:
            bt2_paths.extend(expanded)
        else:
            bt2_paths.append(arg)

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in bt2_paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)
    bt2_paths = unique_paths

    # Derive board name from first file
    board = os.path.basename(bt2_paths[0]).split("_")[0].replace(".bt2", "")
    OUT_DIR = os.path.join(ROOT, "models", f"bt2_{board}")
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    dev = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {dev}")

    # Load all .bt2 files
    print(f"\nLoading {len(bt2_paths)} bt2 file(s):")
    all_data = []
    total_iters = 0
    total_boundaries = 0
    all_pot_stack = set()
    for path in bt2_paths:
        print(f"  {path}")
        d = load_bt2(path)
        all_data.append(d)
        total_iters += len(d["records"])
        total_boundaries += d["num_boundaries"]
        for p, s in zip(d["boundary_pots"], d["boundary_stacks"]):
            all_pot_stack.add((int(p), int(s)))
        print(f"    boundaries={d['num_boundaries']}, iterations={len(d['records'])}")

    ref = all_data[0]
    num_oop = ref["num_oop"]
    num_ip = ref["num_ip"]
    max_hands = max(num_oop, num_ip)
    starting_pot = ref["starting_pot"]
    effective_stack = ref["effective_stack"]
    max_pot = starting_pot + 2 * effective_stack

    print(f"\n  OOP: {num_oop}, IP: {num_ip}, max: {max_hands}")
    print(f"  Pot: {starting_pot}, Stack: {effective_stack}, max_pot: {max_pot}")
    print(f"  Total iterations across files: {total_iters}")
    print(f"  Total boundary slots: {total_boundaries}")

    sorted_ps = sorted(all_pot_stack)
    print(f"  Unique (pot, stack) pairs across all files: {len(sorted_ps)}")
    for pot, stack in sorted_ps:
        print(f"    pot={pot}, stack={stack}, pot_norm={pot/max_pot:.4f}, stack_norm={stack/max_pot:.4f}")

    # Build dataset
    print("\nBuilding dataset (pot/stack features)...")
    X, Y, M, info = build_dataset(all_data)
    n_samples = X.shape[0]
    in_dim = info["in_dim"]
    out_dim = info["out_dim"]
    n_valid = int(M.sum())
    print(f"  Samples: {n_samples}")
    print(f"  Input dim: {in_dim} (pot_norm=1 + stack_norm=1 + player=1 + cfreach={max_hands})")
    print(f"  Output dim: {out_dim} (max_hands={max_hands})")
    print(f"  Valid output values: {n_valid:,} / {n_samples * out_dim:,} ({100*n_valid/(n_samples*out_dim):.1f}%)")

    # Scale targets
    valid_vals = Y[M > 0]
    y_scale = np.abs(valid_vals).max() + 1e-8
    Y_scaled = Y / y_scale
    print(f"  Target scale: {y_scale:.4f}")

    # DataLoader
    loader = DataLoader(
        TensorDataset(torch.tensor(X), torch.tensor(Y_scaled), torch.tensor(M)),
        batch_size=BATCH, shuffle=True,
    )

    # Model
    net = Net(in_dim, out_dim, HIDDEN, LAYERS).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Param/valid ratio: {n_params / n_valid:.2f}x")

    # Ranger21 optimizer
    from ranger21 import Ranger21
    opt = Ranger21(
        net.parameters(),
        lr=LR_MAX,
        weight_decay=WEIGHT_DECAY,
        num_epochs=EPOCHS,
        num_batches_per_epoch=len(loader),
    )

    # OneCycleLR
    tot_steps = len(loader) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR_MAX, total_steps=tot_steps,
        pct_start=0.1, anneal_strategy="cos",
        cycle_momentum=False, div_factor=10, final_div_factor=1e4,
    )

    # EMA
    ema = {n: p.clone().detach() for n, p in net.named_parameters() if p.requires_grad}

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    train_hist = []
    best_loss = float("inf")

    for ep in range(1, EPOCHS + 1):
        net.train()
        total, n = 0.0, 0
        for xb, yb, mb in tqdm(loader, leave=False, desc=f"E{ep:03d}"):
            xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
            loss = masked_huber_loss(net(xb), yb, mb, HUBER_DELTA)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            sched.step()

            with torch.no_grad():
                for name, p in net.named_parameters():
                    if p.requires_grad:
                        ema[name].mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)
            total += loss.item() * mb.sum().item()
            n += mb.sum().item()
        tr_loss = total / n

        # Evaluate with EMA weights
        backup = {}
        with torch.no_grad():
            for name, p in net.named_parameters():
                if p.requires_grad:
                    backup[name] = p.data.clone()
                    p.data.copy_(ema[name])

        net.eval()
        ema_total, ema_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, mb in loader:
                xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
                l = masked_huber_loss(net(xb), yb, mb, HUBER_DELTA)
                ema_total += l.item() * mb.sum().item()
                ema_n += mb.sum().item()
        ema_loss = ema_total / ema_n

        with torch.no_grad():
            for name, p in net.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[name])

        train_hist.append(tr_loss)

        if ema_loss < best_loss:
            best_loss = ema_loss
            torch.save(
                {**ema, "_yscale": float(y_scale)},
                os.path.join(OUT_DIR, "best_ema.pt"),
            )

        plot_path = os.path.join(OUT_DIR, "loss_curve.png")
        save_loss_plot(train_hist, plot_path)

        if ep % 10 == 0 or ep == 1:
            rmse = (tr_loss ** 0.5) * y_scale
            ema_rmse = (ema_loss ** 0.5) * y_scale
            print(f"  Epoch {ep:3d}  train={tr_loss:.8f}  ema={ema_loss:.8f}  "
                  f"rmse={rmse:.4f}  ema_rmse={ema_rmse:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  best={best_loss:.8f}")

    final_rmse = (best_loss ** 0.5) * y_scale
    print(f"\nBest EMA loss: {best_loss:.8f}  RMSE(chips): {final_rmse:.4f}")

    # Save losses CSV
    csv_path = os.path.join(OUT_DIR, "losses.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train"])
        for i, tr in enumerate(train_hist, 1):
            w.writerow([i, f"{tr:.10f}"])
    print(f"Loss curve: {plot_path}")

    # Export ONNX from EMA weights
    print("\nExporting ONNX (EMA weights)...")
    ckpt = torch.load(os.path.join(OUT_DIR, "best_ema.pt"), weights_only=True)
    saved_yscale = float(ckpt.pop("_yscale"))
    net.load_state_dict(ckpt)
    net.eval()
    net.cpu()

    dummy = torch.randn(1, in_dim)
    onnx_path = os.path.join(OUT_DIR, "model.onnx")
    torch.onnx.export(
        net, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    onnx_size = os.path.getsize(onnx_path)
    print(f"  ONNX saved: {onnx_path} ({onnx_size / 1024 / 1024:.1f} MB)")

    # Verify ONNX on training data
    print("\nVerifying ONNX on training data...")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    pred_all = sess.run(None, {"input": X.astype(np.float32)})[0] * saved_yscale

    num_hands = [num_oop, num_ip]
    errs_by_player = {0: [], 1: []}
    for i in range(n_samples):
        player = int(X[i, 2])  # player is at index 2 (after pot_norm, stack_norm)
        n_p = num_hands[player]
        diff = np.abs(pred_all[i, :n_p] - Y[i, :n_p])
        errs_by_player[player].append(diff.mean())

    for p in range(2):
        e = np.array(errs_by_player[p])
        label = "OOP" if p == 0 else "IP"
        print(f"  {label}: mean_err={e.mean():.6f}  max_err={e.max():.6f}  p99={np.percentile(e, 99):.6f} chips")

    # Save metadata for Rust inference
    num_files = len(bt2_paths)
    meta = {
        "board": board,
        "num_files": num_files,
        "files": [os.path.basename(p) for p in bt2_paths],
        "total_iterations": total_iters,
        "starting_pot": starting_pot,
        "effective_stack": effective_stack,
        "max_pot": max_pot,
        "y_scale": saved_yscale,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "max_hands": max_hands,
        "num_oop": num_oop,
        "num_ip": num_ip,
        "unique_pot_stack": sorted_ps,
        "best_loss": float(best_loss),
        "rmse_chips": float(final_rmse),
    }
    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")

    total_time = time.time() - t_start
    print(f"\nDone! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Model: {onnx_path}")
    print(f"  Meta: {meta_path}")

if __name__ == "__main__":
    main()
