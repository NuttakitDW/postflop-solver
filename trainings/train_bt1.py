#!/usr/bin/env python3
"""
train_bt1.py — Train NN on .bt1 boundary data for CFV prediction.

Overfit on single board (KcQh7s). No train/test split.
Follows train_huber.py conventions (Ranger21, OneCycleLR, EMA, HuberLoss).

Uses solver indexing directly (no canonical 1326 expansion).
Input:  [boundary_onehot(25), player(1), cfreach_padded(max_hands)]
Output: [cfv_padded(max_hands)]  — masked loss on valid positions only.

Usage:
  python trainings/train_bt1.py
"""

import struct, json, os, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- constants --------
BOARD = "KcQh7s"

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
BT1_PATH = os.path.join(ROOT, "data", "bt1", f"{BOARD}.bt1")
OUT_DIR = os.path.join(ROOT, "models", f"bt1_{BOARD}")
os.makedirs(OUT_DIR, exist_ok=True)

# -------- hyper-parameters (following train_huber.py) --------
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

# -------- .bt1 loader --------
def load_bt1(path):
    """Parse .bt1 binary → list of iteration records + header info."""
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"BT1\0\0\0\0\0", f"Bad magic: {magic}"

        _version = struct.unpack("<I", f.read(4))[0]
        num_oop = struct.unpack("<I", f.read(4))[0]
        num_ip = struct.unpack("<I", f.read(4))[0]
        num_boundaries = struct.unpack("<I", f.read(4))[0]
        num_iterations = struct.unpack("<I", f.read(4))[0]
        starting_pot = struct.unpack("<f", f.read(4))[0]

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

    return records, num_oop, num_ip, num_boundaries, starting_pot

# -------- build dataset --------
def build_dataset(records, num_boundaries, num_oop, num_ip):
    """Build dataset in solver indexing, padded to max_hands."""
    max_hands = max(num_oop, num_ip)
    num_hands = [num_oop, num_ip]
    in_dim = num_boundaries + 1 + max_hands   # boundary_oh + player + cfreach
    out_dim = max_hands

    inputs, targets, masks = [], [], []

    for boundaries in records:
        for b in range(num_boundaries):
            for player in range(2):
                cfv_solver, cfreach_solver = boundaries[b][player]
                n_player = num_hands[player]

                # Input: boundary_oh + player + cfreach (padded)
                inp = np.zeros(in_dim, dtype=np.float32)
                inp[b] = 1.0                          # boundary one-hot
                inp[num_boundaries] = float(player)   # player flag
                inp[num_boundaries + 1:num_boundaries + 1 + len(cfreach_solver)] = cfreach_solver

                # Target: cfv (padded)
                tgt = np.zeros(out_dim, dtype=np.float32)
                tgt[:n_player] = cfv_solver

                # Mask: 1 for valid positions
                msk = np.zeros(out_dim, dtype=np.float32)
                msk[:n_player] = 1.0

                inputs.append(inp)
                targets.append(tgt)
                masks.append(msk)

    return np.array(inputs), np.array(targets), np.array(masks)

# -------- model (following train_huber.py) --------
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
    """HuberLoss only on valid (masked) positions."""
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.clamp(abs_diff, max=delta)
    loss = 0.5 * quad.pow(2) + delta * (abs_diff - quad)
    return (loss * mask).sum() / mask.sum()

# -------- plotting (following train_huber.py) --------
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
    t_start = time.time()
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    dev = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {dev}")

    # Load .bt1
    print(f"Loading bt1: {BT1_PATH}")
    records, num_oop, num_ip, num_boundaries, starting_pot = load_bt1(BT1_PATH)
    num_iters = len(records)
    max_hands = max(num_oop, num_ip)
    print(f"  OOP: {num_oop}, IP: {num_ip}, max: {max_hands}")
    print(f"  {num_boundaries} boundaries, {num_iters} iterations, pot={starting_pot}")

    # Build dataset
    print("Building dataset (solver indexing)...")
    X, Y, M = build_dataset(records, num_boundaries, num_oop, num_ip)
    n_samples = X.shape[0]
    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    n_valid = int(M.sum())
    print(f"  Samples: {n_samples}")
    print(f"  Input dim: {in_dim} (boundary_oh={num_boundaries} + player=1 + cfreach={max_hands})")
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

    # Ranger21 optimizer (following train_huber.py)
    from ranger21 import Ranger21
    opt = Ranger21(
        net.parameters(),
        lr=LR_MAX,
        weight_decay=WEIGHT_DECAY,
        num_epochs=EPOCHS,
        num_batches_per_epoch=len(loader),
    )

    # OneCycleLR (following train_huber.py)
    tot_steps = len(loader) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR_MAX, total_steps=tot_steps,
        pct_start=0.1, anneal_strategy="cos",
        cycle_momentum=False, div_factor=10, final_div_factor=1e4,
    )

    # EMA (following train_huber.py)
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
        player = int(X[i, num_boundaries])
        n_p = num_hands[player]
        diff = np.abs(pred_all[i, :n_p] - Y[i, :n_p])
        errs_by_player[player].append(diff.mean())

    for p in range(2):
        e = np.array(errs_by_player[p])
        label = "OOP" if p == 0 else "IP"
        print(f"  {label}: mean_err={e.mean():.6f}  max_err={e.max():.6f}  p99={np.percentile(e, 99):.6f} chips")

    # Save metadata for Rust
    meta = {
        "board": BOARD,
        "num_boundaries": num_boundaries,
        "num_iterations": num_iters,
        "starting_pot": starting_pot,
        "y_scale": saved_yscale,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "max_hands": max_hands,
        "num_oop": num_oop,
        "num_ip": num_ip,
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
