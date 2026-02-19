#!/usr/bin/env python3
"""
Training script for presentation: trains the bucketed value network on subset_100 data,
logs per-epoch train/test loss, saves checkpoints every 100 epochs, plots loss curves,
and exports a CSV report.

Usage:
    python scripts/train_presentation.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K = 1000
BOARD_FEATURES = 15
INPUT_DIM = BOARD_FEATURES + 2 * K   # 2015
OUTPUT_DIM = 2 * K                    # 2000
BOARD_GEOM_FEATURES = 12             # indices 0-11 for board grouping

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "experiment", "subset_100")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "presentation")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS = 1000
BATCH_SIZE = 32
LR = 3e-3
WEIGHT_DECAY = 1e-6
HIDDEN_DIM = 500
NUM_LAYERS = 7
HUBER_DELTA = 1.0
GRAD_CLIP = 5.0
SEED = 42

# ---------------------------------------------------------------------------
# Model (same architecture as train.py)
# ---------------------------------------------------------------------------

class ZeroSumCorrectionLayer(nn.Module):
    def forward(self, raw_output, range_oop, range_ip):
        cfv_oop = raw_output[:, :K]
        cfv_ip = raw_output[:, K:]
        game_value = (
            (range_oop * cfv_oop).sum(dim=1, keepdim=True)
            + (range_ip * cfv_ip).sum(dim=1, keepdim=True)
        )
        correction = game_value / 2.0
        oop_total = range_oop.sum(dim=1, keepdim=True).clamp(min=1e-8)
        ip_total = range_ip.sum(dim=1, keepdim=True).clamp(min=1e-8)
        cfv_oop_corrected = cfv_oop - correction / oop_total
        cfv_ip_corrected = cfv_ip - correction / ip_total
        return torch.cat([cfv_oop_corrected, cfv_ip_corrected], dim=1)


class TurnValueNetwork(nn.Module):
    def __init__(self, hidden_dim=500, num_layers=7):
        super().__init__()
        layers = []
        in_dim = INPUT_DIM
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.PReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, OUTPUT_DIM))
        self.backbone = nn.Sequential(*layers)
        self.zero_sum = ZeroSumCorrectionLayer()

    def forward(self, x):
        raw = self.backbone(x)
        range_oop = x[:, BOARD_FEATURES : BOARD_FEATURES + K]
        range_ip = x[:, BOARD_FEATURES + K :]
        return self.zero_sum(raw, range_oop, range_ip)


# ---------------------------------------------------------------------------
# Board-grouped train/test split
# ---------------------------------------------------------------------------

def board_grouped_split(inputs, test_fraction=0.2, seed=42):
    """Split by unique board geometry (80/20) to prevent data leakage."""
    n = inputs.shape[0]
    board_keys = np.round(inputs[:, :BOARD_GEOM_FEATURES], decimals=4)

    board_to_id = {}
    sample_board_ids = np.empty(n, dtype=np.int64)
    for i in range(n):
        key = tuple(board_keys[i])
        if key not in board_to_id:
            board_to_id[key] = len(board_to_id)
        sample_board_ids[i] = board_to_id[key]

    n_boards = len(board_to_id)
    board_counts = np.bincount(sample_board_ids, minlength=n_boards)

    if n_boards < 2:
        print(f"WARNING: Only {n_boards} unique board(s). Falling back to random split.")
        perm = np.random.RandomState(seed).permutation(n)
        test_size = max(1, int(n * test_fraction))
        return perm[test_size:], perm[:test_size]

    rng = np.random.RandomState(seed)
    board_perm = rng.permutation(n_boards)

    test_board_set = set()
    test_sample_count = 0
    target_test = max(1, int(n * test_fraction))

    for bid in board_perm:
        if test_sample_count >= target_test:
            break
        if len(test_board_set) >= n_boards - 1:
            break
        test_board_set.add(bid)
        test_sample_count += board_counts[bid]

    test_mask = np.isin(sample_board_ids, list(test_board_set))
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]

    train_boards = set(sample_board_ids[train_idx])
    test_boards = set(sample_board_ids[test_idx])
    assert train_boards.isdisjoint(test_boards), "Board leakage detected!"

    print(f"Board-grouped split: {n_boards} unique boards")
    print(f"  Train: {len(train_idx)} samples ({len(train_boards)} boards)")
    print(f"  Test:  {len(test_idx)} samples ({len(test_boards)} boards)")

    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets.npy"))
    print(f"Loaded {inputs.shape[0]} samples | input={inputs.shape[1]}, output={targets.shape[1]}")

    # 80/20 board-grouped split
    train_idx, test_idx = board_grouped_split(inputs, test_fraction=0.2, seed=SEED)

    train_x = torch.tensor(inputs[train_idx], dtype=torch.float32)
    train_y = torch.tensor(targets[train_idx], dtype=torch.float32)
    test_x = torch.tensor(inputs[test_idx], dtype=torch.float32)
    test_y = torch.tensor(targets[test_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Model
    model = TurnValueNetwork(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {NUM_LAYERS} layers x {HIDDEN_DIM} neurons, {num_params:,} params")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Warmup for 50 epochs, then cosine decay
    warmup_epochs = 50
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss: plain Huber loss
    criterion = nn.HuberLoss(reduction="mean", delta=HUBER_DELTA)

    # Tracking
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_epoch = 0

    print(f"\nHyperparameters:")
    print(f"  Epochs:       {EPOCHS}")
    print(f"  Batch size:   {BATCH_SIZE}")
    print(f"  Learning rate:{LR}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Huber delta:  {HUBER_DELTA}")
    print(f"  Grad clip:    {GRAD_CLIP}")
    print(f"  Scheduler:    Warmup({warmup_epochs}) + CosineDecay")
    print(f"\nTraining...\n")

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        epoch_train_loss = 0.0
        train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            bs = batch_x.size(0)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_train_loss += loss.item() * bs
            train_samples += bs

        scheduler.step()

        avg_train_loss = epoch_train_loss / max(train_samples, 1)

        # --- Test ---
        model.eval()
        epoch_test_loss = 0.0
        test_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                bs = batch_x.size(0)

                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                epoch_test_loss += loss.item() * bs
                test_samples += bs

        avg_test_loss = epoch_test_loss / max(test_samples, 1)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        # Track best
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), ckpt_path)

        # Log every 50 epochs + first and last
        lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:4d}/{EPOCHS} | "
                f"train={avg_train_loss:.6f}  test={avg_test_loss:.6f} | "
                f"lr={lr:.2e}"
            )

    print(f"\nBest test loss: {best_test_loss:.6f} at epoch {best_epoch}")

    # --- Save CSV report ---
    csv_path = os.path.join(OUTPUT_DIR, "training_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss"])
        for i in range(EPOCHS):
            writer.writerow([i + 1, f"{train_losses[i]:.8f}", f"{test_losses[i]:.8f}"])
    print(f"CSV report saved: {csv_path}")

    # --- Plot loss curves ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, EPOCHS + 1)

    # Full range
    ax1.plot(epochs_range, train_losses, label="Train Loss", linewidth=1.0, alpha=0.85)
    ax1.plot(epochs_range, test_losses, label="Test Loss", linewidth=1.0, alpha=0.85)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Huber Loss")
    ax1.set_title("Train vs Test Loss (Full)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.plot(epochs_range, train_losses, label="Train Loss", linewidth=1.0, alpha=0.85)
    ax2.plot(epochs_range, test_losses, label="Test Loss", linewidth=1.0, alpha=0.85)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Huber Loss (log scale)")
    ax2.set_title("Train vs Test Loss (Log Scale)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Loss curves saved: {plot_path}")

    # --- Summary ---
    print(f"\nCheckpoints saved at: {OUTPUT_DIR}/")
    print(f"  best_model.pt (epoch {best_epoch})")
    for e in range(100, EPOCHS + 1, 100):
        print(f"  checkpoint_epoch_{e}.pt")


if __name__ == "__main__":
    main()
