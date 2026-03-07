#!/usr/bin/env python3
"""
train_rl1.py — Online RL training for boundary CFV prediction.

Model predicts boundary CFVs, solver uses them for flop regret updates,
returns true CFVs as training labels. Dense per-step gradient updates.

Usage:
  python trainings/train_rl1.py config/test_small.json           # fast debug
  python trainings/train_rl1.py config/KcQh7s.json --episodes 20 # full training
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import postflop_solver

# ─── defaults ───
HIDDEN = 500
LAYERS = 7
LR = 1e-4
WEIGHT_DECAY = 1e-4
CLIP = 5.0
EPISODES = 50
SEED = 42


# ─── model (same architecture as bt1) ───
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, h=HIDDEN, n_layers=LAYERS):
        super().__init__()
        seq, d = [], in_dim
        for _ in range(n_layers):
            seq += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        seq.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


# ─── helpers ───
def build_input_batch(cfreaches, player, num_boundaries, max_hands, device):
    """Build [N_BOUNDS, in_dim] input tensor from cfreaches."""
    n = len(cfreaches)
    in_dim = num_boundaries + 1 + max_hands
    inputs = torch.zeros(n, in_dim, device=device)
    for b, cr in enumerate(cfreaches):
        inputs[b, b] = 1.0                          # boundary one-hot
        inputs[b, num_boundaries] = float(player)    # player flag
        inputs[b, num_boundaries + 1:num_boundaries + 1 + len(cr)] = torch.tensor(cr)
    return inputs


def tensor_to_cfv_lists(predicted, player, num_hands):
    """Convert model output [N_BOUNDS, max_hands] to list-of-lists for Rust.

    Extract only the first num_hands[player] values per boundary.
    """
    nh = num_hands[player]
    pred_np = predicted.detach().cpu().numpy()
    return [pred_np[b, :nh].tolist() for b in range(pred_np.shape[0])]


def build_target_batch(true_cfvs, player, max_hands, num_hands, device):
    """Build target tensor [N_BOUNDS, max_hands] and mask from Rust's true CFVs."""
    n = len(true_cfvs)
    nh = num_hands[player]
    targets = torch.zeros(n, max_hands, device=device)
    mask = torch.zeros(n, max_hands, device=device)
    for b, cfv in enumerate(true_cfvs):
        targets[b, :nh] = torch.tensor(cfv)
        mask[b, :nh] = 1.0
    return targets, mask


def masked_mse(pred, target, mask):
    """MSE loss only on valid (masked) positions."""
    diff = pred - target
    return ((diff ** 2) * mask).sum() / mask.sum()


def save_plots(episode_losses, episode_exploits, pot, path):
    """Save loss + exploitability curves."""
    if len(episode_losses) < 2:
        return
    eps = np.arange(1, len(episode_losses) + 1)

    exploit_pct = [e / pot * 100 for e in episode_exploits]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(eps, episode_losses)
    axes[0, 0].set_title("Loss (linear)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(eps, episode_losses)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Loss (log)")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("MSE Loss")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(eps, exploit_pct)
    axes[1, 0].axhline(y=0.5, color="orange", linestyle="--", label="Target 0.5%")
    axes[1, 0].set_title("Exploitability (linear)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Exploitability %")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(eps, exploit_pct)
    axes[1, 1].axhline(y=0.5, color="orange", linestyle="--", label="Target 0.5%")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Exploitability (log)")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Exploitability %")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─── main ───
def main():
    parser = argparse.ArgumentParser(description="RL1 online training")
    parser.add_argument("config", help="Path to config JSON")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--target", type=float, default=0.5, help="Stop when exploitability < target %%")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ── Setup ──
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join("models", f"rl1_{config_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== RL1 Online Training ===")
    print(f"Config: {args.config}")
    print(f"Device: {dev}")
    print(f"Episodes: {args.episodes}")
    print(f"LR: {args.lr}")
    print()

    game = postflop_solver.GameWrapper(args.config)
    num_boundaries = game.num_boundaries()
    num_hands = [game.num_private_hands(0), game.num_private_hands(1)]
    max_hands = max(num_hands)
    max_iters = game.max_iterations()
    pot = game.starting_pot()

    in_dim = num_boundaries + 1 + max_hands
    out_dim = max_hands

    print(f"Boundaries: {num_boundaries}")
    print(f"OOP hands: {num_hands[0]}, IP hands: {num_hands[1]}, max: {max_hands}")
    print(f"Iterations per episode: {max_iters}")
    print(f"Starting pot: {pot}")
    print(f"Input dim: {in_dim}, Output dim: {out_dim}")
    print()

    # ── Model ──
    model = Net(in_dim, out_dim).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    start_episode = 0
    episode_losses = []
    episode_exploits = []

    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=dev)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = ckpt["episode"]
        episode_losses = ckpt.get("losses", [])
        episode_exploits = ckpt.get("exploits", [])
        print(f"  Resumed at episode {start_episode}")

    best_exploit = float("inf")
    print(f"\nTraining for {args.episodes} episodes...")
    print()

    # ── Training loop ──
    for ep in range(start_episode, start_episode + args.episodes):
        ep_start = time.time()
        game.reset()
        model.train()

        total_loss = 0.0
        n_updates = 0

        for t in range(max_iters):
            for player in range(2):
                # 1. Collect model inputs
                cfreaches = game.collect_boundary_cfreaches(player)
                inputs = build_input_batch(cfreaches, player, num_boundaries, max_hands, dev)

                # 2. Model forward
                predicted = model(inputs)

                # 3. Pass to Rust solver (detached)
                model_cfvs = tensor_to_cfv_lists(predicted, player, num_hands)
                true_cfvs = game.solve_step_with_model(t, player, model_cfvs)

                # 4. Build targets
                targets, mask = build_target_batch(true_cfvs, player, max_hands, num_hands, dev)

                # 5. Loss + backprop
                loss = masked_mse(predicted, targets, mask)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()

                total_loss += loss.item()
                n_updates += 1

            # Progress
            if (t + 1) % 10 == 0 or t + 1 == max_iters:
                avg_so_far = total_loss / n_updates
                print(f"\r  ep {ep+1} iter {t+1}/{max_iters}  loss={avg_so_far:.6f}", end="", flush=True)

        # ── Episode done ──
        ep_time = time.time() - ep_start
        avg_loss = total_loss / n_updates

        # Compute exploitability
        exploit = game.compute_exploitability()
        exploit_pct = exploit / pot * 100

        episode_losses.append(avg_loss)
        episode_exploits.append(exploit)

        print(f"\r  Episode {ep+1:3d}  loss={avg_loss:.6f}  "
              f"exploit={exploit:.4f} chips ({exploit_pct:.2f}% of pot)  "
              f"time={ep_time:.1f}s")

        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": ep + 1,
            "losses": episode_losses,
            "exploits": episode_exploits,
            "config": args.config,
        }
        torch.save(ckpt, os.path.join(out_dir, "latest.pt"))

        if exploit < best_exploit:
            best_exploit = exploit
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))

        # Early stop
        if exploit_pct < args.target:
            print(f"\n  Hit target {args.target}% at episode {ep+1}. Stopping.")
            save_plots(episode_losses, episode_exploits, pot, os.path.join(out_dir, "curves.png"))
            csv_path = os.path.join(out_dir, "log.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "avg_loss", "exploitability_chips", "exploitability_pct"])
                for i, (l, e) in enumerate(zip(episode_losses, episode_exploits), 1):
                    w.writerow([i, f"{l:.8f}", f"{e:.6f}", f"{e/pot*100:.4f}"])
            break

        # Save plots + CSV
        save_plots(episode_losses, episode_exploits, pot, os.path.join(out_dir, "curves.png"))

        csv_path = os.path.join(out_dir, "log.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "avg_loss", "exploitability_chips", "exploitability_pct"])
            for i, (l, e) in enumerate(zip(episode_losses, episode_exploits), 1):
                w.writerow([i, f"{l:.8f}", f"{e:.6f}", f"{e/pot*100:.4f}"])

    # ── Done ──
    print()
    print(f"=== Training Complete ===")
    print(f"Best exploitability: {best_exploit:.4f} chips ({best_exploit/pot*100:.2f}% of pot)")
    print(f"Target: < {pot * 0.003:.4f} chips (0.3% of pot)")
    print(f"Checkpoints: {out_dir}/")


if __name__ == "__main__":
    main()
