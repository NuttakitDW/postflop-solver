"""
Train NN with bucketed range histogram features for flop strategy prediction.

Adds to the baseline multi-board model:
  - OOP range histogram (K dims): fraction of OOP range in each strength bucket
  - IP range histogram (K dims): fraction of IP range in each strength bucket
  - Current hand's bucket one-hot (K dims): which bucket this hand falls in

Bucket assignment: sort hands by 5-card strength (2 hole + 3 board),
divide into K equal-population buckets (0 = weakest, K-1 = strongest).

Usage:
  python demo_nn_flop_bucket.py <board1.flop> <board2.flop> ...

Trains baseline (no buckets) + K=10, K=50, K=100 models and prints comparison.
Saves models to models/demo/<boards>_flop_bucket_K{K}.pt
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import postflop_solver

RANKS = "23456789TJQKA"
SUITS = "cdhs"

def card_str(cid):
    return f"{RANKS[cid // 4]}{SUITS[cid % 4]}"

def hand_str(c1, c2):
    return f"{card_str(c1)}{card_str(c2)}"

# ── 5-card hand evaluator (port of src/hand.rs evaluate_internal) ──

def keep_n_msb(x, n):
    ret = 0
    for _ in range(n):
        if x == 0:
            break
        bit = 1 << (x.bit_length() - 1)
        x ^= bit
        ret |= bit
    return ret

def find_straight(rankset):
    WHEEL = 0b1_0000_0000_1111  # A-2-3-4-5
    is_straight = rankset & (rankset << 1) & (rankset << 2) & (rankset << 3) & (rankset << 4)
    if is_straight:
        return keep_n_msb(is_straight, 1)
    if (rankset & WHEEL) == WHEEL:
        return 1 << 3  # wheel top is 5
    return 0

def evaluate_5card(cards):
    """Evaluate a 5-card hand. cards = list of 5 card IDs (rank*4 + suit).
    Returns comparable integer — higher = stronger."""
    rankset = 0
    rankset_suit = [0, 0, 0, 0]
    rank_count = [0] * 13

    for card in cards:
        rank = card // 4
        suit = card % 4
        rankset |= 1 << rank
        rankset_suit[suit] |= 1 << rank
        rank_count[rank] += 1

    rankset_of_count = [0] * 5
    for rank in range(13):
        rankset_of_count[rank_count[rank]] |= 1 << rank

    flush_suit = -1
    for suit in range(4):
        if bin(rankset_suit[suit]).count('1') >= 5:
            flush_suit = suit

    is_straight = find_straight(rankset)

    if flush_suit >= 0:
        is_sf = find_straight(rankset_suit[flush_suit])
        if is_sf:
            return (8 << 26) | is_sf  # straight flush
        else:
            return (5 << 26) | keep_n_msb(rankset_suit[flush_suit], 5)  # flush
    elif rankset_of_count[4]:
        remaining = keep_n_msb(rankset ^ rankset_of_count[4], 1)
        return (7 << 26) | (rankset_of_count[4] << 13) | remaining  # quads
    elif bin(rankset_of_count[3]).count('1') == 2:
        trips = keep_n_msb(rankset_of_count[3], 1)
        pair = rankset_of_count[3] ^ trips
        return (6 << 26) | (trips << 13) | pair  # full house
    elif rankset_of_count[3] and rankset_of_count[2]:
        pair = keep_n_msb(rankset_of_count[2], 1)
        return (6 << 26) | (rankset_of_count[3] << 13) | pair  # full house
    elif is_straight:
        return (4 << 26) | is_straight  # straight
    elif rankset_of_count[3]:
        remaining = keep_n_msb(rankset_of_count[1], 2)
        return (3 << 26) | (rankset_of_count[3] << 13) | remaining  # trips
    elif bin(rankset_of_count[2]).count('1') >= 2:
        pairs = keep_n_msb(rankset_of_count[2], 2)
        remaining = keep_n_msb(rankset ^ pairs, 1)
        return (2 << 26) | (pairs << 13) | remaining  # two pair
    elif rankset_of_count[2]:
        remaining = keep_n_msb(rankset_of_count[1], 3)
        return (1 << 26) | (rankset_of_count[2] << 13) | remaining  # one pair
    else:
        return keep_n_msb(rankset, 5)  # high card

# ── Sanity checks ──
def verify_evaluator():
    """Quick sanity checks for the hand evaluator."""
    # Board: 2c 3c 4h (card IDs: 2c=0, 3c=4, 4h=10)
    board = [0, 4, 10]
    # AA (AcAd = 48, 49) vs KK (KcKd = 44, 45) — AA should be stronger
    aa = evaluate_5card([48, 49] + board)
    kk = evaluate_5card([44, 45] + board)
    assert aa > kk, f"AA ({aa}) should beat KK ({kk})"
    # 56s (5c6c = 12, 16) makes a straight 2-3-4-5-6 — should beat AA
    straight = evaluate_5card([12, 16] + board)
    assert straight > aa, f"Straight ({straight}) should beat AA ({aa})"
    # Flush on monotone board: 2c 3c 4c (0, 4, 8) + 5c 7c (12, 20)
    flush = evaluate_5card([12, 20, 0, 4, 8])
    assert flush > straight, f"Flush ({flush}) should beat straight ({straight})"
    # High card should be weakest
    highcard = evaluate_5card([48, 20, 0, 4, 10])  # Ac 7c 2c 3c 4h — no pair, no flush
    pair = evaluate_5card([48, 49] + board)  # AA pair
    assert pair > highcard, f"Pair ({pair}) should beat high card ({highcard})"
    print("  Hand evaluator: OK")

# ── Bucketing ──

def compute_strengths(cards_list, board_cards):
    """Compute 5-card hand strength for each hand in the range.
    cards_list: list of (c1, c2) tuples
    board_cards: list of 3 card IDs
    Returns: np.array of shape [N] with comparable strength integers.
    """
    return np.array(
        [evaluate_5card(list(h) + board_cards) for h in cards_list],
        dtype=np.int64
    )

def buckets_from_strengths(strengths, K):
    """Assign hands to K equal-population buckets by strength percentile.
    Returns: (buckets [N], histogram [K])
    Bucket 0 = weakest, K-1 = strongest.
    """
    N = len(strengths)
    sorted_idx = np.argsort(strengths)  # ascending: weakest first
    buckets = np.zeros(N, dtype=np.int64)
    for i, idx in enumerate(sorted_idx):
        buckets[idx] = min(int(i * K / N), K - 1)
    histogram = np.bincount(buckets, minlength=K).astype(np.float32) / N
    return buckets, histogram

# ── Model ──

class FlopStrategyNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

# ── Training function ──

def train_model(X_t, Y_t, input_dim, max_actions, label, n_epochs=5000, hidden=512):
    model = FlopStrategyNet(input_dim, hidden, max_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining {label} ({n_params} params, in={input_dim}, hidden={hidden})...")

    best_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        pred = model(X_t)
        loss = -(Y_t * torch.log(pred.clamp(min=1e-8))).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l = loss.item()
        if l < best_loss:
            best_loss = l
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 1000 == 0:
            print(f"  epoch {epoch+1}: loss={l:.6f}")

    model.load_state_dict(best_state)
    print(f"  Best loss: {best_loss:.6f}")
    return model, best_loss

def evaluate_model(model, X_t, Y, all_boards_meta):
    """Evaluate max_diff per board. Returns dict of board_name -> max_diff."""
    with torch.no_grad():
        pred_all = model(X_t).numpy()

    results = {}
    idx = 0
    for meta in all_boards_meta:
        board_name = meta["board_name"]
        oop_cards = meta["oop_cards"]
        ip_cards = meta["ip_cards"]
        board_max = 0.0

        for node in meta["nodes"]:
            player = node["player"]
            n_act = node["n_act"]
            cards = oop_cards if player == 0 else ip_cards
            n_h = len(cards)

            cfr = Y[idx:idx + n_h, :n_act]
            nn_pred = pred_all[idx:idx + n_h, :n_act]
            diff = np.abs(nn_pred - cfr)
            board_max = max(board_max, diff.max())
            idx += n_h

        results[board_name] = board_max
    return results

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if len(sys.argv) < 2:
    print("Usage: python demo_nn_flop_bucket.py <board1.flop> <board2.flop> ...")
    sys.exit(1)

flop_paths = sys.argv[1:]
K_VALUES = [10, 50, 100]
MAX_HISTORY = 6
starting_pot = 55

print("Verifying evaluator...")
verify_evaluator()

# ── Phase 1: Scan boards for max_actions ──

print("\nScanning boards...")
max_actions_global = 0

for flop_path in flop_paths:
    game = postflop_solver.GameWrapper.load_from_file(flop_path)

    def scan_max_actions():
        global max_actions_global
        if game.is_terminal() or game.is_chance():
            return
        n = game.current_num_actions()
        if n > max_actions_global:
            max_actions_global = n
        hist = list(game.history())
        for i in range(n):
            game.play(i)
            scan_max_actions()
            game.back_to_root()
            for a in hist:
                game.play(a)

    game.back_to_root()
    scan_max_actions()
    print(f"  {flop_path}: OOP={game.num_private_hands(0)}, IP={game.num_private_hands(1)}")
    del game

print(f"Max actions across all boards: {max_actions_global}")
history_dim = MAX_HISTORY * max_actions_global

# ── Phase 2: Load boards, compute strengths, collect training data ──

print("\nComputing hand strengths and collecting data...")
all_boards_meta = []
Y_list = []
# Store per-board node encoding info for feature construction
board_encoding_data = []

for flop_path in flop_paths:
    game = postflop_solver.GameWrapper.load_from_file(flop_path)
    oop_cards = game.private_cards(0)
    ip_cards = game.private_cards(1)

    board_name = os.path.basename(flop_path).split("_")[0].split("-")[0]
    board_cards = []
    for i in range(0, 6, 2):
        r = RANKS.index(board_name[i].upper())
        s = SUITS.index(board_name[i + 1].lower())
        board_cards.append(r * 4 + s)

    # Board one-hot encoding
    board_enc = np.zeros(156, dtype=np.float32)
    for slot, cid in enumerate(board_cards):
        board_enc[slot * 52 + cid] = 1.0

    # Compute hand strengths (cached, independent of K)
    oop_strengths = compute_strengths(oop_cards, board_cards)
    ip_strengths = compute_strengths(ip_cards, board_cards)

    print(f"  {board_name}: OOP={len(oop_cards)} hands, IP={len(ip_cards)} hands")

    # Collect node data via DFS
    board_nodes = []
    node_samples = []  # list of (player, cards_idx_list, strategy, hist_enc, pot_frac) per node

    def collect_nodes():
        if game.is_terminal() or game.is_chance():
            return
        player = game.current_player()
        actions = game.current_actions()
        n_act, n_h, strat_flat = game.get_current_strategy()
        hist = list(game.history())
        bets = game.total_bet_amount()
        strategy = np.array(strat_flat).reshape(n_act, n_h)
        cards = oop_cards if player == 0 else ip_cards
        pot = starting_pot + bets[0] + bets[1]

        board_nodes.append({
            "history": hist,
            "player": player,
            "actions": actions,
            "n_act": n_act,
            "bets": bets,
        })

        # Encode history
        hist_enc = np.zeros(history_dim, dtype=np.float32)
        for step, action_idx in enumerate(hist):
            if step < MAX_HISTORY and action_idx < max_actions_global:
                hist_enc[step * max_actions_global + action_idx] = 1.0

        pot_frac = pot / (starting_pot + 360)

        # Store per-hand targets
        for h in range(n_h):
            y = np.zeros(max_actions_global, dtype=np.float32)
            y[:n_act] = strategy[:, h]
            Y_list.append(y)

        node_samples.append({
            "player": player,
            "n_h": n_h,
            "cards": cards,
            "hist_enc": hist_enc,
            "pot_frac": pot_frac,
            "board_enc": board_enc,
        })

        for i in range(len(actions)):
            game.play(i)
            collect_nodes()
            game.back_to_root()
            for a in hist:
                game.play(a)

    game.back_to_root()
    collect_nodes()

    meta = {
        "flop_path": flop_path,
        "board_name": board_name,
        "board_cards": board_cards,
        "n_oop": len(oop_cards),
        "n_ip": len(ip_cards),
        "oop_cards": oop_cards,
        "ip_cards": ip_cards,
        "nodes": board_nodes,
        "oop_strengths": oop_strengths,
        "ip_strengths": ip_strengths,
    }
    all_boards_meta.append(meta)
    board_encoding_data.append(node_samples)

    n_samples = sum(ns["n_h"] for ns in node_samples)
    print(f"    {len(board_nodes)} nodes, {n_samples} samples")
    del game

Y = np.array(Y_list)
Y_t = torch.from_numpy(Y)
total_samples = len(Y)
total_nodes = sum(len(m["nodes"]) for m in all_boards_meta)
print(f"\nTotal: {total_nodes} nodes, {total_samples} samples")

# ── Phase 3: Build features and train for each configuration ──

def build_features(all_boards_meta, board_encoding_data, K=None):
    """Build input feature matrix.
    K=None: baseline (no bucket features).
    K=int: add bucket features.
    """
    if K is not None:
        base_dim = 104 + 156 + 1 + history_dim + 1
        input_dim = base_dim + 3 * K
        # Precompute buckets and histograms per board
        board_buckets = []
        for meta in all_boards_meta:
            oop_b, oop_h = buckets_from_strengths(meta["oop_strengths"], K)
            ip_b, ip_h = buckets_from_strengths(meta["ip_strengths"], K)
            board_buckets.append({
                "oop_buckets": oop_b, "oop_hist": oop_h,
                "ip_buckets": ip_b, "ip_hist": ip_h,
            })
    else:
        input_dim = 104 + 156 + 1 + history_dim + 1
        board_buckets = None

    X_list = []
    for board_idx, (meta, node_samples) in enumerate(zip(all_boards_meta, board_encoding_data)):
        bb = board_buckets[board_idx] if board_buckets else None

        for ns in node_samples:
            player = ns["player"]
            cards = ns["cards"]
            n_h = ns["n_h"]
            hist_enc = ns["hist_enc"]
            pot_frac = ns["pot_frac"]
            board_enc = ns["board_enc"]

            for h in range(n_h):
                c1, c2 = cards[h]
                x = np.zeros(input_dim, dtype=np.float32)
                # Hand one-hot
                x[c1] = 1.0
                x[52 + c2] = 1.0
                # Board one-hot
                x[104:260] = board_enc
                # Player
                x[260] = float(player)
                # History
                x[261:261 + history_dim] = hist_enc
                # Pot fraction
                x[261 + history_dim] = pot_frac

                # Bucket features
                if bb is not None:
                    base = 262 + history_dim
                    # OOP histogram (always OOP first for consistency)
                    x[base:base + K] = bb["oop_hist"]
                    # IP histogram
                    x[base + K:base + 2 * K] = bb["ip_hist"]
                    # Current hand's bucket one-hot
                    if player == 0:
                        bucket_idx = bb["oop_buckets"][h]
                    else:
                        bucket_idx = bb["ip_buckets"][h]
                    x[base + 2 * K + bucket_idx] = 1.0

                X_list.append(x)

    X = np.array(X_list)
    return X, input_dim

# ── Train baseline ──

print(f"\n{'='*70}")
print("BASELINE (no bucket features)")
print(f"{'='*70}")

X_base, base_input_dim = build_features(all_boards_meta, board_encoding_data, K=None)
X_base_t = torch.from_numpy(X_base)

base_model, base_loss = train_model(X_base_t, Y_t, base_input_dim, max_actions_global, "baseline")
base_results = evaluate_model(base_model, X_base_t, Y, all_boards_meta)

for bname, mdiff in base_results.items():
    print(f"  {bname}: max_diff={mdiff:.6f}")
base_overall = max(base_results.values())
print(f"  Overall max_diff: {base_overall:.6f}")

# ── Train bucket models ──

bucket_results_all = {}

for K in K_VALUES:
    print(f"\n{'='*70}")
    print(f"K={K} BUCKET MODEL")
    print(f"{'='*70}")

    X_buck, buck_input_dim = build_features(all_boards_meta, board_encoding_data, K=K)
    X_buck_t = torch.from_numpy(X_buck)

    # Print bucket distribution
    for board_idx, meta in enumerate(all_boards_meta):
        oop_b, oop_h = buckets_from_strengths(meta["oop_strengths"], K)
        ip_b, ip_h = buckets_from_strengths(meta["ip_strengths"], K)
        print(f"  {meta['board_name']}: OOP buckets min/max count = {int(oop_h.min()*len(meta['oop_cards']))}/{int(oop_h.max()*len(meta['oop_cards']))}, "
              f"IP min/max = {int(ip_h.min()*len(meta['ip_cards']))}/{int(ip_h.max()*len(meta['ip_cards']))}")

    model_k, loss_k = train_model(X_buck_t, Y_t, buck_input_dim, max_actions_global, f"K={K}")
    results_k = evaluate_model(model_k, X_buck_t, Y, all_boards_meta)

    for bname, mdiff in results_k.items():
        delta = mdiff - base_results[bname]
        print(f"  {bname}: max_diff={mdiff:.6f}  (delta={delta:+.6f})")
    overall_k = max(results_k.values())
    delta_overall = overall_k - base_overall
    print(f"  Overall max_diff: {overall_k:.6f}  (delta={delta_overall:+.6f})")

    bucket_results_all[K] = {"results": results_k, "overall": overall_k, "loss": loss_k}

    # Save model
    model_dir = "models/demo"
    os.makedirs(model_dir, exist_ok=True)
    board_names = [m["board_name"] for m in all_boards_meta]
    model_name = "_".join(board_names) + f"_flop_bucket_K{K}.pt"
    model_path = os.path.join(model_dir, model_name)
    torch.save({
        "model": model_k.state_dict(),
        "input_dim": buck_input_dim,
        "max_actions": max_actions_global,
        "max_history": MAX_HISTORY,
        "starting_pot": starting_pot,
        "K": K,
        "boards": [{k: v for k, v in m.items() if k not in ("oop_strengths", "ip_strengths")} for m in all_boards_meta],
    }, model_path)
    print(f"  Model saved: {model_path}")

# ── Summary table ──

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  {'Config':<12} | {'Max Diff':<12} | {'Best Loss':<12} | {'Delta':<12}")
print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
print(f"  {'baseline':<12} | {base_overall:<12.6f} | {base_loss:<12.6f} | {'—':<12}")
for K in K_VALUES:
    r = bucket_results_all[K]
    delta = r["overall"] - base_overall
    print(f"  {'K='+str(K):<12} | {r['overall']:<12.6f} | {r['loss']:<12.6f} | {delta:<+12.6f}")
print()
