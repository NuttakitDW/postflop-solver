#!/usr/bin/env python3
"""
Verify PyO3 bridge: mirrors the Rust verify_model_step example.

1. Game A: Normal DCFR via solve_step_recording, records true CFVs per iteration
2. Game B: solve_step_with_model using Game A's true CFVs as model_cfvs
3. Compare exploitabilities — must match (identity test)

Usage:
    python pyo3-bridge/verify.py [config_path]

Default: config/test_small.json
"""

import sys
import time
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

import postflop_solver


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/test_small.json"

    print("=== PyO3 Bridge Verification ===")
    print(f"Config: {config_path}")
    print()

    # --- Game A: standard solve (recording) ---
    game_a = postflop_solver.GameWrapper(config_path)
    max_iters = game_a.max_iterations()
    n_boundaries = game_a.num_boundaries()
    pot = game_a.starting_pot()

    print(f"Iterations: {max_iters}")
    print(f"Boundaries: {n_boundaries}")
    print(f"OOP hands: {game_a.num_private_hands(0)}, IP hands: {game_a.num_private_hands(1)}")
    print(f"Starting pot: {pot}")
    print()

    print("Game A: Standard solve (recording)...")
    t0 = time.time()

    all_cfvs = []  # all_cfvs[iter] = [p0_cfvs, p1_cfvs]
    for t in range(max_iters):
        p0 = game_a.solve_step_recording(t, 0)
        p1 = game_a.solve_step_recording(t, 1)
        all_cfvs.append([p0, p1])

        if (t + 1) % 10 == 0 or t + 1 == max_iters:
            print(f"\r  iteration: {t + 1} / {max_iters}", end="", flush=True)

    exploit_a = game_a.compute_exploitability()
    t1 = time.time()
    print(f" ({t1 - t0:.2f}s)")
    print(f"  Exploitability: {exploit_a:.6f} chips ({exploit_a / pot * 100:.4f}% of pot)")
    print()

    # --- Game B: solve_step_with_model with true CFVs ---
    print("Game B: solve_step_with_model (true CFVs as model)...")
    game_b = postflop_solver.GameWrapper(config_path)
    t2 = time.time()

    for t in range(max_iters):
        for player in range(2):
            true_cfvs = game_b.solve_step_with_model(t, player, all_cfvs[t][player])

            # Verify returned true_cfvs match Game A's recorded values
            recorded = all_cfvs[t][player]
            assert len(true_cfvs) == len(recorded), \
                f"Boundary count mismatch at iter={t} player={player}"
            for b, (tv, rv) in enumerate(zip(true_cfvs, recorded)):
                assert len(tv) == len(rv), \
                    f"CFV length mismatch at iter={t} player={player} boundary={b}"
                for i, (a, b_val) in enumerate(zip(tv, rv)):
                    if a != b_val:
                        print(f"\nMISMATCH at iter={t} player={player} boundary={b} hand={i}: "
                              f"true={a} recorded={b_val}")
                        sys.exit(1)

        if (t + 1) % 10 == 0 or t + 1 == max_iters:
            print(f"\r  iteration: {t + 1} / {max_iters}", end="", flush=True)

    exploit_b = game_b.compute_exploitability()
    t3 = time.time()
    print(f" ({t3 - t2:.2f}s)")
    print(f"  Exploitability: {exploit_b:.6f} chips ({exploit_b / pot * 100:.4f}% of pot)")
    print()

    # --- Compare ---
    diff = abs(exploit_a - exploit_b)
    print("=== Result ===")
    print(f"  Game A (recording): {exploit_a:.6f}")
    print(f"  Game B (model):     {exploit_b:.6f}")
    print(f"  Diff:               {diff:.10f}")

    if diff == 0.0:
        print()
        print("  PASS: Exploitabilities are IDENTICAL")
    elif diff < 1e-6:
        print()
        print("  PASS: Exploitabilities match within epsilon")
    else:
        print()
        print("  FAIL: Exploitabilities differ significantly")
        sys.exit(1)

    # --- Test: reset works ---
    print()
    print("Test: Reset and re-solve...")
    game_b.reset()
    for t in range(5):
        game_b.solve_step_recording(t, 0)
        game_b.solve_step_recording(t, 1)
    exploit_c = game_b.compute_exploitability()
    print(f"  After 5 iters: {exploit_c:.6f} chips")
    print("  PASS: Reset works")

    # --- Test: collect_boundary_cfreaches ---
    print()
    print("Test: collect_boundary_cfreaches...")
    game_b.reset()
    for player in range(2):
        cfreaches = game_b.collect_boundary_cfreaches(player)
        assert len(cfreaches) == n_boundaries, \
            f"Expected {n_boundaries} boundaries, got {len(cfreaches)}"
        nh = game_b.num_private_hands(player ^ 1)
        for i, cr in enumerate(cfreaches):
            assert len(cr) == nh, \
                f"Boundary {i} player {player}: expected {nh} hands, got {len(cr)}"
    print(f"  Shapes correct: {n_boundaries} boundaries, matching hand counts")
    print("  PASS: collect_boundary_cfreaches works")

    print()
    print("=== All tests passed ===")


if __name__ == "__main__":
    main()
