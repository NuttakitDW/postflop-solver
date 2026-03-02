//! Verify that `solve_step_with_model` is a no-op when given true CFVs.
//!
//! Runs two solves on the same config:
//! 1. Standard: uses `solve_step_for_player_recording` (normal full DCFR)
//! 2. Model: uses `solve_step_with_model` with model_cfvs = true CFVs from recording
//!
//! If `solve_step_with_model` is correct, both should produce identical strategies.
//!
//! Usage:
//!   cargo run --example verify_model_step --release --features "bincode rayon" -- config/KcQh7s.json

#[path = "../common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::io::{self, Write as _};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }
    let config_path = &args[1];
    let config = load_config(config_path);
    let (card_config, tree_config) = parse_configs(&config).unwrap();
    let max_iterations = config.solver.max_iterations as usize;

    println!("=== Verify solve_step_with_model ===");
    println!("Config: {}", config_path);
    println!("Iterations: {}", max_iterations);
    println!();

    // --- Solve 1: Standard (recording) ---
    println!("Building standard game tree...");
    let action_tree_std = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_std = PostFlopGame::with_config(card_config.clone(), action_tree_std).unwrap();
    let (mem, _) = game_std.memory_usage();
    println!("  Memory: {:.2} MB", mem as f64 / 1024.0 / 1024.0);
    println!("  OOP hands: {}, IP hands: {}",
        game_std.num_private_hands(0), game_std.num_private_hands(1));
    game_std.allocate_memory(false);

    println!("Solving standard...");
    let std_start = Instant::now();
    // Collect all boundary CFVs per iteration for the model solve
    let mut all_cfvs: Vec<[Vec<Vec<f32>>; 2]> = Vec::with_capacity(max_iterations);

    for t in 0..max_iterations {
        let p0_cfvs = solve_step_for_player_recording(&game_std, t as u32, 0);
        let p1_cfvs = solve_step_for_player_recording(&game_std, t as u32, 1);
        all_cfvs.push([p0_cfvs, p1_cfvs]);

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            print!("\r  iteration: {} / {}", t + 1, max_iterations);
            io::stdout().flush().unwrap();
        }
    }
    println!(" ({:.2}s)", std_start.elapsed().as_secs_f64());
    finalize(&mut game_std);

    // --- Solve 2: Model (with true CFVs) ---
    println!("Building model game tree...");
    let action_tree_model = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_model = PostFlopGame::with_config(card_config.clone(), action_tree_model).unwrap();
    game_model.allocate_memory(false);

    println!("Solving with model (true CFVs)...");
    let model_start = Instant::now();

    for t in 0..max_iterations {
        for player in 0..2 {
            let true_cfvs = solve_step_with_model(
                &game_model,
                t as u32,
                player,
                &all_cfvs[t][player],
            );

            // Verify returned true_cfvs match what recording produced
            let recorded = &all_cfvs[t][player];
            assert_eq!(true_cfvs.len(), recorded.len(),
                "Boundary count mismatch at iter={} player={}", t, player);

            for (b, (true_v, rec_v)) in true_cfvs.iter().zip(recorded.iter()).enumerate() {
                assert_eq!(true_v.len(), rec_v.len(),
                    "CFV length mismatch at iter={} player={} boundary={}", t, player, b);
                for (i, (&tv, &rv)) in true_v.iter().zip(rec_v.iter()).enumerate() {
                    if tv != rv {
                        eprintln!("MISMATCH at iter={} player={} boundary={} hand={}: true={} recorded={}",
                            t, player, b, i, tv, rv);
                        std::process::exit(1);
                    }
                }
            }
        }

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            print!("\r  iteration: {} / {}", t + 1, max_iterations);
            io::stdout().flush().unwrap();
        }
    }
    println!(" ({:.2}s)", model_start.elapsed().as_secs_f64());
    finalize(&mut game_model);

    // --- Compare strategies ---
    println!();
    println!("Comparing strategies...");

    let mut total_diff = 0.0f64;
    let mut max_diff = 0.0f64;
    let mut elements = 0usize;
    let mut nodes = 0usize;

    compare_nodes(&game_std, &game_model,
        &mut game_std.root(), &mut game_model.root(),
        &mut total_diff, &mut max_diff, &mut elements, &mut nodes);

    let avg_diff = if elements > 0 { total_diff / elements as f64 * 100.0 } else { 0.0 };

    println!();
    println!("=== Result ===");
    println!("  Nodes compared: {}", nodes);
    println!("  Elements compared: {}", elements);
    println!("  Avg strategy diff: {:.6}%", avg_diff);
    println!("  Max strategy diff: {:.6}%", max_diff * 100.0);

    if max_diff == 0.0 {
        println!();
        println!("  PASS: Strategies are IDENTICAL (byte-exact)");
    } else if avg_diff < 0.001 {
        println!();
        println!("  PASS: Strategies are effectively identical (avg diff < 0.001%)");
    } else {
        println!();
        println!("  FAIL: Strategies differ significantly");
        std::process::exit(1);
    }
}

fn compare_nodes(
    g1: &PostFlopGame,
    g2: &PostFlopGame,
    n1: &mut PostFlopNode,
    n2: &mut PostFlopNode,
    total_diff: &mut f64,
    max_diff: &mut f64,
    elements: &mut usize,
    nodes: &mut usize,
) {
    if n1.is_terminal() {
        return;
    }

    let na = n1.num_actions();
    if na == 0 {
        return;
    }

    if !n1.is_chance() {
        let player = n1.player();
        let nh = g1.num_private_hands(player);
        let s1 = n1.strategy();
        let s2 = n2.strategy();

        if !s1.is_empty() && !s2.is_empty() {
            *nodes += 1;
            // Normalize and compare
            let ns1 = normalize_strategy(s1, na, nh);
            let ns2 = normalize_strategy(s2, na, nh);

            for (a, b) in ns1.iter().zip(ns2.iter()) {
                let diff = (a - b).abs() as f64;
                *total_diff += diff;
                if diff > *max_diff {
                    *max_diff = diff;
                }
                *elements += 1;
            }
        }
    }

    for action in 0..na {
        compare_nodes(g1, g2, &mut n1.play(action), &mut n2.play(action),
            total_diff, max_diff, elements, nodes);
    }
}

fn normalize_strategy(strategy: &[f32], num_actions: usize, num_hands: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; num_actions * num_hands];
    for h in 0..num_hands {
        let mut sum = 0.0f32;
        for a in 0..num_actions {
            let v = strategy[a * num_hands + h];
            sum += v;
        }
        if sum > 0.0 {
            for a in 0..num_actions {
                result[a * num_hands + h] = strategy[a * num_hands + h] / sum;
            }
        } else {
            for a in 0..num_actions {
                result[a * num_hands + h] = 1.0 / num_actions as f32;
            }
        }
    }
    result
}
