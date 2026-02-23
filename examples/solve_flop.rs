//! Load an `.oracle` file and solve the flop only.
//!
//! Uses the precomputed OracleLookupTable as boundary CFVs — no turn/river
//! solving at runtime. Produces flop strategies via DCFR.
//!
//! Usage:
//!   cargo run --example solve_flop --release --features "bincode rayon" -- config/template.json
//!
//! Requires:
//!   Run `build_oracle` first to create the `.oracle` file.

mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::mem::MaybeUninit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }

    let config = load_config(&args[1]);
    let (card_config, tree_config) = parse_configs(&config)
        .unwrap_or_else(|e| { eprintln!("Config error: {}", e); std::process::exit(1); });

    let target_exploitability =
        tree_config.starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;

    // Load oracle
    let oracle_path = config.output.filename.replace(".flop", ".oracle");
    println!("=== Solve Flop ===");
    println!("Board: {}", config.board.flop);
    println!("Loading oracle: {}", oracle_path);

    let load_start = Instant::now();
    let oracle = OracleLookupTable::load(&oracle_path)
        .unwrap_or_else(|e| { eprintln!("Failed to load oracle: {}", e); std::process::exit(1); });
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Loaded: {} entries ({:.2}s)", oracle.num_entries(), load_time);

    // Build flop-only game
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = FlopSolver::new(card_config, action_tree).unwrap();
    game.set_oracle(oracle);
    println!("FlopSolver: {} nodes", game.num_nodes());
    println!();

    // Solve
    let solve_start = Instant::now();
    let exploitability = solve_flop(
        &mut game,
        config.solver.max_iterations,
        target_exploitability,
        true,
    );
    let solve_time = solve_start.elapsed().as_secs_f64();
    let exploitability_pct = exploitability / tree_config.starting_pot as f32 * 100.0;

    println!();
    println!("Exploitability: {:.4} ({:.3}% of pot)", exploitability, exploitability_pct);
    println!("Solve time: {:.2}s", solve_time);

    // Print root CFVs
    println!();
    println!("--- Root CFVs ---");
    for player in 0..2 {
        let pname = if player == 0 { "OOP" } else { "IP" };
        let num_hands = game.num_private_hands(player);
        let cfreach = game.initial_weights(player ^ 1).to_vec();
        let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
        {
            let mut root = game.root();
            compute_cfvalue_recursive(&mut result, &game, &mut root, player, &cfreach, false);
        }
        let cfvs: Vec<f32> = result.iter().map(|v| unsafe { v.assume_init() }).collect();
        let weighted_sum: f64 = cfvs.iter().zip(game.initial_weights(player))
            .map(|(&v, &w)| v as f64 * w as f64).sum();
        println!("  {} ({} hands): weighted_sum={:.6}", pname, num_hands, weighted_sum);
    }

    println!();
    println!("Total time: {:.2}s (load {:.2}s + solve {:.2}s)",
        load_time + solve_time, load_time, solve_time);
}
