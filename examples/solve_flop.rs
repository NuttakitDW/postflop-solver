//! Load an `.oracle` file and solve the flop only.
//!
//! Uses the precomputed OracleLookupTable at turn chance nodes — no turn/river
//! solving at runtime. Strategies are written directly into PostFlopGame,
//! then saved to a `.flop` file.
//!
//! Usage:
//!   cargo run --example solve_flop --release --features "bincode rayon zstd" -- config/template.json
//!
//! Requires:
//!   Run `build_oracle` first to create the `.oracle` file.

mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::fs;
use std::path::Path;
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

    println!("=== Solve Flop ===");
    println!("Board: {}", config.board.flop);

    // Load oracle
    let oracle_path = config.output.filename.replace(".flop", ".oracle");
    println!("Loading oracle: {}", oracle_path);

    let load_start = Instant::now();
    let oracle = OracleLookupTable::load(&oracle_path)
        .unwrap_or_else(|e| { eprintln!("Failed to load oracle: {}", e); std::process::exit(1); });
    let load_time = load_start.elapsed().as_secs_f64();
    println!("Loaded: {} entries ({:.2}s)", oracle.num_entries(), load_time);

    // Build PostFlopGame (full tree)
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory(false);

    // Create oracle context (flop→turn hand mappings)
    let oracle_ctx = OracleContext::new(&game, oracle);

    // Solve: DCFR on flop nodes, oracle at turn chance nodes
    println!();
    let solve_start = Instant::now();
    let exploitability = solve_with_oracle(
        &mut game,
        &oracle_ctx,
        config.solver.max_iterations,
        target_exploitability,
        true,
    );
    let solve_time = solve_start.elapsed().as_secs_f64();
    let exploitability_pct = exploitability / tree_config.starting_pot as f32 * 100.0;

    println!();
    println!("Exploitability: {:.4} ({:.3}% of pot)", exploitability, exploitability_pct);
    println!("Solve time: {:.2}s", solve_time);

    // Save .flop — strategies are already in PostFlopGame's nodes
    let output_path = &config.output.filename;
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    let memo = config.output.memo.as_deref().unwrap_or("solve_flop");
    save_data_to_file(&game, memo, output_path, config.output.compression_level)
        .expect("Failed to save .flop file");
    println!("Saved .flop: {}", output_path);

    println!();
    println!("Total time: {:.2}s (load {:.2}s + solve {:.2}s)",
        load_time + solve_time, load_time, solve_time);
}
