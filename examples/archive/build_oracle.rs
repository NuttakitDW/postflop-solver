//! Build an OracleLookupTable and save it to a `.oracle` file.
//!
//! Solves all turn+river subtrees for every (boundary_amount, turn_card),
//! extracts MatrixTurnCfv from each, and saves the lookup table to disk.
//!
//! Usage:
//!   cargo run --example build_oracle --release --features "bincode rayon" -- config/template.json
//!
//! Output:
//!   data/out/solution.oracle  (path derived from config output.filename)

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

    // Get boundary amounts from the action tree
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let boundary_amounts = action_tree.boundary_amounts();

    let flop = card_config.flop;
    let bet_config = TurnBetConfig {
        turn_bet_sizes: tree_config.turn_bet_sizes.clone(),
        river_bet_sizes: tree_config.river_bet_sizes.clone(),
        add_allin_threshold: tree_config.add_allin_threshold,
        force_allin_threshold: tree_config.force_allin_threshold,
        merging_threshold: tree_config.merging_threshold,
    };

    println!("=== Build Oracle ===");
    println!("Board: {}", config.board.flop);
    println!("Pot: {}, Stack: {}", tree_config.starting_pot, tree_config.effective_stack);
    println!("Boundary amounts: {:?}", boundary_amounts);
    println!("Max iterations: {}", config.solver.max_iterations);
    println!("Target exploitability: {:.2}% of pot", config.solver.target_exploitability_percent);
    println!();

    // Build oracle
    let start = Instant::now();
    let oracle = OracleLookupTable::build(
        flop,
        &card_config.range[0], &card_config.range[1],
        tree_config.starting_pot, tree_config.effective_stack,
        &boundary_amounts, &bet_config,
        config.solver.max_iterations, target_exploitability,
        true,
    );
    let build_time = start.elapsed().as_secs_f64();

    let matrix_mb = oracle.matrix_memory_bytes() as f64 / 1024.0 / 1024.0;
    println!("Oracle built: {} entries, {:.2} MB matrices, {:.2}s",
        oracle.num_entries(), matrix_mb, build_time);

    // Save to file
    let oracle_path = config.output.filename.replace(".flop", ".oracle");
    if let Some(parent) = Path::new(&oracle_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    oracle.save(&oracle_path).expect("Failed to save oracle");
    let file_mb = fs::metadata(&oracle_path).map(|m| m.len()).unwrap_or(0) as f64 / 1024.0 / 1024.0;
    println!("Saved to {} ({:.2} MB)", oracle_path, file_mb);
}
