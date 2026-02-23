//! Solve a flop game using a pre-built oracle at the turn boundary.
//!
//! Loads a .toracle file and runs flop-only DCFR. The oracle provides CFVs
//! at turn chance nodes via matrix multiply, replacing the turn/river subtree.
//!
//! Usage:
//!   cargo run --example solve_with_oracle --release --features "bincode rayon" -- config/poc.json
//!
//! Prerequisite: build the oracle first:
//!   cargo run --example build_lookup_table --release --features "bincode rayon" -- config/poc.json

#[path = "../common/mod.rs"]
mod common;
#[path = "../poc_precompute/oracle_solver.rs"]
mod oracle_solver;

use common::*;
use oracle_solver::TreeOracle;
use postflop_solver::*;
use std::env;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }
    let config_path = &args[1];
    if !Path::new(config_path).exists() {
        eprintln!("Error: Config file not found: {}", config_path);
        std::process::exit(1);
    }
    let config = load_config(config_path);
    let (card_config, tree_config) = parse_configs(&config).unwrap();

    let oracle_path = config_path.replace(".json", ".toracle");
    if !Path::new(&oracle_path).exists() {
        eprintln!("Error: Oracle file not found: {}", oracle_path);
        eprintln!("Build it first:");
        eprintln!("  cargo run --example build_lookup_table --release --features \"bincode rayon\" -- {}", config_path);
        std::process::exit(1);
    }

    let total_start = Instant::now();

    println!("=== Oracle Solver ===");
    println!("Config: {}", config_path);
    #[cfg(feature = "rayon")]
    println!("Threads: {}", rayon::current_num_threads());
    println!("Mode: Oracle (flop-only DCFR)");
    println!();
    println!("Board: {}", config.board.flop);
    println!("Starting pot: {}", tree_config.starting_pot);
    println!("Effective stack: {}", tree_config.effective_stack);
    println!("Max iterations: {}", config.solver.max_iterations);
    println!("Output: {}", config.output.filename);
    println!();

    // Load oracle
    println!("Loading oracle: {}", oracle_path);
    let load_start = Instant::now();
    let oracle = TreeOracle::load(&oracle_path).expect("Failed to load oracle");
    let load_time = load_start.elapsed().as_secs_f64();
    let [oop_h, ip_h] = oracle.num_hands();
    let file_size = std::fs::metadata(&oracle_path).map(|m| m.len()).unwrap_or(0);
    println!("  {} boundary amounts, OOP={} hands, IP={} hands",
        oracle.num_amounts(), oop_h, ip_h);
    println!("  File size: {:.2} MB, load time: {:.3}s",
        file_size as f64 / 1_048_576.0, load_time);
    println!();

    // Build game tree
    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();

    let (mem_usage, _) = game.memory_usage();
    let memory_mb = mem_usage as f64 / 1024.0 / 1024.0;
    println!("  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0), game.num_private_hands(1));
    println!("  Memory: {:.2} MB", memory_mb);

    game.allocate_memory(false);

    // Verify oracle hand counts match
    if game.num_private_hands(0) != oop_h || game.num_private_hands(1) != ip_h {
        eprintln!("Error: Oracle hand counts ({}, {}) don't match game ({}, {})",
            oop_h, ip_h, game.num_private_hands(0), game.num_private_hands(1));
        std::process::exit(1);
    }
    println!();

    // Solve
    println!("Solving (oracle DCFR)...");
    let solve_start = Instant::now();
    oracle_solver::solve_flop_fixed_iterations(
        &game,
        &oracle,
        config.solver.max_iterations,
        true,
    );
    let solve_time = solve_start.elapsed().as_secs_f64();
    println!();

    // Finalize (full-tree CFV traversal required to set is_solved flag)
    println!("Finalizing (full-tree traversal)...");
    let finalize_start = Instant::now();
    finalize(&mut game);
    let finalize_time = finalize_start.elapsed().as_secs_f64();
    println!("  Finalize: {:.2}s", finalize_time);

    // Save
    let output_path = &config.output.filename;
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let memo = config.output.memo.as_deref().unwrap_or("oracle");
    let save_start = Instant::now();
    save_data_to_file(&game, memo, output_path, config.output.compression_level)
        .expect("Failed to save .flop file");
    let save_time = save_start.elapsed().as_secs_f64();
    println!("  Save: {:.2}s ({})", save_time, output_path);

    let total_time = total_start.elapsed().as_secs_f64();

    // Summary
    println!();
    println!("=== Summary ===");
    println!("Output file: {}", output_path);
    println!("Oracle load: {:.2}s", load_time);
    println!("Solve time: {:.2}s ({} iterations, {:.4}s/iter)",
        solve_time, config.solver.max_iterations,
        solve_time / config.solver.max_iterations as f64);
    println!("Finalize: {:.2}s (full-tree CFV traversal)", finalize_time);
    println!("Save: {:.2}s", save_time);
    println!("Total time: {:.2}s", total_time);
    println!("Memory: {:.2} MB", memory_mb);
    println!("OOP hands: {}, IP hands: {}", oop_h, ip_h);
}
