//! Build tree-extracted oracle (CFV matrices at turn boundary) and save to file.
//!
//! Steps: parse config → build full tree → solve → extract oracle → save .toracle
//!
//! Usage:
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
    let target_exploitability =
        tree_config.starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;
    let starting_pot = tree_config.starting_pot as f32;

    let config_filename = Path::new(config_path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap();
    let output_path = format!("data/{}", config_filename.replace(".json", ".toracle"));
    std::fs::create_dir_all("data").ok();

    println!("=== Build Tree-Extracted Oracle ===");
    println!("Board: {}", config.board.flop);
    println!("Pot: {}, Stack: {}", tree_config.starting_pot, tree_config.effective_stack);

    // Build and solve full tree
    println!();
    println!("--- Step 1: Full Solve ---");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let boundary_amounts = action_tree.boundary_amounts();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory(false);
    println!("  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0), game.num_private_hands(1));
    println!("  Boundary amounts: {:?}", boundary_amounts);

    let solve_start = Instant::now();
    let mut exploitability = compute_exploitability(&game);
    for t in 0..config.solver.max_iterations {
        if exploitability <= target_exploitability {
            break;
        }
        solve_step(&game, t);
        if (t + 1) % 10 == 0 || t + 1 == config.solver.max_iterations {
            exploitability = compute_exploitability(&game);
            let pct = exploitability / starting_pot * 100.0;
            print!("\r  iter: {} / {} (exploitability = {:.2}%)",
                t + 1, config.solver.max_iterations, pct);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }
    println!();
    let solve_time = solve_start.elapsed().as_secs_f64();
    let pct = exploitability / starting_pot * 100.0;
    println!("  Solved in {:.2}s (exploitability = {:.3}%)", solve_time, pct);

    // Extract oracle from solved tree
    println!();
    println!("--- Step 2: Extract Oracle ---");
    let extract_start = Instant::now();
    let oracle = TreeOracle::build(&game, true);
    let extract_time = extract_start.elapsed().as_secs_f64();
    println!("  Extracted in {:.2}s", extract_time);

    // Save
    oracle.save(&output_path).expect("Failed to save oracle");
    let file_size = std::fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);

    println!();
    println!("Done!");
    println!("  Solve time: {:.2}s", solve_time);
    println!("  Extract time: {:.2}s", extract_time);
    println!("  File size: {:.2} MB", file_size as f64 / 1_048_576.0);
    println!("  Saved to: {}", output_path);
}
