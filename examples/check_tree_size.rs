//! Quick tree size check — builds the game tree and reports memory without solving.
//!
//! Usage:
//!   cargo run --example check_tree_size --release --features "bincode rayon" -- config/9s6d6c.json

#[path = "common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json> [config2.json ...]", args[0]);
        std::process::exit(1);
    }

    for config_path in &args[1..] {
        let config = load_config(config_path);
        let (card_config, tree_config) = parse_configs(&config).unwrap();

        let action_tree = ActionTree::new(tree_config.clone()).unwrap();
        let game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
        let (mem_uncompressed, mem_compressed) = game.memory_usage();
        let gb_unc = mem_uncompressed as f64 / 1024.0 / 1024.0 / 1024.0;
        let gb_comp = mem_compressed as f64 / 1024.0 / 1024.0 / 1024.0;

        println!("{}: {:.2} GB uncompressed, {:.2} GB compressed  (OOP={}, IP={})",
            config_path, gb_unc, gb_comp,
            game.num_private_hands(0), game.num_private_hands(1));
    }
}
