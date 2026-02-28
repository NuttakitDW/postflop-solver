//! Dump private card mappings for a config as JSON.
//!
//! Usage:
//!   cargo run --example dump_hands --release --features "bincode rayon" -- config/KcQh7s.json
//!
//! Outputs: data/bt1/<board>_hands.json

#[path = "common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::io::Write;
use std::path::Path;

fn card_str(card: u8) -> String {
    let rank = match card >> 2 {
        0 => '2', 1 => '3', 2 => '4', 3 => '5', 4 => '6', 5 => '7',
        6 => '8', 7 => '9', 8 => 'T', 9 => 'J', 10 => 'Q', 11 => 'K', 12 => 'A', _ => '?',
    };
    let suit = match card & 3 { 0 => 'c', 1 => 'd', 2 => 'h', 3 => 's', _ => '?' };
    format!("{}{}", rank, suit)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }

    let config = load_config(&args[1]);
    let (card_config, tree_config) = parse_configs(&config).unwrap();

    let action_tree = ActionTree::new(tree_config).unwrap();
    let game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let config_filename = Path::new(&args[1]).file_stem().unwrap().to_str().unwrap();
    std::fs::create_dir_all("data/bt1").ok();
    let out_path = format!("data/bt1/{}_hands.json", config_filename);

    let mut f = std::fs::File::create(&out_path).unwrap();

    write!(f, "{{\n").unwrap();
    write!(f, "  \"board\": \"{}\",\n", config.board.flop).unwrap();

    for player in 0..2 {
        let name = if player == 0 { "oop" } else { "ip" };
        let cards = game.private_cards(player);
        write!(f, "  \"{}\": [\n", name).unwrap();
        for (i, &(c1, c2)) in cards.iter().enumerate() {
            let comma = if i + 1 < cards.len() { "," } else { "" };
            write!(f, "    [{}, {}, \"{}{}\"]{}",
                c1, c2, card_str(c1), card_str(c2), comma).unwrap();
            if (i + 1) % 8 == 0 || i + 1 == cards.len() { write!(f, "\n").unwrap(); }
        }
        write!(f, "  ]{}\n", if player == 0 { "," } else { "" }).unwrap();
    }
    write!(f, "}}\n").unwrap();

    println!("Saved: {} (OOP={}, IP={})", out_path,
        game.num_private_hands(0), game.num_private_hands(1));
}
