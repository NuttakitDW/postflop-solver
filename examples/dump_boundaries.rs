//! Dump turn boundary metadata (pot amounts, action paths) as JSON.
//!
//! Usage:
//!   cargo run --example dump_boundaries --release --features "bincode rayon" -- config/KcQh7s.json

#[path = "common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::io::Write;
use std::path::Path;

fn action_str(action: Action) -> String {
    match action {
        Action::Fold => "F".to_string(),
        Action::Check => "X".to_string(),
        Action::Call => "C".to_string(),
        Action::Bet(a) => format!("B{}", a),
        Action::Raise(a) => format!("R{}", a),
        Action::AllIn(a) => format!("A{}", a),
        Action::Chance(_) => "?".to_string(),
        Action::None => "".to_string(),
    }
}

struct BoundaryInfo {
    amount: i32,
    path: String,
}

fn collect_boundaries(game: &PostFlopGame) -> Vec<BoundaryInfo> {
    fn walk(node: &mut PostFlopNode, results: &mut Vec<BoundaryInfo>, path: &mut Vec<String>) {
        if node.is_terminal() { return; }
        if node.is_chance() && node.turn() == NOT_DEALT {
            results.push(BoundaryInfo {
                amount: node.amount(),
                path: path.join("-"),
            });
            return;
        }
        let num_actions = node.num_actions();
        for a in 0..num_actions {
            let mut child = node.play(a);
            let act = action_str(child.prev_action());
            path.push(act);
            walk(&mut child, results, path);
            path.pop();
        }
    }
    let mut root = game.root();
    let mut results = Vec::new();
    let mut path = Vec::new();
    walk(&mut root, &mut results, &mut path);
    results
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }

    let config = load_config(&args[1]);
    let (card_config, tree_config) = parse_configs(&config).unwrap();
    let starting_pot = tree_config.starting_pot;

    let action_tree = ActionTree::new(tree_config).unwrap();
    let game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let boundaries = collect_boundaries(&game);

    let config_filename = Path::new(&args[1]).file_stem().unwrap().to_str().unwrap();
    std::fs::create_dir_all("data/bt1").ok();
    let out_path = format!("data/bt1/{}_boundaries.json", config_filename);

    let mut f = std::fs::File::create(&out_path).unwrap();
    write!(f, "{{\n").unwrap();
    write!(f, "  \"board\": \"{}\",\n", config.board.flop).unwrap();
    write!(f, "  \"starting_pot\": {},\n", starting_pot).unwrap();
    write!(f, "  \"num_boundaries\": {},\n", boundaries.len()).unwrap();
    write!(f, "  \"boundaries\": [\n").unwrap();
    for (i, b) in boundaries.iter().enumerate() {
        let pot = starting_pot + 2 * b.amount;
        let comma = if i + 1 < boundaries.len() { "," } else { "" };
        write!(f, "    {{\"pot\": {}, \"path\": \"{}\"}}{}\n", pot, b.path, comma).unwrap();
    }
    write!(f, "  ]\n}}\n").unwrap();

    println!("Saved: {}", out_path);
    println!("Boundaries: {}", boundaries.len());
    for (i, b) in boundaries.iter().enumerate() {
        let pot = starting_pot + 2 * b.amount;
        println!("  B{}: pot={}, path={}", i, pot, b.path);
    }
}
