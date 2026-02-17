/// Extract turn ranges after flop check-check from a solved flop game.
///
/// Usage:
///   cargo run --example extract_turn_ranges --release --features "bincode,zstd,rayon" -- config/template.json 2s
///
/// Arguments:
///   1. Config JSON (flop game config)
///   2. Turn card (e.g., "2s", "Ah", "Tc")
///
/// Outputs:
///   - OOP and IP range strings for the turn
///   - A new config JSON ready to solve from the turn
use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverConfig {
    board: BoardConfig,
    ranges: RangesConfig,
    bet_sizes: BetSizesConfig,
    tree: TreeSettings,
    solver: SolverSettings,
    output: OutputConfig,
}

#[derive(Debug, Deserialize)]
struct BoardConfig {
    flop: String,
    #[serde(default)]
    turn: Option<String>,
    #[serde(default)]
    river: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RangesConfig {
    oop: String,
    ip: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BetSizesConfig {
    oop_flop_bet: String,
    oop_flop_raise: String,
    oop_turn_bet: String,
    oop_turn_raise: String,
    #[serde(default)]
    oop_turn_donk: String,
    oop_river_bet: String,
    oop_river_raise: String,
    #[serde(default)]
    oop_river_donk: String,
    ip_flop_bet: String,
    ip_flop_raise: String,
    ip_turn_bet: String,
    ip_turn_raise: String,
    ip_river_bet: String,
    ip_river_raise: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TreeSettings {
    starting_pot: i32,
    effective_stack: i32,
    #[serde(default)]
    rake_percent: f64,
    #[serde(default)]
    rake_cap: f64,
    #[serde(default)]
    donk_option: u8,
    #[serde(default = "default_150")]
    add_all_in_threshold: f64,
    #[serde(default = "default_20")]
    force_all_in_threshold: f64,
    #[serde(default = "default_10")]
    merging_threshold: f64,
    #[serde(default)]
    max_raises_per_street: i32,
}

fn default_150() -> f64 { 150.0 }
fn default_20() -> f64 { 20.0 }
fn default_10() -> f64 { 10.0 }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverSettings {
    #[serde(default = "default_1000")]
    max_iterations: u32,
    #[serde(default = "default_half")]
    target_exploitability_percent: f32,
    #[serde(default)]
    use_compression: bool,
}

fn default_1000() -> u32 { 1000 }
fn default_half() -> f32 { 0.5 }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OutputConfig {
    filename: String,
    #[serde(default)]
    compression_level: Option<i32>,
    #[serde(default)]
    memo: Option<String>,
}

/// Output config for turn sim
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnConfig {
    board: TurnBoard,
    ranges: TurnRanges,
    bet_sizes: TurnBetSizes,
    tree: TurnTree,
    solver: TurnSolver,
    output: TurnOutput,
}

#[derive(Debug, Serialize)]
struct TurnBoard {
    flop: String,
    turn: String,
    river: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct TurnRanges {
    oop: String,
    ip: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnBetSizes {
    oop_flop_bet: String,
    oop_flop_raise: String,
    oop_turn_bet: String,
    oop_turn_raise: String,
    oop_turn_donk: String,
    oop_river_bet: String,
    oop_river_raise: String,
    oop_river_donk: String,
    ip_flop_bet: String,
    ip_flop_raise: String,
    ip_turn_bet: String,
    ip_turn_raise: String,
    ip_river_bet: String,
    ip_river_raise: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnTree {
    starting_pot: i32,
    effective_stack: i32,
    rake_percent: f64,
    rake_cap: f64,
    donk_option: u8,
    add_all_in_threshold: f64,
    force_all_in_threshold: f64,
    merging_threshold: f64,
    max_raises_per_street: i32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnSolver {
    max_iterations: u32,
    target_exploitability_percent: f32,
    use_compression: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnOutput {
    filename: String,
    compression_level: Option<i32>,
    memo: String,
}

fn normalize_bet_sizes(s: &str) -> String {
    s.split(',')
        .map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return trimmed.to_string();
            }
            let lower = trimmed.to_lowercase();
            if lower == "a"
                || lower.ends_with('%')
                || lower.ends_with('x')
                || lower.contains('c')
                || lower.contains('e')
            {
                trimmed.to_string()
            } else if trimmed.parse::<f64>().is_ok() {
                format!("{}%", trimmed)
            } else {
                trimmed.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <config.json> <turn_card>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} config/template.json 2s", args[0]);
        eprintln!();
        eprintln!("This will:");
        eprintln!("  1. Solve the flop game from the config");
        eprintln!("  2. Navigate to check-check on flop + the given turn card");
        eprintln!("  3. Extract OOP and IP ranges at that point");
        eprintln!("  4. Output a new turn config JSON you can use with backend_solver");
        std::process::exit(1);
    }

    let config_path = &args[1];
    let turn_card_str = &args[2];

    // Parse config
    let config_content = fs::read_to_string(config_path)
        .unwrap_or_else(|e| { eprintln!("Failed to read {}: {}", config_path, e); std::process::exit(1); });
    let config: SolverConfig = serde_json::from_str(&config_content)
        .unwrap_or_else(|e| { eprintln!("Failed to parse JSON: {}", e); std::process::exit(1); });

    // Parse turn card
    let turn_card = card_from_str(turn_card_str)
        .unwrap_or_else(|e| { eprintln!("Failed to parse turn card '{}': {}", turn_card_str, e); std::process::exit(1); });

    // Build flop game
    let flop = flop_from_str(&config.board.flop).unwrap();
    let oop_range: Range = config.ranges.oop.parse().unwrap();
    let ip_range: Range = config.ranges.ip.parse().unwrap();

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let oop_flop_bet = normalize_bet_sizes(&config.bet_sizes.oop_flop_bet);
    let oop_flop_raise = normalize_bet_sizes(&config.bet_sizes.oop_flop_raise);
    let oop_turn_bet = normalize_bet_sizes(&config.bet_sizes.oop_turn_bet);
    let oop_turn_raise = normalize_bet_sizes(&config.bet_sizes.oop_turn_raise);
    let oop_river_bet = normalize_bet_sizes(&config.bet_sizes.oop_river_bet);
    let oop_river_raise = normalize_bet_sizes(&config.bet_sizes.oop_river_raise);
    let ip_flop_bet = normalize_bet_sizes(&config.bet_sizes.ip_flop_bet);
    let ip_flop_raise = normalize_bet_sizes(&config.bet_sizes.ip_flop_raise);
    let ip_turn_bet = normalize_bet_sizes(&config.bet_sizes.ip_turn_bet);
    let ip_turn_raise = normalize_bet_sizes(&config.bet_sizes.ip_turn_raise);
    let ip_river_bet = normalize_bet_sizes(&config.bet_sizes.ip_river_bet);
    let ip_river_raise = normalize_bet_sizes(&config.bet_sizes.ip_river_raise);

    let oop_flop = BetSizeOptions::try_from((oop_flop_bet.as_str(), oop_flop_raise.as_str())).unwrap();
    let oop_turn = BetSizeOptions::try_from((oop_turn_bet.as_str(), oop_turn_raise.as_str())).unwrap();
    let oop_river = BetSizeOptions::try_from((oop_river_bet.as_str(), oop_river_raise.as_str())).unwrap();
    let ip_flop = BetSizeOptions::try_from((ip_flop_bet.as_str(), ip_flop_raise.as_str())).unwrap();
    let ip_turn_bs = BetSizeOptions::try_from((ip_turn_bet.as_str(), ip_turn_raise.as_str())).unwrap();
    let ip_river = BetSizeOptions::try_from((ip_river_bet.as_str(), ip_river_raise.as_str())).unwrap();

    let turn_donk = if (config.tree.donk_option == 1 || config.tree.donk_option == 3)
        && !config.bet_sizes.oop_turn_donk.is_empty()
    {
        let s = normalize_bet_sizes(&config.bet_sizes.oop_turn_donk);
        Some(DonkSizeOptions::try_from(s.as_str()).unwrap())
    } else {
        None
    };

    let river_donk = if (config.tree.donk_option == 2 || config.tree.donk_option == 3)
        && !config.bet_sizes.oop_river_donk.is_empty()
    {
        let s = normalize_bet_sizes(&config.bet_sizes.oop_river_donk);
        Some(DonkSizeOptions::try_from(s.as_str()).unwrap())
    } else {
        None
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: config.tree.starting_pot,
        effective_stack: config.tree.effective_stack,
        rake_rate: config.tree.rake_percent / 100.0,
        rake_cap: config.tree.rake_cap,
        flop_bet_sizes: [oop_flop, ip_flop],
        turn_bet_sizes: [oop_turn, ip_turn_bs],
        river_bet_sizes: [oop_river, ip_river],
        turn_donk_sizes: turn_donk,
        river_donk_sizes: river_donk,
        add_allin_threshold: config.tree.add_all_in_threshold / 100.0,
        force_allin_threshold: config.tree.force_all_in_threshold / 100.0,
        merging_threshold: config.tree.merging_threshold / 100.0,
        max_raises_per_street: config.tree.max_raises_per_street,
    };

    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let (mem, _) = game.memory_usage();
    eprintln!("Memory: {:.2}MB", mem as f64 / 1024.0 / 1024.0);

    game.allocate_memory(config.solver.use_compression);

    // Solve
    let target = game.tree_config().starting_pot as f32
        * config.solver.target_exploitability_percent / 100.0;
    eprintln!("Solving flop game...");
    let exploit = solve(&mut game, config.solver.max_iterations, target, true);
    eprintln!("Exploitability: {:.4} ({:.3}% of pot)",
        exploit, exploit / game.tree_config().starting_pot as f32 * 100.0);

    // Navigate to check-check + turn card
    game.back_to_root();

    let flop_actions = game.available_actions();
    assert!(matches!(flop_actions[0], Action::Check), "First OOP action must be Check");
    game.play(0); // OOP check

    let ip_actions = game.available_actions();
    assert!(matches!(ip_actions[0], Action::Check), "First IP action must be Check");
    game.play(0); // IP check

    assert!(game.is_chance_node(), "Expected chance node after check-check");
    game.play(turn_card as usize);

    // Extract ranges
    let oop_hands = game.private_cards(0);
    let oop_weights = game.weights(0);
    let ip_hands = game.private_cards(1);
    let ip_weights = game.weights(1);

    let oop_turn_range = Range::from_hands_weights(oop_hands, oop_weights).unwrap();
    let ip_turn_range = Range::from_hands_weights(ip_hands, ip_weights).unwrap();

    let oop_range_str = oop_turn_range.to_string();
    let ip_range_str = ip_turn_range.to_string();

    // Print ranges to stderr for human reading
    let oop_nonzero = oop_weights.iter().filter(|&&w| w > 0.0).count();
    let ip_nonzero = ip_weights.iter().filter(|&&w| w > 0.0).count();
    eprintln!();
    eprintln!("=== Extracted Ranges (after flop check-check, turn {}) ===", turn_card_str);
    eprintln!("OOP ({} hands): {}", oop_nonzero, oop_range_str);
    eprintln!();
    eprintln!("IP ({} hands): {}", ip_nonzero, ip_range_str);
    eprintln!();

    // Build output config JSON
    let turn_config = TurnConfig {
        board: TurnBoard {
            flop: config.board.flop.clone(),
            turn: turn_card_str.to_string(),
            river: serde_json::Value::Null,
        },
        ranges: TurnRanges {
            oop: oop_range_str,
            ip: ip_range_str,
        },
        bet_sizes: TurnBetSizes {
            oop_flop_bet: config.bet_sizes.oop_flop_bet.clone(),
            oop_flop_raise: config.bet_sizes.oop_flop_raise.clone(),
            oop_turn_bet: config.bet_sizes.oop_turn_bet.clone(),
            oop_turn_raise: config.bet_sizes.oop_turn_raise.clone(),
            oop_turn_donk: config.bet_sizes.oop_turn_donk.clone(),
            oop_river_bet: config.bet_sizes.oop_river_bet.clone(),
            oop_river_raise: config.bet_sizes.oop_river_raise.clone(),
            oop_river_donk: config.bet_sizes.oop_river_donk.clone(),
            ip_flop_bet: config.bet_sizes.ip_flop_bet.clone(),
            ip_flop_raise: config.bet_sizes.ip_flop_raise.clone(),
            ip_turn_bet: config.bet_sizes.ip_turn_bet.clone(),
            ip_turn_raise: config.bet_sizes.ip_turn_raise.clone(),
            ip_river_bet: config.bet_sizes.ip_river_bet.clone(),
            ip_river_raise: config.bet_sizes.ip_river_raise.clone(),
        },
        tree: TurnTree {
            starting_pot: config.tree.starting_pot, // same pot after check-check
            effective_stack: config.tree.effective_stack, // same stack after check-check
            rake_percent: config.tree.rake_percent,
            rake_cap: config.tree.rake_cap,
            donk_option: config.tree.donk_option,
            add_all_in_threshold: config.tree.add_all_in_threshold,
            force_all_in_threshold: config.tree.force_all_in_threshold,
            merging_threshold: config.tree.merging_threshold,
            max_raises_per_street: config.tree.max_raises_per_street,
        },
        solver: TurnSolver {
            max_iterations: config.solver.max_iterations,
            target_exploitability_percent: config.solver.target_exploitability_percent,
            use_compression: config.solver.use_compression,
        },
        output: TurnOutput {
            filename: format!("turn_{}.flop",
                turn_card_str),
            compression_level: config.output.compression_level,
            memo: format!("Turn {} after flop {} check-check",
                turn_card_str, config.board.flop),
        },
    };

    // Write JSON to file
    let json = serde_json::to_string_pretty(&turn_config).unwrap();
    let output_path = format!("config/turn_{}.json", turn_card_str);
    fs::write(&output_path, &json).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {}", output_path, e);
        std::process::exit(1);
    });

    println!();
    println!("=== Turn config saved to {} ===", output_path);
    println!();
    println!("Now solve with:");
    println!("  cargo run --example backend_solver --release --features \"bincode,zstd,rayon\" -- {}", output_path);
}
