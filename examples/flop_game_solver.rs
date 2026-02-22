//! FlopGame solver: solves the flop using FlopGame + TurnOracle,
//! injects strategies into PostFlopGame via node locking,
//! then solves turn/river with standard DCFR. Exports as .flop.
//!
//! Usage:
//!   cargo run --example flop_game_solver --release --features "bincode zstd rayon" -- config/20bb.json

use postflop_solver::*;
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ============================================================================
// Config parsing (same format as backend_solver.rs)
// ============================================================================

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
    #[serde(default = "default_add_allin")]
    add_all_in_threshold: f64,
    #[serde(default = "default_force_allin")]
    force_all_in_threshold: f64,
    #[serde(default = "default_merging")]
    merging_threshold: f64,
    #[serde(default = "default_max_raises")]
    max_raises_per_street: i32,
}

fn default_add_allin() -> f64 { 150.0 }
fn default_force_allin() -> f64 { 20.0 }
fn default_merging() -> f64 { 10.0 }
fn default_max_raises() -> i32 { 0 }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverSettings {
    #[serde(default = "default_max_iterations")]
    max_iterations: u32,
    #[serde(default = "default_target_exploitability")]
    target_exploitability_percent: f32,
    #[serde(default)]
    use_compression: bool,
}

fn default_max_iterations() -> u32 { 1000 }
fn default_target_exploitability() -> f32 { 0.5 }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OutputSettings {
    filename: String,
    #[serde(default)]
    compression_level: Option<i32>,
    #[serde(default)]
    memo: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverConfig {
    board: BoardConfig,
    ranges: RangesConfig,
    bet_sizes: BetSizesConfig,
    tree: TreeSettings,
    solver: SolverSettings,
    output: OutputSettings,
}

/// Normalize bet size string: "25, 50" → "25%, 50%", leaves "a" etc. unchanged.
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

/// Parse config into CardConfig and TreeConfig.
fn parse_config(config: &SolverConfig) -> Result<(CardConfig, TreeConfig), String> {
    let oop: Range = config.ranges.oop.parse().map_err(|e| format!("OOP range: {}", e))?;
    let ip: Range = config.ranges.ip.parse().map_err(|e| format!("IP range: {}", e))?;
    let flop = flop_from_str(&config.board.flop).map_err(|e| format!("Flop: {}", e))?;

    let turn = match &config.board.turn {
        Some(t) if !t.is_empty() => card_from_str(t).map_err(|e| format!("Turn: {}", e))?,
        _ => NOT_DEALT,
    };
    let river = match &config.board.river {
        Some(r) if !r.is_empty() => card_from_str(r).map_err(|e| format!("River: {}", e))?,
        _ => NOT_DEALT,
    };

    let card_config = CardConfig { range: [oop, ip], flop, turn, river };

    // Parse bet sizes
    let oop_flop = normalize_bet_sizes(&config.bet_sizes.oop_flop_bet);
    let oop_flop_r = normalize_bet_sizes(&config.bet_sizes.oop_flop_raise);
    let oop_turn = normalize_bet_sizes(&config.bet_sizes.oop_turn_bet);
    let oop_turn_r = normalize_bet_sizes(&config.bet_sizes.oop_turn_raise);
    let oop_river = normalize_bet_sizes(&config.bet_sizes.oop_river_bet);
    let oop_river_r = normalize_bet_sizes(&config.bet_sizes.oop_river_raise);
    let ip_flop = normalize_bet_sizes(&config.bet_sizes.ip_flop_bet);
    let ip_flop_r = normalize_bet_sizes(&config.bet_sizes.ip_flop_raise);
    let ip_turn = normalize_bet_sizes(&config.bet_sizes.ip_turn_bet);
    let ip_turn_r = normalize_bet_sizes(&config.bet_sizes.ip_turn_raise);
    let ip_river = normalize_bet_sizes(&config.bet_sizes.ip_river_bet);
    let ip_river_r = normalize_bet_sizes(&config.bet_sizes.ip_river_raise);

    let flop_bet_sizes = [
        BetSizeOptions::try_from((oop_flop.as_str(), oop_flop_r.as_str()))
            .map_err(|e| format!("OOP flop bet: {}", e))?,
        BetSizeOptions::try_from((ip_flop.as_str(), ip_flop_r.as_str()))
            .map_err(|e| format!("IP flop bet: {}", e))?,
    ];
    let turn_bet_sizes = [
        BetSizeOptions::try_from((oop_turn.as_str(), oop_turn_r.as_str()))
            .map_err(|e| format!("OOP turn bet: {}", e))?,
        BetSizeOptions::try_from((ip_turn.as_str(), ip_turn_r.as_str()))
            .map_err(|e| format!("IP turn bet: {}", e))?,
    ];
    let river_bet_sizes = [
        BetSizeOptions::try_from((oop_river.as_str(), oop_river_r.as_str()))
            .map_err(|e| format!("OOP river bet: {}", e))?,
        BetSizeOptions::try_from((ip_river.as_str(), ip_river_r.as_str()))
            .map_err(|e| format!("IP river bet: {}", e))?,
    ];

    // Donk sizes
    let turn_donk_sizes = if config.tree.donk_option == 1 || config.tree.donk_option == 3 {
        if !config.bet_sizes.oop_turn_donk.is_empty() {
            let s = normalize_bet_sizes(&config.bet_sizes.oop_turn_donk);
            Some(DonkSizeOptions::try_from(s.as_str())
                .map_err(|e| format!("Turn donk: {}", e))?)
        } else { None }
    } else { None };

    let river_donk_sizes = if config.tree.donk_option == 2 || config.tree.donk_option == 3 {
        if !config.bet_sizes.oop_river_donk.is_empty() {
            let s = normalize_bet_sizes(&config.bet_sizes.oop_river_donk);
            Some(DonkSizeOptions::try_from(s.as_str())
                .map_err(|e| format!("River donk: {}", e))?)
        } else { None }
    } else { None };

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: config.tree.starting_pot,
        effective_stack: config.tree.effective_stack,
        rake_rate: config.tree.rake_percent / 100.0,
        rake_cap: config.tree.rake_cap,
        flop_bet_sizes,
        turn_bet_sizes,
        river_bet_sizes,
        turn_donk_sizes,
        river_donk_sizes,
        add_allin_threshold: config.tree.add_all_in_threshold / 100.0,
        force_allin_threshold: config.tree.force_all_in_threshold / 100.0,
        merging_threshold: config.tree.merging_threshold / 100.0,
        max_raises_per_street: config.tree.max_raises_per_street,
    };

    Ok((card_config, tree_config))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }

    let config_path = &args[1];
    if !Path::new(config_path).exists() {
        eprintln!("Error: config not found: {}", config_path);
        std::process::exit(1);
    }

    let config_content = fs::read_to_string(config_path).expect("Failed to read config");
    let config: SolverConfig = serde_json::from_str(&config_content).expect("Failed to parse JSON");

    println!("=== FlopGame Solver ===");
    println!("Config: {}", config_path);
    println!("Board: {}", config.board.flop);
    println!("Pot: {}, Stack: {}", config.tree.starting_pot, config.tree.effective_stack);
    println!("Iterations: {}, Target: {}%", config.solver.max_iterations, config.solver.target_exploitability_percent);
    println!();

    let (card_config, tree_config) = parse_config(&config).expect("Failed to parse config");
    let target_exploitability = tree_config.starting_pot as f32
        * config.solver.target_exploitability_percent / 100.0;

    let total_start = Instant::now();

    // ========================================================================
    // Phase 1: Build and solve FlopGame + TurnOracle
    // ========================================================================
    println!("--- Phase 1: FlopGame + TurnOracle ---");

    let flop_start = Instant::now();

    // Build FlopGame
    let action_tree = ActionTree::new(tree_config.clone()).expect("Failed to create action tree");
    let mut flop_game = FlopGame::new(card_config.clone(), action_tree)
        .expect("Failed to create FlopGame");

    println!("FlopGame nodes: {}", flop_game.num_nodes());
    println!("Boundary amounts: {:?}", flop_game.boundary_amounts());
    println!("OOP hands: {}, IP hands: {}",
        flop_game.num_private_hands(0), flop_game.num_private_hands(1));

    // Build TurnOracle (pre-solves all turn subtrees)
    println!("\nBuilding TurnOracle (solving turn subtrees per boundary)...");
    let oracle_start = Instant::now();
    let oracle = TurnOracle::new(
        &flop_game,
        config.solver.max_iterations,
        target_exploitability,
    );
    let oracle_time = oracle_start.elapsed();
    println!("TurnOracle built in {:.2}s", oracle_time.as_secs_f64());

    // Set oracle and solve flop
    flop_game.set_oracle(Box::new(oracle));

    println!("\nSolving FlopGame...");
    let flop_solve_start = Instant::now();
    let flop_exploitability = solve_flop_nn(
        &mut flop_game,
        config.solver.max_iterations,
        target_exploitability,
        true,
    );
    let flop_solve_time = flop_solve_start.elapsed();
    let flop_total_time = flop_start.elapsed();

    let flop_exploit_pct = flop_exploitability / tree_config.starting_pot as f32 * 100.0;
    println!("FlopGame exploitability: {:.4} ({:.3}% of pot)", flop_exploitability, flop_exploit_pct);
    println!("FlopGame time: {:.2}s (oracle: {:.2}s, solve: {:.2}s)",
        flop_total_time.as_secs_f64(), oracle_time.as_secs_f64(), flop_solve_time.as_secs_f64());

    // ========================================================================
    // Phase 2: Build PostFlopGame, lock flop strategies, solve turn/river
    // ========================================================================
    println!("\n--- Phase 2: PostFlopGame with locked flop strategies ---");

    let phase2_start = Instant::now();
    let action_tree = ActionTree::new(tree_config.clone()).expect("Failed to create action tree");
    let mut postflop_game = PostFlopGame::with_config(card_config, action_tree)
        .expect("Failed to create PostFlopGame");

    let (mem_uncompressed, mem_compressed) = postflop_game.memory_usage();
    println!("Memory: {:.1} MB (uncompressed), {:.1} MB (compressed)",
        mem_uncompressed as f64 / 1048576.0,
        mem_compressed as f64 / 1048576.0);

    postflop_game.allocate_memory(config.solver.use_compression);

    // Transfer flop strategies from FlopGame via node locking
    println!("Locking flop strategies into PostFlopGame...");
    flop_game.lock_strategies_in(&mut postflop_game);

    // Finalize without solving — flop strategies are locked, turn/river get uniform defaults
    println!("Finalizing...");
    finalize(&mut postflop_game);
    let phase2_time = phase2_start.elapsed();
    println!("Phase 2 time: {:.2}s", phase2_time.as_secs_f64());

    // ========================================================================
    // Phase 3: Save .flop file
    // ========================================================================
    println!("\n--- Phase 3: Save .flop ---");

    let output_path = &config.output.filename;
    let memo = config.output.memo.clone().unwrap_or_else(|| {
        format!("FlopGame+Oracle {} pot={} stack={} flop_exploit={:.3}%",
            config.board.flop, config.tree.starting_pot,
            config.tree.effective_stack, flop_exploit_pct)
    });

    // Ensure output directory exists
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent).ok();
    }

    match save_data_to_file(&postflop_game, &memo, output_path, config.output.compression_level) {
        Ok(()) => {
            let file_size = fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
            println!("Saved: {} ({:.1} MB)", output_path, file_size as f64 / 1048576.0);
        }
        Err(e) => {
            eprintln!("Failed to save: {}", e);
            std::process::exit(1);
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    let total_time = total_start.elapsed();
    println!("\n=== Summary ===");
    println!("FlopGame: {} nodes, {:.2}s (oracle: {:.2}s, solve: {:.2}s)",
        flop_game.num_nodes(), flop_total_time.as_secs_f64(),
        oracle_time.as_secs_f64(), flop_solve_time.as_secs_f64());
    println!("PostFlopGame: lock + finalize {:.2}s", phase2_time.as_secs_f64());
    println!("Total time: {:.2}s", total_time.as_secs_f64());
    println!("Output: {}", output_path);
}
