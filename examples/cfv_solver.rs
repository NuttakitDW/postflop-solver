//! CFV Solver: Compares BoundaryCfv (MatrixTurnCfv) against full-game ExactTurnCfv.
//!
//! 1. Solves a standard PostFlopGame (full tree) → saves to .flop
//! 2. For each boundary (amount, turn_card): builds ExactTurnCfv (full game),
//!    extracts MatrixTurnCfv from it, and compares evaluate() outputs.
//!
//! Usage:
//!   cargo run --example cfv_solver --release --features "bincode zstd rayon" -- config/20bb.json

#[cfg(feature = "jemalloc")]
use tikv_jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

// =============================================================================
// Config structs (same as backend_solver)
// =============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct BoardConfig {
    flop: String,
    #[serde(default)]
    turn: Option<String>,
    #[serde(default)]
    river: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RangesConfig {
    oop: String,
    ip: String,
}

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OutputSettings {
    filename: String,
    #[serde(default)]
    compression_level: Option<i32>,
    #[serde(default)]
    memo: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverConfig {
    board: BoardConfig,
    ranges: RangesConfig,
    bet_sizes: BetSizesConfig,
    tree: TreeSettings,
    solver: SolverSettings,
    output: OutputSettings,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SolverResult {
    success: bool,
    standard_solve_seconds: f64,
    standard_exploitability: f32,
    standard_exploitability_percent: f32,
    boundary_amounts: Vec<i32>,
    turn_games_compared: usize,
    oracle_comparison_seconds: f64,
    matrix_max_diff: f32,
    matrix_avg_diff: f64,
    total_matrix_mb: f64,
    total_time_seconds: f64,
    error: Option<String>,
}

// =============================================================================
// Helpers
// =============================================================================

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

fn parse_configs(config: &SolverConfig) -> Result<(CardConfig, TreeConfig), String> {
    let oop: Range = config.ranges.oop.parse()
        .map_err(|e| format!("Failed to parse OOP range: {}", e))?;
    let ip: Range = config.ranges.ip.parse()
        .map_err(|e| format!("Failed to parse IP range: {}", e))?;
    let flop = flop_from_str(&config.board.flop)
        .map_err(|e| format!("Failed to parse flop: {}", e))?;

    let turn = match &config.board.turn {
        Some(t) if !t.is_empty() => card_from_str(t)
            .map_err(|e| format!("Failed to parse turn: {}", e))?,
        _ => NOT_DEALT,
    };
    let river = match &config.board.river {
        Some(r) if !r.is_empty() => card_from_str(r)
            .map_err(|e| format!("Failed to parse river: {}", e))?,
        _ => NOT_DEALT,
    };

    let initial_state = if river != NOT_DEALT {
        BoardState::River
    } else if turn != NOT_DEALT {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    let card_config = CardConfig {
        range: [oop, ip],
        flop,
        turn,
        river,
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

    let oop_flop = BetSizeOptions::try_from((oop_flop_bet.as_str(), oop_flop_raise.as_str()))
        .map_err(|e| format!("Failed to parse OOP flop bet sizes: {}", e))?;
    let oop_turn = BetSizeOptions::try_from((oop_turn_bet.as_str(), oop_turn_raise.as_str()))
        .map_err(|e| format!("Failed to parse OOP turn bet sizes: {}", e))?;
    let oop_river = BetSizeOptions::try_from((oop_river_bet.as_str(), oop_river_raise.as_str()))
        .map_err(|e| format!("Failed to parse OOP river bet sizes: {}", e))?;
    let ip_flop = BetSizeOptions::try_from((ip_flop_bet.as_str(), ip_flop_raise.as_str()))
        .map_err(|e| format!("Failed to parse IP flop bet sizes: {}", e))?;
    let ip_turn = BetSizeOptions::try_from((ip_turn_bet.as_str(), ip_turn_raise.as_str()))
        .map_err(|e| format!("Failed to parse IP turn bet sizes: {}", e))?;
    let ip_river = BetSizeOptions::try_from((ip_river_bet.as_str(), ip_river_raise.as_str()))
        .map_err(|e| format!("Failed to parse IP river bet sizes: {}", e))?;

    let turn_donk_sizes = if config.tree.donk_option == 1 || config.tree.donk_option == 3 {
        if !config.bet_sizes.oop_turn_donk.is_empty() {
            let s = normalize_bet_sizes(&config.bet_sizes.oop_turn_donk);
            Some(DonkSizeOptions::try_from(s.as_str())
                .map_err(|e| format!("Failed to parse turn donk sizes: {}", e))?)
        } else { None }
    } else { None };

    let river_donk_sizes = if config.tree.donk_option == 2 || config.tree.donk_option == 3 {
        if !config.bet_sizes.oop_river_donk.is_empty() {
            let s = normalize_bet_sizes(&config.bet_sizes.oop_river_donk);
            Some(DonkSizeOptions::try_from(s.as_str())
                .map_err(|e| format!("Failed to parse river donk sizes: {}", e))?)
        } else { None }
    } else { None };

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.tree.starting_pot,
        effective_stack: config.tree.effective_stack,
        rake_rate: config.tree.rake_percent / 100.0,
        rake_cap: config.tree.rake_cap,
        flop_bet_sizes: [oop_flop, ip_flop],
        turn_bet_sizes: [oop_turn, ip_turn],
        river_bet_sizes: [oop_river, ip_river],
        turn_donk_sizes,
        river_donk_sizes,
        add_allin_threshold: config.tree.add_all_in_threshold / 100.0,
        force_allin_threshold: config.tree.force_all_in_threshold / 100.0,
        merging_threshold: config.tree.merging_threshold / 100.0,
        max_raises_per_street: config.tree.max_raises_per_street,
    };

    Ok((card_config, tree_config))
}

// =============================================================================
// Solver
// =============================================================================

fn create_error_result(error: String) -> SolverResult {
    SolverResult {
        success: false, standard_solve_seconds: 0.0,
        standard_exploitability: 0.0, standard_exploitability_percent: 0.0,
        boundary_amounts: Vec::new(), turn_games_compared: 0,
        oracle_comparison_seconds: 0.0, matrix_max_diff: 0.0,
        matrix_avg_diff: 0.0, total_matrix_mb: 0.0,
        total_time_seconds: 0.0, error: Some(error),
    }
}

fn run_solver(config: &SolverConfig) -> SolverResult {
    let total_start = Instant::now();

    let (card_config, tree_config) = match parse_configs(config) {
        Ok(c) => c,
        Err(e) => return create_error_result(e),
    };

    let target_exploitability =
        tree_config.starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;

    // =========================================================================
    // [A] Standard full-tree solve → .flop
    // =========================================================================
    println!("--- Standard Full-Tree Solve ---");
    let std_start = Instant::now();
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let boundary_amounts = action_tree.boundary_amounts();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory(false);
    let exploitability = solve(&mut game, config.solver.max_iterations, target_exploitability, true);
    let std_time = std_start.elapsed().as_secs_f64();
    let exploitability_percent = exploitability / tree_config.starting_pot as f32 * 100.0;

    println!("  Exploitability: {:.4} ({:.3}% of pot)", exploitability, exploitability_percent);
    println!("  Solve time: {:.2}s", std_time);

    // Save .flop
    let output_path = &config.output.filename;
    if let Some(parent) = Path::new(output_path).parent() {
        fs::create_dir_all(parent).ok();
    }
    let memo = config.output.memo.as_deref().unwrap_or("cfv_solver");
    save_data_to_file(&game, memo, output_path, config.output.compression_level)
        .expect("Failed to save .flop file");
    println!("  Saved to {}", output_path);

    // Root CFVs
    println!();
    println!("--- Standard Root CFVs ---");
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

    // Free standard game memory
    drop(game);

    // =========================================================================
    // [B] BoundaryCfv comparison: ExactTurnCfv (full game) vs MatrixTurnCfv
    // =========================================================================
    println!();
    println!("--- BoundaryCfv Comparison: ExactTurnCfv vs MatrixTurnCfv ---");
    println!("  Boundary amounts: {:?}", boundary_amounts);

    let flop = card_config.flop;
    let flop_mask: u64 = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);
    let num_turn_cards = 52 - 3; // 49

    let bet_config = TurnBetConfig {
        turn_bet_sizes: tree_config.turn_bet_sizes.clone(),
        river_bet_sizes: tree_config.river_bet_sizes.clone(),
        add_allin_threshold: tree_config.add_allin_threshold,
        force_allin_threshold: tree_config.force_allin_threshold,
        merging_threshold: tree_config.merging_threshold,
    };

    let total_games = boundary_amounts.len() * num_turn_cards;
    let mut games_done = 0usize;
    let mut total_matrix_bytes = 0usize;
    let mut global_max_diff = 0.0f32;
    let mut global_total_diff = 0.0f64;
    let mut global_comparisons = 0usize;

    let oracle_start = Instant::now();

    for &amount in &boundary_amounts {
        let pot = tree_config.starting_pot + 2 * amount;
        let stack = tree_config.effective_stack - amount;
        let actual_stack = if stack > 0 { stack } else { 1 };

        let mut amount_max_diff = 0.0f32;

        for card in 0u8..52 {
            if flop_mask & (1u64 << card) != 0 {
                continue;
            }

            // Build ExactTurnCfv (full game — keeps tree)
            let exact = ExactTurnCfv::new(
                flop, card,
                &card_config.range[0], &card_config.range[1],
                pot, actual_stack, &bet_config,
                config.solver.max_iterations, target_exploitability,
            ).unwrap();

            // Extract MatrixTurnCfv from the same solved game (no double-solving)
            let matrix = MatrixTurnCfv::from_exact(&exact);
            total_matrix_bytes += matrix.matrix_memory_bytes();

            // Compare for both players with initial reaches
            for player in 0..2 {
                let cfreach = exact.initial_weights(player ^ 1).to_vec();
                let exact_cfvs = exact.evaluate(player, &cfreach);
                let matrix_cfvs = matrix.evaluate(player, &cfreach);

                for (a, b) in exact_cfvs.iter().zip(&matrix_cfvs) {
                    let diff = (a - b).abs();
                    global_max_diff = global_max_diff.max(diff);
                    amount_max_diff = amount_max_diff.max(diff);
                    global_total_diff += diff as f64;
                    global_comparisons += 1;
                }
            }

            games_done += 1;
            if games_done % 10 == 0 || games_done == total_games {
                eprint!("\r  Turn games: {}/{}", games_done, total_games);
            }
        }

        eprintln!();
        println!("  Amount {}: max_diff={:.8}", amount, amount_max_diff);
    }

    let oracle_time = oracle_start.elapsed().as_secs_f64();
    let global_avg_diff = if global_comparisons > 0 {
        global_total_diff / global_comparisons as f64
    } else {
        0.0
    };
    let matrix_mb = total_matrix_bytes as f64 / 1024.0 / 1024.0;

    println!();
    println!("  Overall: max_diff={:.8}, avg_diff={:.8} ({} hand comparisons)",
        global_max_diff, global_avg_diff, global_comparisons);
    println!("  Matrix memory: {:.2} MB", matrix_mb);
    println!("  Comparison time: {:.2}s ({} turn games)", oracle_time, total_games);

    SolverResult {
        success: true,
        standard_solve_seconds: std_time,
        standard_exploitability: exploitability,
        standard_exploitability_percent: exploitability_percent,
        boundary_amounts: boundary_amounts.clone(),
        turn_games_compared: total_games,
        oracle_comparison_seconds: oracle_time,
        matrix_max_diff: global_max_diff,
        matrix_avg_diff: global_avg_diff,
        total_matrix_mb: matrix_mb,
        total_time_seconds: total_start.elapsed().as_secs_f64(),
        error: None,
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .format_timestamp(None)
        .init();

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

    let config_content = fs::read_to_string(config_path)
        .expect("Failed to read config file");
    let config: SolverConfig = serde_json::from_str(&config_content)
        .expect("Failed to parse config JSON");

    println!("=== CFV Solver ===");
    println!("Config: {}", config_path);
    println!("Board: {} {} {}",
        config.board.flop,
        config.board.turn.as_deref().unwrap_or("-"),
        config.board.river.as_deref().unwrap_or("-")
    );
    println!("Starting pot: {}", config.tree.starting_pot);
    println!("Effective stack: {}", config.tree.effective_stack);
    println!("Max iterations: {}", config.solver.max_iterations);
    println!("Target exploitability: {}% of pot", config.solver.target_exploitability_percent);
    println!();

    let result = run_solver(&config);

    let result_json = serde_json::to_string_pretty(&result).expect("Failed to serialize result");
    println!();
    println!("=== Result ===");
    println!("{}", result_json);

    if result.success {
        println!();
        println!("=== Summary ===");
        println!("Standard solve: {:.2}s (exploitability {:.3}%)",
            result.standard_solve_seconds, result.standard_exploitability_percent);
        println!("Oracle comparison: {:.2}s ({} turn games, {:.2} MB matrices)",
            result.oracle_comparison_seconds, result.turn_games_compared, result.total_matrix_mb);
        println!("Matrix vs Exact: max_diff={:.8}, avg_diff={:.8}",
            result.matrix_max_diff, result.matrix_avg_diff);
        println!("Total: {:.2}s", result.total_time_seconds);
    } else {
        eprintln!();
        eprintln!("Error: {}", result.error.unwrap_or_default());
        std::process::exit(1);
    }
}
