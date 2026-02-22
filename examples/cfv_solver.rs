//! CFV Solver: Config-driven deepstack solver using FlopGame + ExactTurnCfv oracle.
//!
//! Reads JSON config (same format as backend_solver), builds a flop-only game tree,
//! pre-solves turn subtrees with ExactTurnCfv, then solves the flop using the oracle.
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
use std::collections::HashMap;
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
    solve_time_seconds: f64,
    total_time_seconds: f64,
    exploitability: f32,
    exploitability_percent: f32,
    oop_hands: usize,
    ip_hands: usize,
    flop_nodes: usize,
    boundary_amounts: Vec<i32>,
    turn_games_solved: usize,
    oracle_build_seconds: f64,
    error: Option<String>,
}

// =============================================================================
// CfvOracle: BoundaryCfv using ExactTurnCfv
// =============================================================================

struct CfvOracle {
    evaluators_by_amount: HashMap<i32, Vec<Option<ExactTurnCfv>>>,
    flop_to_turn: Vec<Option<[Vec<usize>; 2]>>,
    total_turn_cards: usize,
}

impl CfvOracle {
    fn new(game: &FlopGame, max_iterations: u32, target_exploitability: f32) -> Self {
        let card_config = game.card_config();
        let tree_config = game.tree_config();
        let flop = card_config.flop;
        let flop_mask: u64 = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);

        let bet_config = TurnBetConfig {
            turn_bet_sizes: tree_config.turn_bet_sizes.clone(),
            river_bet_sizes: tree_config.river_bet_sizes.clone(),
            add_allin_threshold: tree_config.add_allin_threshold,
            force_allin_threshold: tree_config.force_allin_threshold,
            merging_threshold: tree_config.merging_threshold,
        };

        // Build flop→turn hand index mappings
        let mut flop_to_turn: Vec<Option<[Vec<usize>; 2]>> = Vec::with_capacity(52);
        let mut total_turn_cards = 0usize;

        for card in 0u8..52 {
            if flop_mask & (1u64 << card) != 0 {
                flop_to_turn.push(None);
                continue;
            }
            total_turn_cards += 1;

            let mut mappings: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            for player in 0..2 {
                let flop_hands = game.private_cards(player);
                let mut turn_idx = 0usize;
                for &(c1, c2) in flop_hands {
                    if c1 == card || c2 == card {
                        mappings[player].push(usize::MAX);
                    } else {
                        mappings[player].push(turn_idx);
                        turn_idx += 1;
                    }
                }
            }
            flop_to_turn.push(Some(mappings));
        }

        // Pre-solve turn games for each (boundary_amount, turn_card)
        let amounts = game.boundary_amounts();
        let mut evaluators_by_amount = HashMap::new();
        let total_games = amounts.len() * total_turn_cards;
        let mut games_done = 0usize;

        for &amount in &amounts {
            let pot = tree_config.starting_pot + 2 * amount;
            let stack = tree_config.effective_stack - amount;

            let mut card_evals: Vec<Option<ExactTurnCfv>> = (0..52).map(|_| None).collect();

            for card in 0u8..52 {
                if flop_mask & (1u64 << card) != 0 {
                    continue;
                }

                let actual_stack = if stack > 0 { stack } else { 1 };
                let eval = ExactTurnCfv::new(
                    flop,
                    card,
                    &card_config.range[0],
                    &card_config.range[1],
                    pot,
                    actual_stack,
                    &bet_config,
                    max_iterations,
                    target_exploitability,
                )
                .unwrap();

                card_evals[card as usize] = Some(eval);
                games_done += 1;

                if games_done % 10 == 0 || games_done == total_games {
                    eprint!("\r  Oracle: {}/{} turn games solved", games_done, total_games);
                }
            }

            evaluators_by_amount.insert(amount, card_evals);
        }
        eprintln!();

        Self {
            evaluators_by_amount,
            flop_to_turn,
            total_turn_cards,
        }
    }
}

impl BoundaryCfv for CfvOracle {
    fn compute(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        result.iter_mut().for_each(|r| { r.write(0.0); });
        let result_f32 = unsafe { &mut *(result as *mut [MaybeUninit<f32>] as *mut [f32]) };

        let card_evals = match self.evaluators_by_amount.get(&node.amount()) {
            Some(evals) => evals,
            None => return,
        };

        let opponent = player ^ 1;

        for card in 0u8..52 {
            let mapping = match &self.flop_to_turn[card as usize] {
                Some(m) => m,
                None => continue,
            };

            let eval = match &card_evals[card as usize] {
                Some(e) => e,
                None => continue,
            };

            let num_turn_hands_opp = eval.num_private_hands(opponent);
            let mut turn_cfreach = vec![0.0f32; num_turn_hands_opp];
            for (flop_idx, &turn_idx) in mapping[opponent].iter().enumerate() {
                if turn_idx != usize::MAX {
                    turn_cfreach[turn_idx] = cfreach[flop_idx];
                }
            }

            let turn_cfvs = eval.evaluate(player, &turn_cfreach);

            for (flop_idx, &turn_idx) in mapping[player].iter().enumerate() {
                if turn_idx != usize::MAX {
                    result_f32[flop_idx] += turn_cfvs[turn_idx];
                }
            }
        }

        let scale = 1.0 / self.total_turn_cards as f32;
        for v in result_f32.iter_mut() {
            *v *= scale;
        }
    }
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

    // Normalize bet sizes
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
        success: false, solve_time_seconds: 0.0, total_time_seconds: 0.0,
        exploitability: 0.0, exploitability_percent: 0.0,
        oop_hands: 0, ip_hands: 0, flop_nodes: 0,
        boundary_amounts: Vec::new(), turn_games_solved: 0,
        oracle_build_seconds: 0.0, error: Some(error),
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

    // Build FlopGame
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = FlopGame::new(card_config, action_tree).unwrap();

    let oop_hands = game.num_private_hands(0);
    let ip_hands = game.num_private_hands(1);
    let flop_nodes = game.num_nodes();
    let boundary_amounts = game.boundary_amounts();
    let num_turn_cards = 49;
    let turn_games_total = boundary_amounts.len() * num_turn_cards;

    println!("  Flop tree nodes: {}", flop_nodes);
    println!("  OOP hands: {}  IP hands: {}", oop_hands, ip_hands);
    println!("  Boundary amounts: {:?}", boundary_amounts);
    println!("  Turn games to solve: {} amounts x {} cards = {}",
        boundary_amounts.len(), num_turn_cards, turn_games_total);
    println!();

    // Build CfvOracle with ExactTurnCfv
    let oracle_start = Instant::now();
    let oracle = CfvOracle::new(&game, config.solver.max_iterations, target_exploitability);
    let oracle_time = oracle_start.elapsed().as_secs_f64();
    println!("  Oracle build time: {:.2}s", oracle_time);

    game.set_oracle(Box::new(oracle));

    // Solve flop game
    let solve_start = Instant::now();
    let exploitability = solve_flop_nn(
        &mut game,
        config.solver.max_iterations,
        target_exploitability,
        true,
    );
    let solve_time = solve_start.elapsed().as_secs_f64();
    let exploitability_percent = exploitability / tree_config.starting_pot as f32 * 100.0;

    println!("  Exploitability: {:.4} ({:.3}% of pot)", exploitability, exploitability_percent);
    println!("  Flop solve time: {:.2}s", solve_time);

    // Extract and display root CFVs
    println!();
    println!("--- Root CFVs ---");
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

    SolverResult {
        success: true,
        solve_time_seconds: solve_time,
        total_time_seconds: total_start.elapsed().as_secs_f64(),
        exploitability,
        exploitability_percent,
        oop_hands,
        ip_hands,
        flop_nodes,
        boundary_amounts: boundary_amounts.clone(),
        turn_games_solved: turn_games_total,
        oracle_build_seconds: oracle_time,
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

    println!("=== CFV Solver (Deepstack) ===");
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
        println!("Oracle build: {:.2}s ({} turn games)", result.oracle_build_seconds, result.turn_games_solved);
        println!("Flop solve: {:.2}s", result.solve_time_seconds);
        println!("Total: {:.2}s", result.total_time_seconds);
        println!("Exploitability: {:.4} ({:.3}% of pot)", result.exploitability, result.exploitability_percent);
    } else {
        eprintln!();
        eprintln!("Error: {}", result.error.unwrap_or_default());
        std::process::exit(1);
    }
}
