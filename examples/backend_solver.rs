//! Backend solver that reads configuration from JSON and outputs a .flop file for UI import.
//!
//! Run with: cargo run --example backend_solver --release --features "bincode zstd"
//!
//! Input: JSON configuration file (matches desktop-postflop UI format)
//! Output: .flop file that can be loaded in desktop-postflop
//!
//! Usage:
//!   cargo run --example backend_solver --release --features "bincode zstd" -- config/50bb.json
//!   cargo run --example backend_solver --release --features "bincode zstd" -- --generate-template
//!
//! The JSON format matches the desktop-postflop configurations.json format.

#[cfg(feature = "jemalloc")]
use tikv_jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use postflop_solver::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Board configuration
#[derive(Debug, Serialize, Deserialize)]
struct BoardConfig {
    flop: String,
    #[serde(default)]
    turn: Option<String>,
    #[serde(default)]
    river: Option<String>,
}

/// Ranges configuration
#[derive(Debug, Serialize, Deserialize)]
struct RangesConfig {
    oop: String,
    ip: String,
}

/// Bet sizes configuration - matches UI format with separate OOP/IP settings
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BetSizesConfig {
    // OOP bet sizes
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
    // IP bet sizes
    ip_flop_bet: String,
    ip_flop_raise: String,
    ip_turn_bet: String,
    ip_turn_raise: String,
    ip_river_bet: String,
    ip_river_raise: String,
}

/// Tree configuration - matches UI format
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TreeSettings {
    starting_pot: i32,
    effective_stack: i32,
    #[serde(default)]
    rake_percent: f64,
    #[serde(default)]
    rake_cap: f64,
    /// 0 = no donk, 1 = turn only, 2 = river only, 3 = turn and river
    #[serde(default)]
    donk_option: u8,
    /// Threshold as percentage (e.g., 150 = 1.5x)
    #[serde(default = "default_add_allin")]
    add_all_in_threshold: f64,
    /// Threshold as percentage (e.g., 20 = 0.2x)
    #[serde(default = "default_force_allin")]
    force_all_in_threshold: f64,
    /// Threshold as percentage (e.g., 10 = 0.1x)
    #[serde(default = "default_merging")]
    merging_threshold: f64,
    /// Maximum raises per street (0 = unlimited, 5 = GTO Wizard default)
    #[serde(default = "default_max_raises")]
    max_raises_per_street: i32,
}

fn default_add_allin() -> f64 { 150.0 }
fn default_force_allin() -> f64 { 20.0 }
fn default_merging() -> f64 { 10.0 }
fn default_max_raises() -> i32 { 0 }

/// Solver mode enum
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
enum SolverModeConfig {
    /// Full precision solving (standard DCFR)
    #[default]
    Full,
    /// Blueprint mode: solve with card abstraction
    Blueprint,
    /// Subgame mode: solve blueprint + refine subgames
    Subgame,
}

/// Subgame-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SubgameSettings {
    /// Number of turn buckets for card abstraction
    #[serde(default = "default_turn_buckets")]
    turn_buckets: u8,
    /// Number of river buckets for card abstraction
    #[serde(default = "default_river_buckets")]
    river_buckets: u8,
    /// Number of iterations for blueprint solving
    #[serde(default = "default_blueprint_iterations")]
    blueprint_iterations: u32,
    /// Target exploitability for blueprint (percent of pot)
    #[serde(default = "default_blueprint_exploitability")]
    blueprint_exploitability_percent: f32,
    /// Number of iterations for subgame solving
    #[serde(default = "default_subgame_iterations")]
    subgame_iterations: u32,
    /// Target exploitability for subgames (percent of pot)
    #[serde(default = "default_subgame_exploitability")]
    subgame_exploitability_percent: f32,
    /// Whether to use safe subgame solving
    #[serde(default)]
    safe_solving: bool,
}

fn default_turn_buckets() -> u8 { 10 }
fn default_river_buckets() -> u8 { 10 }
fn default_blueprint_iterations() -> u32 { 500 }
fn default_blueprint_exploitability() -> f32 { 2.0 }
fn default_subgame_iterations() -> u32 { 1000 }
fn default_subgame_exploitability() -> f32 { 0.5 }

impl Default for SubgameSettings {
    fn default() -> Self {
        Self {
            turn_buckets: default_turn_buckets(),
            river_buckets: default_river_buckets(),
            blueprint_iterations: default_blueprint_iterations(),
            blueprint_exploitability_percent: default_blueprint_exploitability(),
            subgame_iterations: default_subgame_iterations(),
            subgame_exploitability_percent: default_subgame_exploitability(),
            safe_solving: false,
        }
    }
}

/// Solver settings
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SolverSettings {
    #[serde(default = "default_max_iterations")]
    max_iterations: u32,
    #[serde(default = "default_target_exploitability")]
    target_exploitability_percent: f32,
    #[serde(default)]
    use_compression: bool,
    /// Solver mode: "full", "blueprint", or "subgame"
    #[serde(default)]
    mode: SolverModeConfig,
    /// Subgame-specific settings (only used when mode is "blueprint" or "subgame")
    #[serde(default)]
    subgame: Option<SubgameSettings>,
    /// Random seed for reproducibility
    #[serde(default)]
    seed: Option<u64>,
    /// Whether to sample chance nodes
    #[serde(default = "default_sample_chance")]
    sample_chance: bool,
    /// Whether to use confidence interval for convergence
    #[serde(default = "default_use_ci")]
    use_ci: bool,
    /// Target confidence interval
    #[serde(default = "default_target_ci")]
    target_ci: f32,
}

fn default_max_iterations() -> u32 { 1000 }
fn default_target_exploitability() -> f32 { 0.5 }
fn default_sample_chance() -> bool { true }
fn default_use_ci() -> bool { true }
fn default_target_ci() -> f32 { 5.0 }

/// Output settings
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct OutputSettings {
    filename: String,
    #[serde(default)]
    compression_level: Option<i32>,
    #[serde(default)]
    memo: Option<String>,
}

/// Main configuration structure
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

/// Result structure returned after solving
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SolverResult {
    success: bool,
    output_file: String,
    solve_time_seconds: f64,
    total_time_seconds: f64,
    final_exploitability: f32,
    exploitability_percent: f32,
    memory_mb: f64,
    iterations_used: u32,
    oop_hands: usize,
    ip_hands: usize,
    /// Solver mode used
    mode: String,
    /// Number of subgames solved (if subgame mode)
    subgames_solved: Option<u32>,
    /// Blueprint exploitability (if subgame mode)
    blueprint_exploitability: Option<f32>,
    error: Option<String>,
}

fn generate_template() -> SolverConfig {
    SolverConfig {
        board: BoardConfig {
            flop: "Td9d6h".to_string(),
            turn: None,
            river: None,
        },
        ranges: RangesConfig {
            oop: "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo".to_string(),
            ip: "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo".to_string(),
        },
        bet_sizes: BetSizesConfig {
            // OOP bet sizes
            oop_flop_bet: "33, a".to_string(),
            oop_flop_raise: "33, 55, a".to_string(),
            oop_turn_bet: "20, 33, 55, 83, 125, 200, a".to_string(),
            oop_turn_raise: "33, 55, a".to_string(),
            oop_turn_donk: "".to_string(),
            oop_river_bet: "11, 35, 60, 85, 149, a".to_string(),
            oop_river_raise: "33, 55, a".to_string(),
            oop_river_donk: "".to_string(),
            // IP bet sizes
            ip_flop_bet: "20, 33, 55, 83, 125, a".to_string(),
            ip_flop_raise: "33, 55, a".to_string(),
            ip_turn_bet: "20, 33, 55, 83, 125, 200, a".to_string(),
            ip_turn_raise: "33, 55, a".to_string(),
            ip_river_bet: "11, 35, 60, 85, 149, a".to_string(),
            ip_river_raise: "33, 55, a".to_string(),
        },
        tree: TreeSettings {
            starting_pot: 55,
            effective_stack: 180,
            rake_percent: 0.0,
            rake_cap: 0.0,
            donk_option: 0,
            add_all_in_threshold: 150.0,
            force_all_in_threshold: 20.0,
            merging_threshold: 10.0,
            max_raises_per_street: 0, // 0 = unlimited, 5 = GTO Wizard default
        },
        solver: SolverSettings {
            max_iterations: 1000,
            target_exploitability_percent: 0.5,
            use_compression: false,
            mode: SolverModeConfig::Full,
            subgame: Some(SubgameSettings {
                turn_buckets: 10,
                river_buckets: 10,
                blueprint_iterations: 500,
                blueprint_exploitability_percent: 2.0,
                subgame_iterations: 1000,
                subgame_exploitability_percent: 0.5,
                safe_solving: false,
            }),
            seed: None,
            sample_chance: true,
            use_ci: true,
            target_ci: 5.0,
        },
        output: OutputSettings {
            filename: "solution.flop".to_string(),
            compression_level: Some(3),
            memo: Some("Generated by backend_solver".to_string()),
        },
    }
}

/// Normalize bet size string to add % suffix where needed.
/// Converts "25, 50, 100" to "25%, 50%, 100%" but leaves "a" (all-in) and
/// other special formats (like "2x", "100c", "2e") unchanged.
fn normalize_bet_sizes(s: &str) -> String {
    s.split(',')
        .map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                return trimmed.to_string();
            }
            // Check if it's already a special format or has % suffix
            let lower = trimmed.to_lowercase();
            if lower == "a"
                || lower.ends_with('%')
                || lower.ends_with('x')
                || lower.contains('c')
                || lower.contains('e')
            {
                trimmed.to_string()
            } else if trimmed.parse::<f64>().is_ok() {
                // It's a plain number, add %
                format!("{}%", trimmed)
            } else {
                // Unknown format, pass through
                trimmed.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn create_error_result(error: String) -> SolverResult {
    SolverResult {
        success: false,
        output_file: String::new(),
        solve_time_seconds: 0.0,
        total_time_seconds: 0.0,
        final_exploitability: 0.0,
        exploitability_percent: 0.0,
        memory_mb: 0.0,
        iterations_used: 0,
        oop_hands: 0,
        ip_hands: 0,
        mode: "full".to_string(),
        subgames_solved: None,
        blueprint_exploitability: None,
        error: Some(error),
    }
}

fn run_solver(config: &SolverConfig) -> SolverResult {
    let total_start = Instant::now();

    // Parse ranges
    let oop: Range = match config.ranges.oop.parse() {
        Ok(r) => r,
        Err(e) => return create_error_result(format!("Failed to parse OOP range: {}", e)),
    };

    let ip: Range = match config.ranges.ip.parse() {
        Ok(r) => r,
        Err(e) => return create_error_result(format!("Failed to parse IP range: {}", e)),
    };

    // Parse flop
    let flop = match flop_from_str(&config.board.flop) {
        Ok(f) => f,
        Err(e) => return create_error_result(format!("Failed to parse flop: {}", e)),
    };

    // Parse turn if provided
    let turn = match &config.board.turn {
        Some(t) if !t.is_empty() => match card_from_str(t) {
            Ok(c) => c,
            Err(e) => return create_error_result(format!("Failed to parse turn: {}", e)),
        },
        _ => NOT_DEALT,
    };

    // Parse river if provided
    let river = match &config.board.river {
        Some(r) if !r.is_empty() => match card_from_str(r) {
            Ok(c) => c,
            Err(e) => return create_error_result(format!("Failed to parse river: {}", e)),
        },
        _ => NOT_DEALT,
    };

    // Determine initial state
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

    // Normalize bet sizes (add % where needed)
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

    // Parse bet sizes - OOP (index 0)
    let oop_flop_bet_sizes = match BetSizeOptions::try_from((
        oop_flop_bet.as_str(),
        oop_flop_raise.as_str(),
    )) {
        Ok(b) => b,
        Err(e) => return create_error_result(format!("Failed to parse OOP flop bet sizes: {}", e)),
    };

    let oop_turn_bet_sizes = match BetSizeOptions::try_from((
        oop_turn_bet.as_str(),
        oop_turn_raise.as_str(),
    )) {
        Ok(b) => b,
        Err(e) => return create_error_result(format!("Failed to parse OOP turn bet sizes: {}", e)),
    };

    let oop_river_bet_sizes = match BetSizeOptions::try_from((
        oop_river_bet.as_str(),
        oop_river_raise.as_str(),
    )) {
        Ok(b) => b,
        Err(e) => return create_error_result(format!("Failed to parse OOP river bet sizes: {}", e)),
    };

    // Parse bet sizes - IP (index 1)
    let ip_flop_bet_sizes = match BetSizeOptions::try_from((
        ip_flop_bet.as_str(),
        ip_flop_raise.as_str(),
    )) {
        Ok(b) => b,
        Err(e) => return create_error_result(format!("Failed to parse IP flop bet sizes: {}", e)),
    };

    let ip_turn_bet_sizes = match BetSizeOptions::try_from((
        ip_turn_bet.as_str(),
        ip_turn_raise.as_str(),
    )) {
        Ok(b) => b,
        Err(e) => return create_error_result(format!("Failed to parse IP turn bet sizes: {}", e)),
    };

    let ip_river_bet_sizes = match BetSizeOptions::try_from((
        ip_river_bet.as_str(),
        ip_river_raise.as_str(),
    )) {
        Ok(b) => b,
        Err(e) => return create_error_result(format!("Failed to parse IP river bet sizes: {}", e)),
    };

    // Parse donk sizes if enabled
    let turn_donk_sizes = if config.tree.donk_option == 1 || config.tree.donk_option == 3 {
        if !config.bet_sizes.oop_turn_donk.is_empty() {
            let oop_turn_donk = normalize_bet_sizes(&config.bet_sizes.oop_turn_donk);
            match DonkSizeOptions::try_from(oop_turn_donk.as_str()) {
                Ok(d) => Some(d),
                Err(e) => return create_error_result(format!("Failed to parse turn donk sizes: {}", e)),
            }
        } else {
            None
        }
    } else {
        None
    };

    let river_donk_sizes = if config.tree.donk_option == 2 || config.tree.donk_option == 3 {
        if !config.bet_sizes.oop_river_donk.is_empty() {
            let oop_river_donk = normalize_bet_sizes(&config.bet_sizes.oop_river_donk);
            match DonkSizeOptions::try_from(oop_river_donk.as_str()) {
                Ok(d) => Some(d),
                Err(e) => return create_error_result(format!("Failed to parse river donk sizes: {}", e)),
            }
        } else {
            None
        }
    } else {
        None
    };

    // Convert threshold percentages to decimals (UI uses 150 for 1.5x, etc.)
    let add_allin_threshold = config.tree.add_all_in_threshold / 100.0;
    let force_allin_threshold = config.tree.force_all_in_threshold / 100.0;
    let merging_threshold = config.tree.merging_threshold / 100.0;

    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.tree.starting_pot,
        effective_stack: config.tree.effective_stack,
        rake_rate: config.tree.rake_percent / 100.0, // Convert percentage to rate
        rake_cap: config.tree.rake_cap,
        flop_bet_sizes: [oop_flop_bet_sizes, ip_flop_bet_sizes],
        turn_bet_sizes: [oop_turn_bet_sizes, ip_turn_bet_sizes],
        river_bet_sizes: [oop_river_bet_sizes, ip_river_bet_sizes],
        turn_donk_sizes,
        river_donk_sizes,
        add_allin_threshold,
        force_allin_threshold,
        merging_threshold,
        max_raises_per_street: config.tree.max_raises_per_street,
    };

    // Build action tree
    let action_tree = match ActionTree::new(tree_config) {
        Ok(t) => t,
        Err(e) => return create_error_result(format!("Failed to create action tree: {}", e)),
    };

    // Create game
    let mut game = match PostFlopGame::with_config(card_config, action_tree) {
        Ok(g) => g,
        Err(e) => return create_error_result(format!("Failed to create game: {}", e)),
    };

    let oop_hands = game.private_cards(0).len();
    let ip_hands = game.private_cards(1).len();

    let (mem_uncompressed, _) = game.memory_usage();
    let memory_mb = mem_uncompressed as f64 / 1024.0 / 1024.0;

    // Allocate memory
    game.allocate_memory(config.solver.use_compression);

    // Solve based on mode
    let solve_start = Instant::now();
    let mode_str = match config.solver.mode {
        SolverModeConfig::Full => "full",
        SolverModeConfig::Blueprint => "blueprint",
        SolverModeConfig::Subgame => "subgame",
    };

    let (exploitability, subgames_solved, blueprint_exploitability) = match config.solver.mode {
        SolverModeConfig::Full => {
            // Standard full-precision solving
            let target_exploitability =
                game.tree_config().starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;

            let exploitability = solve(
                &mut game,
                config.solver.max_iterations,
                target_exploitability,
                true,
            );
            (exploitability, None, None)
        }

        SolverModeConfig::Blueprint | SolverModeConfig::Subgame => {
            // Get subgame settings
            let subgame_settings = config.solver.subgame.as_ref()
                .cloned()
                .unwrap_or_default();

            #[cfg(feature = "subgame")]
            {
                use postflop_solver::subgame::{
                    SolverMode, SubgameSolverConfig, solve_with_subgames
                };

                let subgame_config = SubgameSolverConfig {
                    mode: match config.solver.mode {
                        SolverModeConfig::Blueprint => SolverMode::Blueprint,
                        SolverModeConfig::Subgame => SolverMode::Subgame,
                        _ => SolverMode::Full,
                    },
                    turn_buckets: subgame_settings.turn_buckets,
                    river_buckets: subgame_settings.river_buckets,
                    blueprint_iterations: subgame_settings.blueprint_iterations,
                    blueprint_target_exploitability: subgame_settings.blueprint_exploitability_percent / 100.0,
                    subgame_iterations: subgame_settings.subgame_iterations,
                    subgame_target_exploitability: subgame_settings.subgame_exploitability_percent / 100.0,
                    safe_solving: subgame_settings.safe_solving,
                    parallel: true,
                    print_progress: true,
                };

                match solve_with_subgames(&mut game, &subgame_config) {
                    Ok(result) => {
                        let exploitability = result.final_exploitability * game.tree_config().starting_pot as f32;
                        let subgames = if result.subgames_solved > 0 {
                            Some(result.subgames_solved)
                        } else {
                            None
                        };
                        let blueprint_exp = result.blueprint.as_ref().map(|b| b.exploitability * 100.0);
                        (exploitability, subgames, blueprint_exp)
                    }
                    Err(e) => {
                        return create_error_result(format!("Subgame solving failed: {}", e));
                    }
                }
            }

            #[cfg(not(feature = "subgame"))]
            {
                // Fallback to full solving if subgame feature is not enabled
                println!("Warning: subgame feature not enabled, falling back to full solving");
                let target_exploitability =
                    game.tree_config().starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;

                let exploitability = solve(
                    &mut game,
                    config.solver.max_iterations,
                    target_exploitability,
                    true,
                );
                (exploitability, None, None)
            }
        }
    };

    let solve_time = solve_start.elapsed();
    let exploitability_percent = exploitability / game.tree_config().starting_pot as f32 * 100.0;

    // Generate memo
    let memo = config.output.memo.clone().unwrap_or_else(|| {
        let mode_info = match config.solver.mode {
            SolverModeConfig::Full => "full".to_string(),
            SolverModeConfig::Blueprint => format!("blueprint ({}x{} buckets)",
                config.solver.subgame.as_ref().map_or(10, |s| s.turn_buckets),
                config.solver.subgame.as_ref().map_or(10, |s| s.river_buckets)),
            SolverModeConfig::Subgame => format!("subgame ({} solved)",
                subgames_solved.unwrap_or(0)),
        };
        format!(
            "{} {} {}, pot={}, stack={}, mode={}, exploitability={:.4} ({:.3}%)",
            config.board.flop,
            config.board.turn.as_deref().unwrap_or("-"),
            config.board.river.as_deref().unwrap_or("-"),
            config.tree.starting_pot,
            config.tree.effective_stack,
            mode_info,
            exploitability,
            exploitability_percent
        )
    });

    // Save to file
    if let Err(e) = save_data_to_file(&game, &memo, &config.output.filename, config.output.compression_level) {
        return SolverResult {
            success: false,
            output_file: String::new(),
            solve_time_seconds: solve_time.as_secs_f64(),
            total_time_seconds: total_start.elapsed().as_secs_f64(),
            final_exploitability: exploitability,
            exploitability_percent,
            memory_mb,
            iterations_used: config.solver.max_iterations,
            oop_hands,
            ip_hands,
            mode: mode_str.to_string(),
            subgames_solved,
            blueprint_exploitability,
            error: Some(format!("Failed to save file: {}", e)),
        };
    }

    SolverResult {
        success: true,
        output_file: config.output.filename.clone(),
        solve_time_seconds: solve_time.as_secs_f64(),
        total_time_seconds: total_start.elapsed().as_secs_f64(),
        final_exploitability: exploitability,
        exploitability_percent,
        memory_mb,
        iterations_used: config.solver.max_iterations,
        oop_hands,
        ip_hands,
        mode: mode_str.to_string(),
        subgames_solved,
        blueprint_exploitability,
        error: None,
    }
}

fn main() {
    // Initialize logger with debug level as default
    #[cfg(feature = "logging")]
    {
        use std::io::Write;
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
            .format(|buf, record| writeln!(buf, "[{}] {}", record.level(), record.args()))
            .init();
    }

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        eprintln!("       {} --generate-template", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  <config.json>       Path to JSON configuration file");
        eprintln!("  --generate-template Generate a template config file (config/template.json)");
        std::process::exit(1);
    }

    if args[1] == "--generate-template" {
        let template = generate_template();
        let json = serde_json::to_string_pretty(&template).expect("Failed to serialize template");
        fs::write("config/template.json", &json).expect("Failed to write template file");
        println!("Template written to config/template.json");
        println!();
        println!("{}", json);
        return;
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

    println!("=== Backend Solver ===");
    println!("Config: {}", config_path);
    println!("Threads: {}", rayon::current_num_threads());
    println!();
    println!("Board: {} {} {}",
        config.board.flop,
        config.board.turn.as_deref().unwrap_or("-"),
        config.board.river.as_deref().unwrap_or("-")
    );
    println!("Starting pot: {}", config.tree.starting_pot);
    println!("Effective stack: {}", config.tree.effective_stack);
    println!("Max iterations: {}", config.solver.max_iterations);
    println!("Target exploitability: {}% of pot", config.solver.target_exploitability_percent);
    println!("Solver mode: {:?}", config.solver.mode);
    if config.solver.mode != SolverModeConfig::Full {
        if let Some(ref subgame) = config.solver.subgame {
            println!("  Turn buckets: {}", subgame.turn_buckets);
            println!("  River buckets: {}", subgame.river_buckets);
            println!("  Blueprint iterations: {}", subgame.blueprint_iterations);
            println!("  Subgame iterations: {}", subgame.subgame_iterations);
        }
    }
    println!("Output: {}", config.output.filename);
    println!();

    let result = run_solver(&config);

    // Output result as JSON for programmatic use
    let result_json = serde_json::to_string_pretty(&result).expect("Failed to serialize result");

    println!("=== Result ===");
    println!("{}", result_json);

    // Also print human-readable summary
    if result.success {
        println!();
        println!("=== Summary ===");
        println!("Output file: {}", result.output_file);
        println!("Solver mode: {}", result.mode);
        println!("Solve time: {:.2}s", result.solve_time_seconds);
        println!("Total time: {:.2}s", result.total_time_seconds);
        println!("Exploitability: {:.4} ({:.3}% of pot)",
            result.final_exploitability, result.exploitability_percent);
        if let Some(blueprint_exp) = result.blueprint_exploitability {
            println!("Blueprint exploitability: {:.3}% of pot", blueprint_exp);
        }
        if let Some(subgames) = result.subgames_solved {
            println!("Subgames solved: {}", subgames);
        }
        println!("Memory: {:.2} MB", result.memory_mb);
        println!("OOP hands: {}, IP hands: {}", result.oop_hands, result.ip_hands);
    } else {
        eprintln!();
        eprintln!("Error: {}", result.error.unwrap_or_default());
        std::process::exit(1);
    }
}
