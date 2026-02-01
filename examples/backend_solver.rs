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
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;

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
    /// Export flop-only: solve full tree but only save flop strategies
    /// This gives accurate GTO strategies with 99% storage reduction
    #[serde(default)]
    export_flop_only: bool,
    /// Path to regret snapshot file to use for warm-starting
    /// If provided, loads precomputed regrets before solving
    #[serde(default)]
    warm_start_from: Option<String>,
    /// Save regret snapshot after solving (for future warm starts)
    #[serde(default)]
    save_regret_snapshot: Option<String>,
    /// Extract EV map from solved full tree and save to this path
    /// The EV map captures hand EVs at flop terminals including future street value
    #[serde(default)]
    extract_ev_map: Option<String>,
    /// Load EV map for flop-only solving with precomputed leaf values
    /// Uses Range-Agnostic transfer for different ranges/bet sizes
    #[serde(default)]
    load_ev_map: Option<String>,
}

fn default_max_iterations() -> u32 { 1000 }
fn default_target_exploitability() -> f32 { 0.5 }

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
            export_flop_only: false,
            warm_start_from: None,
            save_regret_snapshot: None,
            extract_ev_map: None,
            load_ev_map: None,
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
        error: Some(error),
    }
}

/// Converts a card pair to a string like "AhKs"
fn hole_to_string(hole: (u8, u8)) -> String {
    let card_to_str = |card: u8| -> String {
        let rank = card >> 2;
        let suit = card & 3;
        let rank_char = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'][rank as usize];
        let suit_char = ['c', 'd', 'h', 's'][suit as usize];
        format!("{}{}", rank_char, suit_char)
    };
    // Put higher card first
    let max_card = u8::max(hole.0, hole.1);
    let min_card = u8::min(hole.0, hole.1);
    format!("{}{}", card_to_str(max_card), card_to_str(min_card))
}

/// Groups hands by canonical form (e.g., "AKs", "AKo", "AA")
fn group_hands_by_canonical(private_cards: &[(u8, u8)]) -> HashMap<String, Vec<(usize, String)>> {
    let mut hand_groups: HashMap<String, Vec<(usize, String)>> = HashMap::new();
    for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
        let r1 = c1 >> 2;
        let r2 = c2 >> 2;
        let s1 = c1 & 3;
        let s2 = c2 & 3;
        let rank_chars = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
        let high_rank = rank_chars[u8::max(r1, r2) as usize];
        let low_rank = rank_chars[u8::min(r1, r2) as usize];
        let suited = if r1 == r2 { "" } else if s1 == s2 { "s" } else { "o" };
        let canonical = format!("{}{}{}", high_rank, low_rank, suited);
        let full_name = hole_to_string((c1, c2));
        hand_groups.entry(canonical).or_default().push((hand_idx, full_name));
    }
    hand_groups
}

/// Prints strategy at current node
fn print_node_strategy(game: &PostFlopGame, node_path: &str, num_hands_to_show: usize) {
    if game.is_terminal_node() {
        println!("\n[{}] TERMINAL NODE", node_path);
        return;
    }
    if game.is_chance_node() {
        println!("\n[{}] CHANCE NODE (card dealt)", node_path);
        return;
    }

    let actions = game.available_actions();
    let player = game.current_player();
    let private_cards = game.private_cards(player).to_vec();
    let num_hands = private_cards.len();
    let strategy = game.strategy();

    let position = if player == 0 { "OOP" } else { "IP" };
    println!("\n╔══════════════════════════════════════════════════════════════");
    println!("║ [{}] {} to act ({} hands)", node_path, position, num_hands);
    println!("║ Actions: {:?}", actions);
    println!("╚══════════════════════════════════════════════════════════════");

    // Calculate aggregate strategy
    let mut action_totals: Vec<f64> = vec![0.0; actions.len()];
    for action_idx in 0..actions.len() {
        for hand_idx in 0..num_hands {
            action_totals[action_idx] += strategy[action_idx * num_hands + hand_idx] as f64;
        }
    }
    let total_weight: f64 = action_totals.iter().sum();

    println!("\nAggregate Strategy:");
    for (i, action) in actions.iter().enumerate() {
        let freq = if total_weight > 0.0 { action_totals[i] / total_weight * 100.0 } else { 0.0 };
        println!("  {:?}: {:.1}%", action, freq);
    }

    // Group hands
    let hand_groups = group_hands_by_canonical(&private_cards);

    // For each action, show top hands
    println!("\nStrategy by Action:");
    for (action_idx, action) in actions.iter().enumerate() {
        println!("\n  --- {:?} ---", action);

        let mut hand_freqs: Vec<(String, f64)> = Vec::new();
        for (canonical, combos) in &hand_groups {
            let total_freq: f64 = combos.iter()
                .map(|(h_idx, _)| strategy[action_idx * num_hands + h_idx] as f64)
                .sum();
            let avg_freq = total_freq / combos.len() as f64;
            if avg_freq > 0.05 { // Only show if > 5%
                hand_freqs.push((canonical.clone(), avg_freq));
            }
        }

        hand_freqs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let show_count = num_hands_to_show.min(hand_freqs.len());
        for (canonical, freq) in hand_freqs.iter().take(show_count) {
            println!("    {}: {:.0}%", canonical, freq * 100.0);
        }
        if hand_freqs.len() > show_count {
            println!("    ... and {} more hands", hand_freqs.len() - show_count);
        }
    }

    // Key hands detailed
    println!("\n  Key Hands:");
    let key_hands = ["AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "KQs", "QJs", "JTs", "T9s", "98s", "87s", "76s", "65s"];
    for key in key_hands.iter() {
        if let Some(combos) = hand_groups.get(*key) {
            let mut avg_strategy: Vec<f64> = vec![0.0; actions.len()];
            for (hand_idx, _) in combos {
                for action_idx in 0..actions.len() {
                    avg_strategy[action_idx] += strategy[action_idx * num_hands + hand_idx] as f64;
                }
            }
            for s in avg_strategy.iter_mut() {
                *s /= combos.len() as f64;
            }

            let strat_str: Vec<String> = actions.iter().zip(avg_strategy.iter())
                .filter(|(_, &freq)| freq > 0.01)
                .map(|(a, freq)| format!("{:?}:{:.0}%", a, freq * 100.0))
                .collect();
            if !strat_str.is_empty() {
                println!("    {}: {}", key, strat_str.join(", "));
            }
        }
    }
}

/// Prints strategy results for flop-only mode to console, traversing the tree
fn print_flop_strategy(game: &mut PostFlopGame, num_hands_to_show: usize) {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║              FLOP-ONLY STRATEGY RESULTS                            ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");

    // ==========================================
    // ROOT: OOP's first action
    // ==========================================
    game.back_to_root();
    print_node_strategy(game, "Root: OOP", num_hands_to_show);

    let root_actions = game.available_actions();

    // Find check action index
    let check_idx = root_actions.iter().position(|a| matches!(a, Action::Check));

    if let Some(check_idx) = check_idx {
        // ==========================================
        // OOP checks -> IP's turn
        // ==========================================
        game.back_to_root();
        game.play(check_idx);

        if !game.is_terminal_node() && !game.is_chance_node() {
            print_node_strategy(game, "OOP Check → IP", num_hands_to_show);

            let ip_actions = game.available_actions();

            // Find IP's bet actions
            for (ip_action_idx, ip_action) in ip_actions.iter().enumerate() {
                match ip_action {
                    Action::Check => {
                        // OOP check -> IP check = end of flop action (goes to turn)
                        game.back_to_root();
                        game.play(check_idx);
                        game.play(ip_action_idx);
                        if game.is_chance_node() {
                            println!("\n[OOP Check → IP Check] → CHANCE NODE (deal turn)");
                        }
                    }
                    Action::Bet(size) => {
                        // ==========================================
                        // OOP check -> IP bet -> OOP's response
                        // ==========================================
                        game.back_to_root();
                        game.play(check_idx);
                        game.play(ip_action_idx);

                        if !game.is_terminal_node() && !game.is_chance_node() {
                            let path = format!("OOP Check → IP Bet({}) → OOP", size);
                            print_node_strategy(game, &path, num_hands_to_show);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Find bet action index at root
    let bet_idx = root_actions.iter().position(|a| matches!(a, Action::Bet(_)));

    if let Some(bet_idx) = bet_idx {
        // ==========================================
        // OOP bets -> IP's response
        // ==========================================
        game.back_to_root();
        game.play(bet_idx);

        if !game.is_terminal_node() && !game.is_chance_node() {
            let bet_size = match root_actions[bet_idx] {
                Action::Bet(s) => s,
                _ => 0,
            };
            let path = format!("OOP Bet({}) → IP", bet_size);
            print_node_strategy(game, &path, num_hands_to_show);
        }
    }

    game.back_to_root();
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("                         END STRATEGY RESULTS");
    println!("═══════════════════════════════════════════════════════════════════════\n");
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

    // Allocate memory (always full tree - we solve everything then export what we need)
    game.allocate_memory(config.solver.use_compression);

    // Log export mode
    if config.solver.export_flop_only {
        if initial_state != BoardState::Flop {
            return create_error_result("Export flop-only requires board to be at flop (no turn/river specified)".to_string());
        }
        println!("Export mode: FLOP ONLY (solve full tree, save only flop strategies)");

        // Enable flop-only solving mode
        game.set_solve_flop_only(true);
    }

    // Load EV map for flop-only solving if provided
    if let Some(ref ev_map_path) = config.solver.load_ev_map {
        println!("Loading EV map from: {}", ev_map_path);
        let load_start = Instant::now();
        match EvMap::load_from_file(ev_map_path) {
            Ok(ev_map) => {
                println!("  Loaded {} flop terminals in {:.2}s",
                    ev_map.terminals.len(), load_start.elapsed().as_secs_f64());
                game.set_ev_map(Some(Arc::new(ev_map)));
                println!("  EV map enabled - using precomputed leaf values!");
            }
            Err(e) => {
                println!("  Warning: Failed to load EV map: {}", e);
                println!("  Continuing with equity-based evaluation...");
            }
        }
    }

    // Warm start from precomputed regrets if provided
    if let Some(ref snapshot_path) = config.solver.warm_start_from {
        println!("Loading regret snapshot from: {}", snapshot_path);
        let load_start = Instant::now();
        match RegretSnapshot::load_from_file(snapshot_path) {
            Ok(snapshot) => {
                match snapshot.apply_to_game(&mut game) {
                    Ok(applied) => {
                        println!("  Applied regrets to {} nodes in {:.2}s",
                            applied, load_start.elapsed().as_secs_f64());
                        println!("  Warm start enabled - solver will converge faster!");
                    }
                    Err(e) => {
                        println!("  Warning: Failed to apply regrets: {}", e);
                        println!("  Continuing without warm start...");
                    }
                }
            }
            Err(e) => {
                println!("  Warning: Failed to load snapshot: {}", e);
                println!("  Continuing without warm start...");
            }
        }
    }

    // Calculate target exploitability
    let target_exploitability =
        game.tree_config().starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;

    // Solve
    let solve_start = Instant::now();
    let exploitability = solve(
        &mut game,
        config.solver.max_iterations,
        target_exploitability,
        true, // Print progress
    );
    let solve_time = solve_start.elapsed();

    let exploitability_percent = exploitability / game.tree_config().starting_pot as f32 * 100.0;

    // Save regret snapshot if requested (for future warm starts)
    if let Some(ref snapshot_path) = config.solver.save_regret_snapshot {
        println!("Saving regret snapshot to: {}", snapshot_path);
        let save_start = Instant::now();
        match RegretSnapshot::from_game(&game) {
            Ok(snapshot) => {
                // Ensure parent directory exists
                if let Some(parent) = std::path::Path::new(snapshot_path).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match snapshot.save_to_file(snapshot_path) {
                    Ok(_) => {
                        println!("  Saved {} node regrets in {:.2}s",
                            snapshot.node_regrets.len(), save_start.elapsed().as_secs_f64());
                        // Get file size
                        if let Ok(metadata) = std::fs::metadata(snapshot_path) {
                            println!("  Snapshot file size: {:.2} MB",
                                metadata.len() as f64 / 1024.0 / 1024.0);
                        }
                    }
                    Err(e) => println!("  Warning: Failed to save snapshot: {}", e),
                }
            }
            Err(e) => println!("  Warning: Failed to create snapshot: {}", e),
        }
    }

    // Extract and save EV map if requested (for future flop-only solves)
    if let Some(ref ev_map_path) = config.solver.extract_ev_map {
        println!("Extracting EV map to: {}", ev_map_path);
        let extract_start = Instant::now();
        match EvMap::from_solved_game(&game) {
            Ok(ev_map) => {
                // Ensure parent directory exists
                if let Some(parent) = std::path::Path::new(ev_map_path).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match ev_map.save_to_file(ev_map_path) {
                    Ok(_) => {
                        println!("  Extracted {} flop terminals in {:.2}s",
                            ev_map.terminals.len(), extract_start.elapsed().as_secs_f64());
                        // Get file size
                        if let Ok(metadata) = std::fs::metadata(ev_map_path) {
                            println!("  EV map file size: {:.2} MB",
                                metadata.len() as f64 / 1024.0 / 1024.0);
                        }
                    }
                    Err(e) => println!("  Warning: Failed to save EV map: {}", e),
                }
            }
            Err(e) => println!("  Warning: Failed to extract EV map: {}", e),
        }
    }

    // Generate memo
    let memo = config.output.memo.clone().unwrap_or_else(|| {
        format!(
            "{} {} {}, pot={}, stack={}, exploitability={:.4} ({:.3}%)",
            config.board.flop,
            config.board.turn.as_deref().unwrap_or("-"),
            config.board.river.as_deref().unwrap_or("-"),
            config.tree.starting_pot,
            config.tree.effective_stack,
            exploitability,
            exploitability_percent
        )
    });

    // Set target storage mode for export
    if config.solver.export_flop_only {
        if let Err(e) = game.set_target_storage_mode(BoardState::Flop) {
            return create_error_result(format!("Failed to set flop-only export mode: {}", e));
        }
        println!("Exporting flop-only strategies...");
        let full_memory = game.memory_usage();
        let flop_memory = game.target_memory_usage();
        println!("  Full tree memory: {:.2} MB", full_memory.0 as f64 / 1024.0 / 1024.0);
        println!("  Flop-only memory: {:.2} MB", flop_memory as f64 / 1024.0 / 1024.0);
        println!("  Reduction: {:.1}%", (1.0 - flop_memory as f64 / full_memory.0 as f64) * 100.0);

        // Print strategy results to console
        print_flop_strategy(&mut game, 15);
    }

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
        error: None,
    }
}

fn main() {
    // Initialize logger for debug output
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .format_timestamp(None)
        .init();

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
        println!("Solve time: {:.2}s", result.solve_time_seconds);
        println!("Total time: {:.2}s", result.total_time_seconds);
        println!("Exploitability: {:.4} ({:.3}% of pot)",
            result.final_exploitability, result.exploitability_percent);
        println!("Memory: {:.2} MB", result.memory_mb);
        println!("OOP hands: {}, IP hands: {}", result.oop_hands, result.ip_hands);
    } else {
        eprintln!();
        eprintln!("Error: {}", result.error.unwrap_or_default());
        std::process::exit(1);
    }
}
