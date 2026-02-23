//! Shared config parsing for examples.

use postflop_solver::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct BoardConfig {
    pub flop: String,
    #[serde(default)]
    pub turn: Option<String>,
    #[serde(default)]
    pub river: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RangesConfig {
    pub oop: String,
    pub ip: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BetSizesConfig {
    pub oop_flop_bet: String,
    pub oop_flop_raise: String,
    pub oop_turn_bet: String,
    pub oop_turn_raise: String,
    #[serde(default)]
    pub oop_turn_donk: String,
    pub oop_river_bet: String,
    pub oop_river_raise: String,
    #[serde(default)]
    pub oop_river_donk: String,
    pub ip_flop_bet: String,
    pub ip_flop_raise: String,
    pub ip_turn_bet: String,
    pub ip_turn_raise: String,
    pub ip_river_bet: String,
    pub ip_river_raise: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeSettings {
    pub starting_pot: i32,
    pub effective_stack: i32,
    #[serde(default)]
    pub rake_percent: f64,
    #[serde(default)]
    pub rake_cap: f64,
    #[serde(default)]
    pub donk_option: u8,
    #[serde(default = "default_add_allin")]
    pub add_all_in_threshold: f64,
    #[serde(default = "default_force_allin")]
    pub force_all_in_threshold: f64,
    #[serde(default = "default_merging")]
    pub merging_threshold: f64,
    #[serde(default = "default_max_raises")]
    pub max_raises_per_street: i32,
}

fn default_add_allin() -> f64 { 150.0 }
fn default_force_allin() -> f64 { 20.0 }
fn default_merging() -> f64 { 10.0 }
fn default_max_raises() -> i32 { 0 }

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolverSettings {
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
    #[serde(default = "default_target_exploitability")]
    pub target_exploitability_percent: f32,
    #[serde(default)]
    pub use_compression: bool,
}

fn default_max_iterations() -> u32 { 1000 }
fn default_target_exploitability() -> f32 { 0.5 }

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutputSettings {
    pub filename: String,
    #[serde(default)]
    pub compression_level: Option<i32>,
    #[serde(default)]
    pub memo: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolverConfig {
    pub board: BoardConfig,
    pub ranges: RangesConfig,
    pub bet_sizes: BetSizesConfig,
    pub tree: TreeSettings,
    pub solver: SolverSettings,
    pub output: OutputSettings,
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

pub fn parse_configs(config: &SolverConfig) -> Result<(CardConfig, TreeConfig), String> {
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

pub fn load_config(path: &str) -> SolverConfig {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read config file '{}': {}", path, e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse config JSON '{}': {}", path, e))
}
