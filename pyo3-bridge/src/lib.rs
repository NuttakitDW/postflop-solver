#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use ::postflop_solver::*;
use serde::Deserialize;

// ─── Config structs (matching examples/common/mod.rs JSON format) ───

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
    #[serde(default)]
    target_exploitability_percent: f32,
    #[serde(default)]
    use_compression: bool,
}

fn default_max_iterations() -> u32 { 1000 }

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

// ─── Config parsing ───

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
        initial_state: BoardState::Flop,
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

fn build_game(card_config: &CardConfig, tree_config: &TreeConfig) -> Result<PostFlopGame, String> {
    let action_tree = ActionTree::new(tree_config.clone())?;
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree)?;
    game.allocate_memory(false);
    Ok(game)
}

// ─── GameWrapper pyclass ───

#[pyclass]
struct GameWrapper {
    game: PostFlopGame,
    card_config: CardConfig,
    tree_config: TreeConfig,
    cached_num_boundaries: usize,
    max_iterations: u32,
}

#[pymethods]
impl GameWrapper {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read config: {}", e)))?;
        let config: SolverConfig = serde_json::from_str(&content)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to parse config: {}", e)))?;
        let max_iterations = config.solver.max_iterations;
        let (card_config, tree_config) = parse_configs(&config)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let game = build_game(&card_config, &tree_config)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        // Count boundaries by traversing the fresh game
        let cached_num_boundaries = collect_boundary_cfreaches(&game, 0).len();

        Ok(GameWrapper {
            game,
            card_config,
            tree_config,
            cached_num_boundaries,
            max_iterations,
        })
    }

    /// Reset the game state (zeros all regrets and strategies).
    fn reset(&mut self) -> PyResult<()> {
        self.game = build_game(&self.card_config, &self.tree_config)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(())
    }

    /// Number of private hands for a player (0=OOP, 1=IP).
    fn num_private_hands(&self, player: usize) -> usize {
        self.game.num_private_hands(player)
    }

    /// Number of turn boundary nodes (same for both players).
    fn num_boundaries(&self) -> usize {
        self.cached_num_boundaries
    }

    /// Maximum iterations from config.
    fn max_iterations(&self) -> u32 {
        self.max_iterations
    }

    /// Starting pot from config.
    fn starting_pot(&self) -> i32 {
        self.tree_config.starting_pot
    }

    /// Collect opponent cfreach at each turn boundary (DFS order).
    /// Returns list of cfreach vectors, one per boundary.
    fn collect_boundary_cfreaches(&self, player: usize) -> Vec<Vec<f32>> {
        collect_boundary_cfreaches(&self.game, player)
    }

    /// Run one DCFR iteration for a player with model-predicted boundary CFVs.
    ///
    /// At boundaries: turn/river subtrees traversed normally, true CFVs recorded,
    /// but model_cfvs used for flop parent regret update.
    ///
    /// Returns the true boundary CFVs (training labels) in DFS order.
    fn solve_step_with_model(
        &self,
        iteration: u32,
        player: usize,
        model_cfvs: Vec<Vec<f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let result = solve_step_with_model(&self.game, iteration, player, &model_cfvs);
        Ok(result)
    }

    /// Run one DCFR iteration for a player replaying pre-recorded boundary CFVs.
    /// No turn/river subtree traversal — just uses the provided CFVs directly.
    fn solve_step_replay(
        &self,
        iteration: u32,
        player: usize,
        cfvs: Vec<Vec<f32>>,
    ) -> PyResult<()> {
        solve_step_for_player_replay(&self.game, iteration, player, &cfvs);
        Ok(())
    }

    /// Run one normal DCFR iteration for a player, recording boundary CFVs.
    /// Full tree traversal with correct regret updates everywhere.
    /// Returns the true boundary CFVs in DFS order.
    fn solve_step_recording(
        &self,
        iteration: u32,
        player: usize,
    ) -> Vec<Vec<f32>> {
        solve_step_for_player_recording(&self.game, iteration, player)
    }

    /// Compute exploitability of the current strategy (in chips).
    /// Works on in-progress games without finalize.
    fn compute_exploitability(&self) -> f32 {
        compute_exploitability(&self.game)
    }

    /// Finalize the game (normalize strategies for output).
    /// Must be called before save_to_file. Cannot solve further after this.
    fn finalize(&mut self) -> PyResult<()> {
        finalize(&mut self.game);
        Ok(())
    }

    /// Save the solved game to a .flop file.
    /// Must call finalize() first.
    fn save_to_file(&self, path: &str, memo: &str) -> PyResult<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        save_data_to_file(&self.game, memo, path, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save: {}", e)))?;
        Ok(())
    }
}

// ─── Python module ───

#[pymodule]
fn postflop_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameWrapper>()?;
    Ok(())
}
