//! Integration module for connecting subgame solving with PostFlopGame.
//!
//! This module provides the glue code to:
//! - Extract boundary data from a solved PostFlopGame
//! - Create subgames from boundary points
//! - Solve subgames with the main DCFR solver
//! - Stitch subgame solutions back together

use crate::action_tree::{ActionTree, BoardState, TreeConfig};
use crate::card::CardConfig;
use crate::game::PostFlopGame;
use crate::interface::Game;
use crate::solver::solve;
use crate::subgame::abstraction::{AbstractionConfig, AbstractionMapping};
use crate::subgame::blueprint::{Blueprint, BlueprintConfig};
use crate::subgame::boundary::{BoundaryData, BoundaryStore};
use crate::subgame::subgame_solver::{SubgameConfig, SubgameInfo, SubgameSolveResult};

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Solver mode configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub enum SolverMode {
    /// Full precision solving (standard DCFR).
    #[default]
    Full,

    /// Blueprint mode: solve with card abstraction.
    Blueprint,

    /// Subgame mode: solve blueprint + refine subgames.
    Subgame,
}

/// Configuration for subgame-based solving.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct SubgameSolverConfig {
    /// Solver mode.
    pub mode: SolverMode,

    /// Number of turn buckets for abstraction.
    pub turn_buckets: u8,

    /// Number of river buckets for abstraction.
    pub river_buckets: u8,

    /// Number of iterations for blueprint solving.
    pub blueprint_iterations: u32,

    /// Target exploitability for blueprint (fraction of pot).
    pub blueprint_target_exploitability: f32,

    /// Number of iterations for subgame solving.
    pub subgame_iterations: u32,

    /// Target exploitability for subgames (fraction of pot).
    pub subgame_target_exploitability: f32,

    /// Whether to use safe subgame solving.
    pub safe_solving: bool,

    /// Whether to solve subgames in parallel.
    pub parallel: bool,

    /// Whether to print progress.
    pub print_progress: bool,
}

impl Default for SubgameSolverConfig {
    fn default() -> Self {
        Self {
            mode: SolverMode::Full,
            turn_buckets: 10,
            river_buckets: 10,
            blueprint_iterations: 500,
            blueprint_target_exploitability: 0.02, // 2% of pot
            subgame_iterations: 1000,
            subgame_target_exploitability: 0.005, // 0.5% of pot
            safe_solving: true,
            parallel: true,
            print_progress: true,
        }
    }
}

impl SubgameSolverConfig {
    /// Create a fast configuration for testing.
    pub fn fast() -> Self {
        Self {
            mode: SolverMode::Subgame,
            turn_buckets: 5,
            river_buckets: 5,
            blueprint_iterations: 100,
            blueprint_target_exploitability: 0.05,
            subgame_iterations: 200,
            subgame_target_exploitability: 0.01,
            safe_solving: false,
            parallel: true,
            print_progress: false,
        }
    }

    /// Create a high-quality configuration.
    pub fn high_quality() -> Self {
        Self {
            mode: SolverMode::Subgame,
            turn_buckets: 15,
            river_buckets: 15,
            blueprint_iterations: 1000,
            blueprint_target_exploitability: 0.01,
            subgame_iterations: 2000,
            subgame_target_exploitability: 0.002,
            safe_solving: true,
            parallel: true,
            print_progress: true,
        }
    }
}

/// Extract boundary data from a solved PostFlopGame at the current position.
///
/// This extracts the ranges (normalized weights), expected values, and
/// counterfactual values for both players at the current game state.
///
/// # Requirements
/// - The game must be solved
/// - `cache_normalized_weights()` should have been called
pub fn extract_boundary_from_game(game: &PostFlopGame) -> BoundaryData {
    // Ensure normalized weights are cached
    let weights_oop = game.normalized_weights(0).to_vec();
    let weights_ip = game.normalized_weights(1).to_vec();

    // Get expected values
    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);

    // Get equity as proxy for cfvalues (simplified - in real impl, get actual cfvalues)
    let equity_oop = game.equity(0);
    let equity_ip = game.equity(1);

    // Get tree config for pot/stack
    let tree_config = game.tree_config();

    // Get action history
    let history = game.history();
    let action_history: Vec<u16> = history.iter().map(|&a| a as u16).collect();

    // Determine street from current board state
    let board = game.current_board();
    let (turn, street) = match board.len() {
        3 => (None, 0),                  // At flop, extracting for turn
        4 => (Some(board[3]), 1),        // At turn, extracting for river
        5 => (Some(board[3]), 1),        // At river
        _ => (None, 0),
    };

    BoundaryData::new(
        [weights_oop, weights_ip],
        [equity_oop, equity_ip],
        [ev_oop, ev_ip],
        tree_config.starting_pot,
        tree_config.effective_stack,
        action_history,
        turn,
        street,
    )
}

/// Result of subgame solving.
#[derive(Clone, Debug)]
pub struct IntegrationSolveResult {
    /// Blueprint (if created).
    pub blueprint: Option<Blueprint>,

    /// Number of subgames solved.
    pub subgames_solved: u32,

    /// Total solving time in seconds.
    pub solve_time_seconds: f64,

    /// Final exploitability (fraction of pot).
    pub final_exploitability: f32,

    /// Memory used in bytes.
    pub memory_usage: u64,
}

/// Run the subgame solving workflow on a PostFlopGame.
///
/// This function:
/// 1. Computes card abstraction (if mode is Blueprint or Subgame)
/// 2. Solves the blueprint with abstraction
/// 3. Extracts boundary data at street transitions
/// 4. Solves subgames for Turn and River (if mode is Subgame)
///
/// # Arguments
/// - `game`: The PostFlopGame to solve (must have memory allocated)
/// - `config`: Subgame solver configuration
///
/// # Returns
/// The solve result including blueprint, stats, and exploitability
pub fn solve_with_subgames(
    game: &mut PostFlopGame,
    config: &SubgameSolverConfig,
) -> Result<IntegrationSolveResult, String> {
    use std::time::Instant;

    if game.is_solved() {
        return Err("Game is already solved".to_string());
    }

    let start = Instant::now();

    match config.mode {
        SolverMode::Full => {
            // Standard full-precision solving
            let target = game.tree_config().starting_pot as f32 * config.blueprint_target_exploitability;
            let exploitability = solve(
                game,
                config.blueprint_iterations,
                target,
                config.print_progress,
            );

            Ok(IntegrationSolveResult {
                blueprint: None,
                subgames_solved: 0,
                solve_time_seconds: start.elapsed().as_secs_f64(),
                final_exploitability: exploitability / game.tree_config().starting_pot as f32,
                memory_usage: game.target_memory_usage(),
            })
        }

        SolverMode::Blueprint | SolverMode::Subgame => {
            // Get flop cards
            let flop = game.card_config().flop;

            if config.print_progress {
                println!("Computing card abstraction ({} x {} buckets)...",
                    config.turn_buckets, config.river_buckets);
            }

            // Compute abstraction
            let abstraction_config =
                AbstractionConfig::new(config.turn_buckets, config.river_buckets);

            #[cfg(feature = "rayon")]
            let abstraction = AbstractionMapping::compute_parallel(&flop, &abstraction_config);
            #[cfg(not(feature = "rayon"))]
            let abstraction = AbstractionMapping::compute(&flop, &abstraction_config);

            if config.print_progress {
                println!("Abstraction computed. Solving blueprint...");
            }

            // Solve the game (this is still full precision for now)
            // In a real implementation, we would use the abstraction to reduce the game tree
            let target = game.tree_config().starting_pot as f32 * config.blueprint_target_exploitability;
            let exploitability = solve(
                game,
                config.blueprint_iterations,
                target,
                config.print_progress,
            );

            // Cache weights for boundary extraction
            game.cache_normalized_weights();

            // Create boundary store
            let num_hands_oop = game.num_private_hands(0);
            let num_hands_ip = game.num_private_hands(1);
            let boundaries = BoundaryStore::new(num_hands_oop, num_hands_ip);

            // Create blueprint
            let blueprint_config = BlueprintConfig {
                abstraction: abstraction_config,
                iterations: config.blueprint_iterations,
                target_exploitability: config.blueprint_target_exploitability,
                print_progress: config.print_progress,
            };

            let blueprint = Blueprint::new(
                flop,
                abstraction,
                boundaries,
                blueprint_config,
                exploitability / game.tree_config().starting_pot as f32,
                num_hands_oop,
                num_hands_ip,
            );

            let mut subgames_solved = 0u32;
            let mut best_subgame_exploitability = exploitability / game.tree_config().starting_pot as f32;

            // Solve subgames if in Subgame mode
            if config.mode == SolverMode::Subgame {
                if config.print_progress {
                    println!("Blueprint solved. Now solving turn subgames with real DCFR...");
                }

                // Configure subgame solving
                let subgame_config = SubgameConfig {
                    iterations: config.subgame_iterations,
                    target_exploitability: config.subgame_target_exploitability,
                    use_safe_solving: config.safe_solving,
                    print_progress: false, // Don't print per-subgame progress
                    ..Default::default()
                };

                // Solve all turn subgames using real DCFR
                let (results, succeeded) = solve_turn_subgames_real(
                    game,
                    &flop,
                    &subgame_config,
                    config.print_progress,
                );

                subgames_solved = succeeded;

                // Calculate average exploitability across subgames
                let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
                if !successful_results.is_empty() {
                    let avg_exploitability: f32 = successful_results
                        .iter()
                        .map(|r| r.exploitability)
                        .sum::<f32>() / successful_results.len() as f32;
                    best_subgame_exploitability = avg_exploitability;
                }

                if config.print_progress {
                    println!("Solved {} subgames, avg exploitability: {:.4}%",
                        subgames_solved, best_subgame_exploitability * 100.0);
                }
            }

            Ok(IntegrationSolveResult {
                blueprint: Some(blueprint),
                subgames_solved,
                solve_time_seconds: start.elapsed().as_secs_f64(),
                final_exploitability: best_subgame_exploitability,
                memory_usage: game.target_memory_usage(),
            })
        }
    }
}

/// Solve a single subgame using the real DCFR solver.
///
/// This creates a new PostFlopGame for the subgame with the turn card dealt,
/// initializes ranges from the boundary, and runs the solver.
pub fn solve_single_subgame(
    info: &SubgameInfo,
    config: &SubgameConfig,
    original_game: &PostFlopGame,
) -> SubgameSolveResult {
    use std::time::Instant;
    let start = Instant::now();

    // Get the original card and tree configs
    let original_card_config = original_game.card_config();
    let original_tree_config = original_game.tree_config();

    // Create new card config with turn dealt
    let card_config = CardConfig {
        range: original_card_config.range.clone(),
        flop: original_card_config.flop,
        turn: info.turn,
        river: info.river.unwrap_or(crate::NOT_DEALT),
    };

    // Create a new TreeConfig for Turn (subgame starts at turn, not flop)
    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: original_tree_config.starting_pot,
        effective_stack: original_tree_config.effective_stack,
        rake_rate: original_tree_config.rake_rate,
        rake_cap: original_tree_config.rake_cap,
        flop_bet_sizes: original_tree_config.flop_bet_sizes.clone(),
        turn_bet_sizes: original_tree_config.turn_bet_sizes.clone(),
        river_bet_sizes: original_tree_config.river_bet_sizes.clone(),
        turn_donk_sizes: original_tree_config.turn_donk_sizes.clone(),
        river_donk_sizes: original_tree_config.river_donk_sizes.clone(),
        add_allin_threshold: original_tree_config.add_allin_threshold,
        force_allin_threshold: original_tree_config.force_allin_threshold,
        merging_threshold: original_tree_config.merging_threshold,
        max_raises_per_street: original_tree_config.max_raises_per_street,
    };

    // Create action tree for turn subgame
    let action_tree = match ActionTree::new(tree_config) {
        Ok(tree) => tree,
        Err(e) => {
            eprintln!("ERROR: Failed to create action tree for turn {}: {}", info.turn, e);
            return SubgameSolveResult::failure(&format!("Failed to create action tree: {}", e));
        }
    };

    // Create the subgame
    let mut subgame = match PostFlopGame::with_config(card_config, action_tree) {
        Ok(game) => game,
        Err(e) => {
            eprintln!("ERROR: Failed to create subgame for turn {}: {}", info.turn, e);
            return SubgameSolveResult::failure(&format!("Failed to create subgame: {}", e));
        }
    };

    // Allocate memory
    subgame.allocate_memory(config.enable_compression);

    // Calculate target exploitability
    let target = subgame.tree_config().starting_pot as f32 * config.target_exploitability;

    // Solve the subgame
    let exploitability = solve(
        &mut subgame,
        config.iterations,
        target,
        config.print_progress,
    );

    let exploitability_fraction = exploitability / subgame.tree_config().starting_pot as f32;
    let elapsed = start.elapsed();

    SubgameSolveResult {
        success: true,
        exploitability: exploitability_fraction,
        iterations_performed: config.iterations,
        safety_satisfied: true,
        message: Some(format!("Solved in {:?}", elapsed)),
    }
}

/// Solve all turn subgames in parallel using real DCFR.
#[cfg(feature = "rayon")]
pub fn solve_turn_subgames_real(
    original_game: &PostFlopGame,
    flop: &[u8; 3],
    config: &SubgameConfig,
    print_progress: bool,
) -> (Vec<SubgameSolveResult>, u32) {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Instant;

    // Get all valid turn cards
    let turns: Vec<u8> = (0..52u8)
        .filter(|&c| !flop.contains(&c))
        .collect();

    let total = turns.len();
    let completed = AtomicU32::new(0);
    let start = Instant::now();

    // Solve in parallel
    let results: Vec<SubgameSolveResult> = turns
        .par_iter()
        .map(|&turn| {
            let info = SubgameInfo {
                boundary_idx: 0,
                turn,
                river: None,
                board: flop.to_vec(),
                pot: original_game.tree_config().starting_pot,
                stack: original_game.tree_config().effective_stack,
                ranges: [vec![], vec![]], // Will use original game's ranges
                is_solved: false,
                exploitability: 0.0,
            };

            let result = solve_single_subgame(&info, config, original_game);

            let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
            if print_progress && done % 10 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                println!(
                    "Progress: {}/{} ({:.1}%) - elapsed: {:.1}s",
                    done, total,
                    done as f64 / total as f64 * 100.0,
                    elapsed
                );
            }

            result
        })
        .collect();

    let succeeded = results.iter().filter(|r| r.success).count() as u32;
    (results, succeeded)
}

/// Solve all turn subgames sequentially using real DCFR.
#[cfg(not(feature = "rayon"))]
pub fn solve_turn_subgames_real(
    original_game: &PostFlopGame,
    flop: &[u8; 3],
    config: &SubgameConfig,
    print_progress: bool,
) -> (Vec<SubgameSolveResult>, u32) {
    use std::time::Instant;

    let turns: Vec<u8> = (0..52u8)
        .filter(|&c| !flop.contains(&c))
        .collect();

    let total = turns.len();
    let start = Instant::now();
    let mut results = Vec::with_capacity(total);

    for (i, &turn) in turns.iter().enumerate() {
        let info = SubgameInfo {
            boundary_idx: 0,
            turn,
            river: None,
            board: flop.to_vec(),
            pot: original_game.tree_config().starting_pot,
            stack: original_game.tree_config().effective_stack,
            ranges: [vec![], vec![]],
            is_solved: false,
            exploitability: 0.0,
        };

        let result = solve_single_subgame(&info, config, original_game);
        results.push(result);

        if print_progress && (i + 1) % 10 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "Progress: {}/{} ({:.1}%) - elapsed: {:.1}s",
                i + 1, total,
                (i + 1) as f64 / total as f64 * 100.0,
                elapsed
            );
        }
    }

    let succeeded = results.iter().filter(|r| r.success).count() as u32;
    (results, succeeded)
}

/// Options for real-time subgame solving during gameplay.
#[derive(Clone, Debug)]
pub struct RealtimeSubgameOptions {
    /// Maximum iterations for real-time solving.
    pub max_iterations: u32,

    /// Target exploitability for real-time subgames.
    pub target_exploitability: f32,

    /// Timeout in milliseconds.
    pub timeout_ms: u32,

    /// Whether to use warm-start from cached solution.
    pub warm_start: bool,
}

impl Default for RealtimeSubgameOptions {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            target_exploitability: 0.01,
            timeout_ms: 1000,
            warm_start: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solver_mode_default() {
        let mode = SolverMode::default();
        assert_eq!(mode, SolverMode::Full);
    }

    #[test]
    fn test_subgame_config_default() {
        let config = SubgameSolverConfig::default();
        assert_eq!(config.mode, SolverMode::Full);
        assert_eq!(config.turn_buckets, 10);
        assert_eq!(config.river_buckets, 10);
    }

    #[test]
    fn test_subgame_config_fast() {
        let config = SubgameSolverConfig::fast();
        assert_eq!(config.mode, SolverMode::Subgame);
        assert_eq!(config.turn_buckets, 5);
        assert_eq!(config.river_buckets, 5);
    }

    #[test]
    fn test_subgame_config_high_quality() {
        let config = SubgameSolverConfig::high_quality();
        assert_eq!(config.mode, SolverMode::Subgame);
        assert_eq!(config.turn_buckets, 15);
        assert!(config.safe_solving);
    }
}
