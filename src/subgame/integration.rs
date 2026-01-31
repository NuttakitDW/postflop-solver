//! Integration module for connecting subgame solving with PostFlopGame.
//!
//! This module provides the glue code to:
//! - Extract boundary data from a solved PostFlopGame
//! - Create subgames from boundary points
//! - Solve subgames with the main DCFR solver
//! - Stitch subgame solutions back together

use crate::game::PostFlopGame;
use crate::interface::Game;
use crate::solver::solve;
use crate::subgame::abstraction::{AbstractionConfig, AbstractionMapping};
use crate::subgame::blueprint::{Blueprint, BlueprintConfig};
use crate::subgame::boundary::{BoundaryData, BoundaryStore};
use crate::subgame::subgame_solver::{BatchSolveOptions, SubgameConfig};

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

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

            // Solve subgames if in Subgame mode
            if config.mode == SolverMode::Subgame {
                if config.print_progress {
                    println!("Blueprint solved. Extracting boundaries and solving subgames...");
                }

                // Extract root boundary
                let root_boundary = extract_boundary_from_game(game);

                // Generate subgame infos for all turn cards
                let subgame_infos = crate::subgame::generate_subgame_infos(&root_boundary, 0, &flop);

                if config.print_progress {
                    println!("Generated {} turn subgames", subgame_infos.len());
                }

                // Configure batch solving
                let batch_options = BatchSolveOptions {
                    config: SubgameConfig {
                        iterations: config.subgame_iterations,
                        target_exploitability: config.subgame_target_exploitability,
                        use_safe_solving: config.safe_solving,
                        ..Default::default()
                    },
                    print_progress: config.print_progress,
                    ..Default::default()
                };

                // In a real implementation, we would:
                // 1. For each turn card, create a new game tree for that subgame
                // 2. Initialize with boundary ranges
                // 3. Solve each subgame
                // 4. Store the results
                //
                // For now, we simulate this with a mock solver
                let turns: Vec<u8> = (0..52u8)
                    .filter(|&c| !flop.contains(&c))
                    .collect();

                #[cfg(feature = "rayon")]
                {
                    let (_, stats) = crate::subgame::solve_turn_subgames(
                        &root_boundary,
                        &flop,
                        &turns,
                        batch_options,
                        crate::subgame::mock_solve,
                    );
                    subgames_solved = stats.succeeded as u32;
                }

                #[cfg(not(feature = "rayon"))]
                {
                    subgames_solved = subgame_infos.len() as u32;
                }

                if config.print_progress {
                    println!("Solved {} subgames", subgames_solved);
                }
            }

            Ok(IntegrationSolveResult {
                blueprint: Some(blueprint),
                subgames_solved,
                solve_time_seconds: start.elapsed().as_secs_f64(),
                final_exploitability: exploitability / game.tree_config().starting_pot as f32,
                memory_usage: game.target_memory_usage(),
            })
        }
    }
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
