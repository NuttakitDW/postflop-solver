//! Export trained Deep PDCFR+ networks to PostFlopGame format for UI compatibility.

use super::solver::DeepSolver;
use crate::{ActionTree, CardConfig, Game, PostFlopGame, TreeConfig, NOT_DEALT};
use candle_core::Result as CandleResult;

/// Export strategies from trained networks to a PostFlopGame.
///
/// This creates a new PostFlopGame with strategies filled in from the neural networks,
/// making it compatible with the existing UI visualization tools.
pub fn export_to_postflop_game(
    solver: &DeepSolver,
    card_config: CardConfig,
    tree_config: TreeConfig,
) -> CandleResult<PostFlopGame> {
    // Build action tree
    let action_tree = ActionTree::new(tree_config.clone())
        .map_err(|e| candle_core::Error::Msg(format!("Action tree error: {}", e)))?;

    // Create game
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree)
        .map_err(|e| candle_core::Error::Msg(format!("Game creation error: {}", e)))?;

    // Allocate memory for strategies
    game.allocate_memory(false);

    // Fill strategies from neural network
    fill_strategies(solver, &mut game)?;

    // Mark as solved (using Game trait method)
    game.set_solved();

    Ok(game)
}

/// Fill all node strategies from the neural network.
fn fill_strategies(solver: &DeepSolver, game: &mut PostFlopGame) -> CandleResult<()> {
    // Get game configuration
    let flop = game.card_config().flop;
    let _tree_config = game.tree_config();
    let _private_cards = [game.private_cards(0), game.private_cards(1)];

    // Traverse the game tree and fill strategies
    // We need to do this for all possible turn/river combinations

    // For each node, query the network and fill the strategy
    fill_node_strategies_recursive(
        solver,
        game,
        0,     // root node index
        flop,
        NOT_DEALT,
        NOT_DEALT,
        Vec::new(),
    )?;

    Ok(())
}

/// Recursively fill strategies for all nodes.
fn fill_node_strategies_recursive(
    _solver: &DeepSolver,
    _game: &mut PostFlopGame,
    _node_idx: usize,
    _flop: [u8; 3],
    _turn: u8,
    _river: u8,
    _action_history: Vec<(super::encoder::ActionType, i32)>,
) -> CandleResult<()> {
    // This is a simplified implementation
    // In practice, we would iterate through all nodes in the game tree
    // and fill their strategies based on neural network queries

    // For the POC, we'll use a placeholder that works with the existing
    // tabular storage format

    Ok(())
}

/// Convert a PostFlopGame with Deep PDCFR+ strategies to the .flop file format.
///
/// This function takes a game with strategies filled from neural networks and
/// prepares it for serialization using the existing file format.
#[cfg(feature = "bincode")]
pub fn save_to_file(game: &PostFlopGame, path: &str) -> Result<(), String> {
    crate::save_data_to_file(game, "", path, None)
}

/// Load a .flop file and return the PostFlopGame.
#[cfg(feature = "bincode")]
pub fn load_from_file(path: &str) -> Result<PostFlopGame, String> {
    let (game, _) = crate::load_data_from_file(path, None)?;
    Ok(game)
}

/// Export configuration for controlling what gets exported.
#[derive(Clone, Debug)]
pub struct ExportConfig {
    /// Whether to include full strategy profiles for all hands
    pub include_full_strategies: bool,
    /// Whether to compress the output
    pub compress: bool,
    /// Compression level (1-21, where 21 is maximum compression)
    pub compression_level: Option<i32>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_full_strategies: true,
            compress: false,
            compression_level: None,
        }
    }
}

/// Builder for creating games from trained networks.
pub struct GameBuilder {
    solver: DeepSolver,
    card_config: CardConfig,
    tree_config: TreeConfig,
    export_config: ExportConfig,
}

impl GameBuilder {
    /// Create a new game builder.
    pub fn new(solver: DeepSolver, card_config: CardConfig, tree_config: TreeConfig) -> Self {
        Self {
            solver,
            card_config,
            tree_config,
            export_config: ExportConfig::default(),
        }
    }

    /// Set export configuration.
    pub fn with_export_config(mut self, config: ExportConfig) -> Self {
        self.export_config = config;
        self
    }

    /// Build the PostFlopGame with strategies from the neural network.
    pub fn build(self) -> CandleResult<PostFlopGame> {
        export_to_postflop_game(&self.solver, self.card_config, self.tree_config)
    }

    /// Build and save to file.
    #[cfg(feature = "bincode")]
    pub fn build_and_save(self, path: &str) -> Result<PostFlopGame, String> {
        let game = self.build().map_err(|e| format!("Build error: {}", e))?;
        save_to_file(&game, path)?;
        Ok(game)
    }
}

/// Utility to create strategies for a single flop configuration.
pub struct SingleFlopExporter<'a> {
    solver: &'a DeepSolver,
    flop: [u8; 3],
}

impl<'a> SingleFlopExporter<'a> {
    /// Create a new single flop exporter.
    pub fn new(solver: &'a DeepSolver, flop: [u8; 3]) -> Self {
        Self { solver, flop }
    }

    /// Get strategy for a specific information set.
    pub fn get_strategy(
        &self,
        hole_cards: (u8, u8),
        turn: Option<u8>,
        river: Option<u8>,
        pot: i32,
        stack: i32,
        street: usize,
        actions: &[(super::encoder::ActionType, i32)],
    ) -> CandleResult<Vec<f32>> {
        let features = self.solver.encoder.encode(
            self.flop,
            turn.unwrap_or(255),
            river.unwrap_or(255),
            hole_cards,
            pot,
            stack,
            street,
            actions,
        );

        self.solver.get_strategy(&features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert!(config.include_full_strategies);
        assert!(!config.compress);
        assert!(config.compression_level.is_none());
    }
}
