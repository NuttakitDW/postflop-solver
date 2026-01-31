//! Blueprint strategy generation.
//!
//! A blueprint is a coarse solution of the full game tree using card abstraction.
//! It provides:
//! - Flop strategies at full precision
//! - Turn/River strategies at bucket-level precision
//! - Boundary data for initializing subgames
//!
//! The blueprint is generated using the standard DCFR algorithm, but with
//! chance nodes grouped by card buckets rather than individual cards.

use crate::subgame::abstraction::{AbstractionConfig, AbstractionMapping};
use crate::subgame::boundary::{BoundaryData, BoundaryStore};
use crate::Card;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// Configuration for blueprint generation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct BlueprintConfig {
    /// Card abstraction configuration.
    pub abstraction: AbstractionConfig,

    /// Max iterations for blueprint (safety limit).
    pub iterations: u32,

    /// Delta threshold for convergence (fraction of pot).
    pub delta_threshold: f32,

    /// Whether to print progress during generation.
    pub print_progress: bool,
}

impl Default for BlueprintConfig {
    fn default() -> Self {
        Self {
            abstraction: AbstractionConfig::standard(),
            iterations: 1000,
            delta_threshold: 0.001, // 0.1% of pot
            print_progress: true,
        }
    }
}

impl BlueprintConfig {
    /// Create a new blueprint config with custom parameters.
    pub fn new(
        turn_buckets: u8,
        river_buckets: u8,
        iterations: u32,
        delta_threshold: f32,
    ) -> Self {
        Self {
            abstraction: AbstractionConfig::new(turn_buckets, river_buckets),
            iterations,
            delta_threshold,
            print_progress: true,
        }
    }

    /// Create a fast config for testing (fewer buckets, fewer iterations).
    pub fn fast() -> Self {
        Self {
            abstraction: AbstractionConfig::new(5, 5),
            iterations: 100,
            delta_threshold: 0.01, // 1% of pot
            print_progress: false,
        }
    }
}

/// A coarse solution for the full game tree with card abstraction.
///
/// The blueprint contains:
/// - Card abstraction mapping (EHS2 buckets)
/// - Boundary data at street transitions
/// - Configuration used for generation
/// - Final exploitability achieved
///
/// Note: The actual game tree and strategies are stored separately.
/// This struct primarily holds the metadata needed for subgame solving.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Blueprint {
    /// Flop board this blueprint was computed for.
    pub board: [Card; 3],

    /// Card abstraction mapping.
    pub abstraction: AbstractionMapping,

    /// Boundary data at street transitions.
    pub boundaries: BoundaryStore,

    /// Configuration used to generate this blueprint.
    pub config: BlueprintConfig,

    /// Final exploitability achieved (as fraction of pot).
    pub exploitability: f32,

    /// Number of private hands for OOP.
    pub num_oop_hands: usize,

    /// Number of private hands for IP.
    pub num_ip_hands: usize,
}

impl Blueprint {
    /// Create a new blueprint with pre-computed components.
    ///
    /// This is typically called after solving the game tree separately.
    pub fn new(
        board: [Card; 3],
        abstraction: AbstractionMapping,
        boundaries: BoundaryStore,
        config: BlueprintConfig,
        exploitability: f32,
        num_oop_hands: usize,
        num_ip_hands: usize,
    ) -> Self {
        Self {
            board,
            abstraction,
            boundaries,
            config,
            exploitability,
            num_oop_hands,
            num_ip_hands,
        }
    }

    /// Get the turn bucket for a card.
    #[inline]
    pub fn turn_bucket(&self, turn: Card) -> u8 {
        self.abstraction.turn_bucket(turn)
    }

    /// Get the river bucket for a card given a turn.
    #[inline]
    pub fn river_bucket(&self, turn: Card, river: Card) -> u8 {
        self.abstraction.river_bucket(turn, river)
    }

    /// Get a boundary for a specific position.
    pub fn get_boundary(&self, boundary_idx: usize) -> Option<&BoundaryData> {
        self.boundaries.get_flop_boundary(boundary_idx)
    }

    /// Get all flop boundaries.
    pub fn flop_boundaries(&self) -> &[BoundaryData] {
        self.boundaries.flop_boundaries()
    }

    /// Get turn boundaries for a specific flop boundary.
    pub fn turn_boundaries(&self, flop_idx: usize) -> Option<&[BoundaryData]> {
        self.boundaries.turn_boundaries(flop_idx)
    }

    /// Check if the blueprint is valid.
    pub fn is_valid(&self) -> bool {
        !self.boundaries.is_empty()
            && self.abstraction.num_turn_buckets() > 0
            && self.abstraction.num_river_buckets() > 0
    }

    /// Get statistics about this blueprint.
    pub fn stats(&self) -> BlueprintStats {
        BlueprintStats {
            board: self.board,
            turn_buckets: self.abstraction.num_turn_buckets(),
            river_buckets: self.abstraction.num_river_buckets(),
            num_flop_boundaries: self.boundaries.num_flop_boundaries(),
            total_boundaries: self.boundaries.total_boundaries(),
            exploitability: self.exploitability,
            iterations: self.config.iterations,
        }
    }
}

/// Statistics about a blueprint.
#[derive(Clone, Debug)]
pub struct BlueprintStats {
    /// Flop board.
    pub board: [Card; 3],
    /// Number of turn buckets.
    pub turn_buckets: u8,
    /// Number of river buckets per turn.
    pub river_buckets: u8,
    /// Number of Flop→Turn boundaries.
    pub num_flop_boundaries: usize,
    /// Total number of boundaries.
    pub total_boundaries: usize,
    /// Final exploitability.
    pub exploitability: f32,
    /// Number of iterations used.
    pub iterations: u32,
}

impl std::fmt::Display for BlueprintStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Blueprint[board={:?}, buckets={}x{}, boundaries={}/{}, exploit={:.4}, iter={}]",
            self.board,
            self.turn_buckets,
            self.river_buckets,
            self.num_flop_boundaries,
            self.total_boundaries,
            self.exploitability,
            self.iterations
        )
    }
}

/// Builder for creating blueprints step by step.
///
/// This allows more control over the blueprint generation process.
pub struct BlueprintBuilder {
    board: Option<[Card; 3]>,
    config: BlueprintConfig,
    abstraction: Option<AbstractionMapping>,
    boundaries: Option<BoundaryStore>,
    exploitability: f32,
    num_oop_hands: usize,
    num_ip_hands: usize,
}

impl Default for BlueprintBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BlueprintBuilder {
    /// Create a new blueprint builder.
    pub fn new() -> Self {
        Self {
            board: None,
            config: BlueprintConfig::default(),
            abstraction: None,
            boundaries: None,
            exploitability: f32::MAX,
            num_oop_hands: 0,
            num_ip_hands: 0,
        }
    }

    /// Set the flop board.
    pub fn board(mut self, board: [Card; 3]) -> Self {
        self.board = Some(board);
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: BlueprintConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the number of private hands.
    pub fn num_hands(mut self, oop: usize, ip: usize) -> Self {
        self.num_oop_hands = oop;
        self.num_ip_hands = ip;
        self
    }

    /// Compute the card abstraction.
    pub fn compute_abstraction(mut self) -> Result<Self, String> {
        let board = self
            .board
            .ok_or_else(|| "Board must be set before computing abstraction".to_string())?;

        #[cfg(feature = "rayon")]
        {
            self.abstraction =
                Some(AbstractionMapping::compute_parallel(&board, &self.config.abstraction));
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.abstraction =
                Some(AbstractionMapping::compute(&board, &self.config.abstraction));
        }

        Ok(self)
    }

    /// Set a pre-computed abstraction.
    pub fn with_abstraction(mut self, abstraction: AbstractionMapping) -> Self {
        self.abstraction = Some(abstraction);
        self
    }

    /// Set boundary store.
    pub fn with_boundaries(mut self, boundaries: BoundaryStore) -> Self {
        self.boundaries = Some(boundaries);
        self
    }

    /// Set the final exploitability.
    pub fn exploitability(mut self, exploitability: f32) -> Self {
        self.exploitability = exploitability;
        self
    }

    /// Build the blueprint.
    pub fn build(self) -> Result<Blueprint, String> {
        let board = self
            .board
            .ok_or_else(|| "Board must be set".to_string())?;
        let abstraction = self
            .abstraction
            .ok_or_else(|| "Abstraction must be computed".to_string())?;
        let boundaries = self
            .boundaries
            .ok_or_else(|| "Boundaries must be set".to_string())?;

        Ok(Blueprint::new(
            board,
            abstraction,
            boundaries,
            self.config,
            self.exploitability,
            self.num_oop_hands,
            self.num_ip_hands,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blueprint_config_default() {
        let config = BlueprintConfig::default();
        assert_eq!(config.abstraction.turn_buckets, 10);
        assert_eq!(config.abstraction.river_buckets, 10);
        assert_eq!(config.iterations, 1000);
    }

    #[test]
    fn test_blueprint_config_fast() {
        let config = BlueprintConfig::fast();
        assert_eq!(config.abstraction.turn_buckets, 5);
        assert_eq!(config.abstraction.river_buckets, 5);
        assert_eq!(config.iterations, 100);
    }

    #[test]
    fn test_blueprint_builder() {
        let board = [0, 4, 8]; // 2c, 3c, 4c
        let config = BlueprintConfig::fast();

        let result = BlueprintBuilder::new()
            .board(board)
            .config(config)
            .num_hands(100, 100)
            .compute_abstraction();

        assert!(result.is_ok());

        let builder = result.unwrap();

        // Add empty boundaries for testing
        let boundaries = BoundaryStore::new(100, 100);
        let result = builder.with_boundaries(boundaries).exploitability(0.01).build();

        assert!(result.is_ok());
        let blueprint = result.unwrap();

        assert_eq!(blueprint.board, board);
        assert!(blueprint.exploitability > 0.0);
    }

    #[test]
    fn test_blueprint_stats() {
        let board = [0, 4, 8];
        let config = BlueprintConfig::fast();
        let abstraction = AbstractionMapping::compute(&board, &config.abstraction);
        let boundaries = BoundaryStore::new(100, 100);

        let blueprint = Blueprint::new(
            board,
            abstraction,
            boundaries,
            config,
            0.01,
            100,
            100,
        );

        let stats = blueprint.stats();
        assert_eq!(stats.board, board);
        assert_eq!(stats.turn_buckets, 5);
        assert_eq!(stats.river_buckets, 5);
    }
}
