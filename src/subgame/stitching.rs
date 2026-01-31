//! Stitched game access for transparent subgame handling.
//!
//! The StitchedGame provides a unified interface for accessing game strategies
//! whether they come from the blueprint or from refined subgames. It handles:
//!
//! - Lazy loading of subgames from archive or on-demand solving
//! - LRU caching of recently accessed subgames
//! - Transparent fallback to blueprint when subgames aren't available
//!
//! This allows the desktop client to use the same API regardless of whether
//! it's accessing a full-precision solve or a decomposed blueprint+subgames.

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

#[cfg(feature = "subgame-archive")]
use std::num::NonZeroUsize;

use crate::subgame::blueprint::Blueprint;
use crate::subgame::subgame_solver::{SubgameConfig, SubgameInfo};
use crate::Card;

#[cfg(feature = "subgame-archive")]
use crate::subgame::archive::PfsArchiveReader;

/// Default LRU cache size for subgames.
pub const DEFAULT_CACHE_SIZE: usize = 100;

/// A wrapper around subgame data for caching.
#[derive(Clone)]
pub struct CachedSubgame {
    /// Subgame info.
    pub info: SubgameInfo,

    /// Strategy data (flattened).
    pub strategy: Vec<f32>,

    /// Whether this came from cache.
    pub from_cache: bool,
}

/// Statistics about stitched game usage.
#[derive(Clone, Debug, Default)]
pub struct StitchedGameStats {
    /// Total strategy lookups.
    pub total_lookups: u64,

    /// Blueprint strategy lookups.
    pub blueprint_lookups: u64,

    /// Subgame strategy lookups.
    pub subgame_lookups: u64,

    /// Cache hits.
    pub cache_hits: u64,

    /// Cache misses.
    pub cache_misses: u64,

    /// On-demand solves performed.
    pub on_demand_solves: u64,
}

impl StitchedGameStats {
    /// Calculate cache hit ratio.
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Configuration for StitchedGame behavior.
#[derive(Clone, Debug)]
pub struct StitchedGameConfig {
    /// Maximum number of subgames to cache.
    pub cache_size: usize,

    /// Whether to enable on-demand solving.
    pub enable_on_demand_solving: bool,

    /// Configuration for on-demand subgame solving.
    pub subgame_config: SubgameConfig,

    /// Whether to collect usage statistics.
    pub collect_stats: bool,
}

impl Default for StitchedGameConfig {
    fn default() -> Self {
        Self {
            cache_size: DEFAULT_CACHE_SIZE,
            enable_on_demand_solving: false,
            subgame_config: SubgameConfig::default(),
            collect_stats: true,
        }
    }
}

/// Unified game access with transparent subgame handling.
///
/// This facade provides the same interface whether accessing:
/// - A full-precision monolithic solve
/// - A blueprint with pre-computed subgames
/// - A blueprint with on-demand subgame solving
pub struct StitchedGame {
    /// The base blueprint.
    blueprint: Arc<Blueprint>,

    /// Pre-computed subgames (loaded from archive or solved ahead of time).
    precomputed: RwLock<BTreeMap<(Card, Option<Card>), Arc<CachedSubgame>>>,

    /// LRU cache for recently accessed subgames.
    #[cfg(feature = "subgame-archive")]
    cache: RwLock<lru::LruCache<(Card, Option<Card>), Arc<CachedSubgame>>>,

    #[cfg(not(feature = "subgame-archive"))]
    cache: RwLock<BTreeMap<(Card, Option<Card>), Arc<CachedSubgame>>>,

    /// Archive reader (if loaded from .pfs file).
    #[cfg(feature = "subgame-archive")]
    archive: Option<RwLock<PfsArchiveReader>>,

    /// Configuration.
    config: StitchedGameConfig,

    /// Usage statistics.
    stats: RwLock<StitchedGameStats>,
}

impl StitchedGame {
    /// Create a new StitchedGame from a blueprint.
    pub fn new(blueprint: Blueprint) -> Self {
        Self::with_config(blueprint, StitchedGameConfig::default())
    }

    /// Create a new StitchedGame with custom configuration.
    pub fn with_config(blueprint: Blueprint, config: StitchedGameConfig) -> Self {
        #[cfg(feature = "subgame-archive")]
        let cache = RwLock::new(lru::LruCache::new(
            NonZeroUsize::new(config.cache_size).unwrap_or(NonZeroUsize::new(1).unwrap()),
        ));

        #[cfg(not(feature = "subgame-archive"))]
        let cache = RwLock::new(BTreeMap::new());

        Self {
            blueprint: Arc::new(blueprint),
            precomputed: RwLock::new(BTreeMap::new()),
            cache,
            #[cfg(feature = "subgame-archive")]
            archive: None,
            config,
            stats: RwLock::new(StitchedGameStats::default()),
        }
    }

    /// Create from an archive file.
    #[cfg(feature = "subgame-archive")]
    pub fn from_archive(reader: PfsArchiveReader) -> Result<Self, String> {
        let blueprint = reader
            .blueprint
            .clone()
            .ok_or_else(|| "Archive has no blueprint".to_string())?;

        let config = StitchedGameConfig::default();
        let cache = RwLock::new(lru::LruCache::new(
            NonZeroUsize::new(config.cache_size).unwrap_or(NonZeroUsize::new(1).unwrap()),
        ));

        Ok(Self {
            blueprint: Arc::new(blueprint),
            precomputed: RwLock::new(BTreeMap::new()),
            cache,
            archive: Some(RwLock::new(reader)),
            config,
            stats: RwLock::new(StitchedGameStats::default()),
        })
    }

    /// Get the blueprint.
    pub fn blueprint(&self) -> &Blueprint {
        &self.blueprint
    }

    /// Get the flop board.
    pub fn board(&self) -> &[Card; 3] {
        &self.blueprint.board
    }

    /// Check if a subgame is available (precomputed or in archive).
    pub fn has_subgame(&self, turn: Card, river: Option<Card>) -> bool {
        let key = (turn, river);

        // Check precomputed
        if self.precomputed.read().unwrap().contains_key(&key) {
            return true;
        }

        // Check archive
        #[cfg(feature = "subgame-archive")]
        if let Some(ref archive) = self.archive {
            if archive.read().unwrap().has_subgame(turn, river) {
                return true;
            }
        }

        false
    }

    /// Add a precomputed subgame.
    pub fn add_precomputed(&self, subgame: CachedSubgame) {
        let key = (subgame.info.turn, subgame.info.river);
        self.precomputed
            .write()
            .unwrap()
            .insert(key, Arc::new(subgame));
    }

    /// Get a subgame if available (doesn't solve on demand).
    pub fn get_subgame(&self, turn: Card, river: Option<Card>) -> Option<Arc<CachedSubgame>> {
        let key = (turn, river);

        // Update stats
        if self.config.collect_stats {
            self.stats.write().unwrap().total_lookups += 1;
        }

        // Check precomputed first
        if let Some(subgame) = self.precomputed.read().unwrap().get(&key) {
            if self.config.collect_stats {
                self.stats.write().unwrap().subgame_lookups += 1;
            }
            return Some(subgame.clone());
        }

        // Check cache
        #[cfg(feature = "subgame-archive")]
        {
            if let Some(subgame) = self.cache.write().unwrap().get(&key) {
                if self.config.collect_stats {
                    let mut stats = self.stats.write().unwrap();
                    stats.cache_hits += 1;
                    stats.subgame_lookups += 1;
                }
                return Some(subgame.clone());
            }
        }

        #[cfg(not(feature = "subgame-archive"))]
        {
            if let Some(subgame) = self.cache.read().unwrap().get(&key) {
                if self.config.collect_stats {
                    let mut stats = self.stats.write().unwrap();
                    stats.cache_hits += 1;
                    stats.subgame_lookups += 1;
                }
                return Some(subgame.clone());
            }
        }

        // Try loading from archive
        #[cfg(feature = "subgame-archive")]
        if let Some(ref archive) = self.archive {
            if archive.read().unwrap().has_subgame(turn, river) {
                if self.config.collect_stats {
                    self.stats.write().unwrap().cache_misses += 1;
                }
                // Load from archive and cache
                // In a real implementation, this would decode the subgame data
                // For now, return None to indicate it needs to be loaded
            }
        }

        // Blueprint fallback
        if self.config.collect_stats {
            self.stats.write().unwrap().blueprint_lookups += 1;
        }

        None
    }

    /// Get the current turn bucket for a card.
    pub fn turn_bucket(&self, turn: Card) -> u8 {
        self.blueprint.turn_bucket(turn)
    }

    /// Get the current river bucket for a card pair.
    pub fn river_bucket(&self, turn: Card, river: Card) -> u8 {
        self.blueprint.river_bucket(turn, river)
    }

    /// Get usage statistics.
    pub fn stats(&self) -> StitchedGameStats {
        self.stats.read().unwrap().clone()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.write().unwrap() = StitchedGameStats::default();
    }

    /// Get the number of cached subgames.
    pub fn cache_size(&self) -> usize {
        #[cfg(feature = "subgame-archive")]
        {
            self.cache.read().unwrap().len()
        }

        #[cfg(not(feature = "subgame-archive"))]
        {
            self.cache.read().unwrap().len()
        }
    }

    /// Get the number of precomputed subgames.
    pub fn precomputed_count(&self) -> usize {
        self.precomputed.read().unwrap().len()
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        #[cfg(feature = "subgame-archive")]
        {
            self.cache.write().unwrap().clear();
        }

        #[cfg(not(feature = "subgame-archive"))]
        {
            self.cache.write().unwrap().clear();
        }
    }

    /// Preload subgames for specific cards.
    pub fn preload(&self, cards: &[(Card, Option<Card>)]) {
        for &(turn, river) in cards {
            // Check if already loaded
            if self.has_subgame(turn, river) {
                continue;
            }

            // Try to load from archive
            #[cfg(feature = "subgame-archive")]
            if let Some(ref _archive) = self.archive {
                // Load and cache
                // In a full implementation, this would decode the subgame
            }
        }
    }
}

/// Builder for StitchedGame.
pub struct StitchedGameBuilder {
    blueprint: Option<Blueprint>,
    config: StitchedGameConfig,
    precomputed: Vec<CachedSubgame>,
}

impl Default for StitchedGameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StitchedGameBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            blueprint: None,
            config: StitchedGameConfig::default(),
            precomputed: Vec::new(),
        }
    }

    /// Set the blueprint.
    pub fn blueprint(mut self, blueprint: Blueprint) -> Self {
        self.blueprint = Some(blueprint);
        self
    }

    /// Set the configuration.
    pub fn config(mut self, config: StitchedGameConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the cache size.
    pub fn cache_size(mut self, size: usize) -> Self {
        self.config.cache_size = size;
        self
    }

    /// Enable on-demand solving.
    pub fn enable_on_demand_solving(mut self, enable: bool) -> Self {
        self.config.enable_on_demand_solving = enable;
        self
    }

    /// Add a precomputed subgame.
    pub fn add_precomputed(mut self, subgame: CachedSubgame) -> Self {
        self.precomputed.push(subgame);
        self
    }

    /// Build the StitchedGame.
    pub fn build(self) -> Result<StitchedGame, String> {
        let blueprint = self
            .blueprint
            .ok_or_else(|| "Blueprint is required".to_string())?;

        let game = StitchedGame::with_config(blueprint, self.config);

        // Add precomputed subgames
        for subgame in self.precomputed {
            game.add_precomputed(subgame);
        }

        Ok(game)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subgame::abstraction::AbstractionMapping;
    use crate::subgame::blueprint::BlueprintConfig;
    use crate::subgame::boundary::BoundaryStore;

    fn make_test_blueprint() -> Blueprint {
        let board = [0, 4, 8];
        let config = BlueprintConfig::fast();
        let abstraction = AbstractionMapping::compute(&board, &config.abstraction);
        let boundaries = BoundaryStore::new(10, 10);

        Blueprint::new(board, abstraction, boundaries, config, 0.01, 10, 10)
    }

    #[test]
    fn test_stitched_game_new() {
        let blueprint = make_test_blueprint();
        let game = StitchedGame::new(blueprint);

        assert_eq!(game.board(), &[0, 4, 8]);
        assert_eq!(game.precomputed_count(), 0);
    }

    #[test]
    fn test_stitched_game_with_config() {
        let blueprint = make_test_blueprint();
        let config = StitchedGameConfig {
            cache_size: 50,
            enable_on_demand_solving: true,
            ..Default::default()
        };
        let game = StitchedGame::with_config(blueprint, config);

        assert_eq!(game.board(), &[0, 4, 8]);
    }

    #[test]
    fn test_stitched_game_builder() {
        let blueprint = make_test_blueprint();
        let result = StitchedGameBuilder::new()
            .blueprint(blueprint)
            .cache_size(200)
            .enable_on_demand_solving(true)
            .build();

        assert!(result.is_ok());
        let game = result.unwrap();
        assert_eq!(game.board(), &[0, 4, 8]);
    }

    #[test]
    fn test_stitched_game_stats() {
        let blueprint = make_test_blueprint();
        let game = StitchedGame::new(blueprint);

        // Trigger some lookups
        let _ = game.get_subgame(12, Some(16));
        let _ = game.get_subgame(12, Some(20));

        let stats = game.stats();
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.blueprint_lookups, 2); // Fallback to blueprint

        game.reset_stats();
        let stats = game.stats();
        assert_eq!(stats.total_lookups, 0);
    }

    #[test]
    fn test_add_precomputed() {
        let blueprint = make_test_blueprint();
        let game = StitchedGame::new(blueprint);

        let subgame = CachedSubgame {
            info: SubgameInfo::from_boundary(
                &crate::subgame::boundary::BoundaryData::default(),
                0,
                &[0, 4, 8],
                12,
                Some(16),
            ),
            strategy: vec![0.5; 10],
            from_cache: false,
        };

        game.add_precomputed(subgame);
        assert_eq!(game.precomputed_count(), 1);
        assert!(game.has_subgame(12, Some(16)));
        assert!(!game.has_subgame(12, Some(20)));
    }

    #[test]
    fn test_bucket_lookups() {
        let blueprint = make_test_blueprint();
        let game = StitchedGame::new(blueprint);

        // Turn bucket should be valid (< num_buckets)
        let bucket = game.turn_bucket(12);
        assert!(bucket < 5 || bucket == 255); // 5 buckets in fast config

        // Board cards should be invalid
        let bucket = game.turn_bucket(0);
        assert_eq!(bucket, 255);
    }
}
