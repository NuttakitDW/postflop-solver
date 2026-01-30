# Subgame Solving Framework Implementation Plan

## Overview

Design a modular subgame solving framework for the DCFR poker solver that generates a coarse "blueprint" strategy, then refines Turn/River subgames with high precision on-demand.

**Goal:** Generate a full 50bb library with extreme accuracy while significantly reducing total simulation time through decomposition.

---

## Confirmed Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Card Abstraction** | EHS2 (Expected Hand Strength²) | Fast, good accuracy, industry standard |
| **Storage Format** | Single archive bundle (.pfs) | Same UX as current .bin files |
| **UI Changes** | None required | Transparent lazy loading in solver library |
| **Clustering Method** | K-means on EHS2 vectors | Simple, effective, parallelizable |
| **Safe Subgame Solving** | Yes, with gift mechanism | Guarantees no exploitation at boundaries |
| **Bucket Count** | 10 buckets per street | Industry standard (PioSOLVER, GTO+) |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Desktop Client (UNCHANGED)                        │
│  game_load_file() → game.play() → game.strategy()                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Same API
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PostFlopGame (Enhanced Internally)                   │
│  ┌─────────────────┐    ┌────────────────────────────────────────┐ │
│  │  Archive Reader │───▶│  Transparent Subgame Loading           │ │
│  │  (.pfs file)    │    │  - Flop: immediate from blueprint      │ │
│  └─────────────────┘    │  - Turn/River: load chunk on-demand    │ │
│                         └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Storage Format: Archive Bundle (.pfs)

### File Structure
```
game_50bb.pfs (ZIP archive, single file)
├── manifest.json           # Version, config, metadata
├── blueprint.bin           # Flop strategies + bucketed Turn/River
├── boundaries.bin          # Ranges/EVs at street transitions
├── abstraction.bin         # Card-to-bucket mapping tables
└── subgames/
    ├── index.bin           # Subgame offset table for random access
    ├── t00.bin             # Turn card 0, all rivers
    ├── t01.bin             # Turn card 1, all rivers
    ├── ...
    └── t46.bin             # Turn card 46, all rivers
```

### Manifest Schema
```json
{
  "version": "1.0.0",
  "format": "pfs-subgame",
  "created": "2026-01-30T12:00:00Z",
  "config": {
    "turn_buckets": 10,
    "river_buckets": 10,
    "clustering_method": "ehs2",
    "safe_solving": true,
    "blueprint_iterations": 1000,
    "subgame_iterations": 500
  },
  "game": {
    "board": [12, 25, 38],
    "starting_pot": 100,
    "effective_stack": 5000,
    "oop_range": "...",
    "ip_range": "..."
  },
  "stats": {
    "total_subgames": 2162,
    "solved_subgames": 2162,
    "blueprint_exploitability": 0.005,
    "avg_subgame_exploitability": 0.001
  }
}
```

### Random Access Loading
```rust
// src/subgame/archive.rs

pub struct PfsArchive {
    reader: ZipArchive<BufReader<File>>,
    manifest: Manifest,
    blueprint: Blueprint,           // Loaded immediately
    subgame_index: SubgameIndex,    // Offset table for quick lookup
    subgame_cache: LruCache<(Card, Option<Card>), Arc<Subgame>>,
}

impl PfsArchive {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mut archive = ZipArchive::new(BufReader::new(file))?;

        // Load manifest (tiny)
        let manifest = load_manifest(&mut archive)?;

        // Load blueprint (flop strategies + boundaries)
        let blueprint = load_blueprint(&mut archive)?;

        // Load subgame index (offset table only, not actual data)
        let subgame_index = load_subgame_index(&mut archive)?;

        Ok(Self { ... })
    }

    pub fn load_subgame(&mut self, turn: Card, river: Option<Card>) -> Arc<Subgame> {
        // Check cache first
        if let Some(subgame) = self.subgame_cache.get(&(turn, river)) {
            return subgame.clone();
        }

        // Random access into archive using index
        let offset = self.subgame_index.offset_for(turn, river);
        let subgame = self.reader.by_index(offset)?.decompress()?;

        // Cache and return
        let subgame = Arc::new(subgame);
        self.subgame_cache.put((turn, river), subgame.clone());
        subgame
    }
}
```

### Backward Compatibility
```rust
// src/file.rs - Modified load_data_from_file

pub fn load_data_from_file<T: FileData>(
    path: &Path,
    max_memory_usage: Option<u64>,
) -> Result<T, String> {
    // Detect file type by extension or magic number
    let extension = path.extension().and_then(|s| s.to_str());

    match extension {
        Some("pfs") => {
            // New archive format - returns PostFlopGame with lazy loading
            load_pfs_archive(path, max_memory_usage)
        }
        Some("bin") | _ => {
            // Legacy format - existing behavior unchanged
            load_legacy_bin(path, max_memory_usage)
        }
    }
}
```

---

## 2. Module Structure

```
src/
├── subgame/
│   ├── mod.rs              # Module exports, feature flags
│   ├── abstraction.rs      # EHS2 computation, k-means clustering
│   ├── blueprint.rs        # Blueprint struct, generation
│   ├── boundary.rs         # BoundaryData, BoundaryStore
│   ├── subgame_solver.rs   # Subgame solving (safe/unsafe modes)
│   ├── stitching.rs        # StitchedGame facade
│   ├── archive.rs          # .pfs archive reader/writer
│   └── serialization.rs    # Bincode encoding for subgame types
├── solver.rs               # Add SolverMode enum
├── file.rs                 # Add .pfs detection and loading
└── game/
    ├── mod.rs              # Add boundary detection
    ├── base.rs             # Add subgame extraction
    └── interpreter.rs      # Hook for lazy subgame loading
```

---

## 3. Core Data Structures

### 3.1 Card Abstraction (EHS2)

```rust
// src/subgame/abstraction.rs

/// Configuration for card abstraction
#[derive(Clone, Debug, Encode, Decode)]
pub struct AbstractionConfig {
    pub turn_buckets: u8,              // e.g., 10
    pub river_buckets_per_turn: u8,    // e.g., 10
}

/// Precomputed abstraction mapping for O(1) lookup
#[derive(Clone, Debug, Encode, Decode)]
pub struct AbstractionMapping {
    /// Flop board this abstraction was computed for
    pub board: [Card; 3],

    /// Turn card -> bucket_id (52 entries, invalid cards = 255)
    pub turn_to_bucket: [u8; 52],

    /// (Turn card, River card) -> bucket_id
    /// Indexed as: turn_card * 52 + river_card
    pub river_to_bucket: [u8; 52 * 52],

    /// Representative card for each turn bucket (for display)
    pub turn_representatives: Vec<Card>,

    /// Representative card for each (turn_bucket, river_bucket)
    pub river_representatives: Vec<Vec<Card>>,
}

impl AbstractionMapping {
    /// Compute abstraction for a given flop board
    pub fn compute(board: &[Card; 3], config: &AbstractionConfig) -> Self {
        // 1. Compute EHS2 for all (turn, hand) combinations
        let turn_features = compute_turn_ehs2(board);

        // 2. K-means cluster turn cards
        let turn_clusters = kmeans(&turn_features, config.turn_buckets as usize);

        // 3. For each turn bucket, compute river EHS2 and cluster
        let river_clusters = turn_clusters.iter()
            .map(|turn_bucket| {
                let river_features = compute_river_ehs2(board, turn_bucket);
                kmeans(&river_features, config.river_buckets_per_turn as usize)
            })
            .collect();

        // 4. Build lookup tables
        Self::from_clusters(board, turn_clusters, river_clusters)
    }

    #[inline]
    pub fn turn_bucket(&self, turn: Card) -> u8 {
        self.turn_to_bucket[turn as usize]
    }

    #[inline]
    pub fn river_bucket(&self, turn: Card, river: Card) -> u8 {
        self.river_to_bucket[turn as usize * 52 + river as usize]
    }
}

/// Compute EHS² (Expected Hand Strength squared) for a hand
fn compute_ehs2(board: &[Card], hand: (Card, Card), num_samples: u32) -> f32 {
    // Monte Carlo sampling of remaining cards
    let mut equity_sum = 0.0;
    let mut equity_sq_sum = 0.0;

    for _ in 0..num_samples {
        let remaining = sample_remaining_cards(board, hand);
        let equity = compute_equity_vs_random(board, hand, &remaining);
        equity_sum += equity;
        equity_sq_sum += equity * equity;
    }

    // Return E[equity²] which captures both mean and variance
    equity_sq_sum / num_samples as f32
}
```

### 3.2 Blueprint

```rust
// src/subgame/blueprint.rs

/// A coarse solution for the full game tree with card abstraction
#[derive(Clone)]
pub struct Blueprint {
    /// The abstracted game tree
    pub game: PostFlopGame,

    /// Card abstraction mapping
    pub abstraction: AbstractionMapping,

    /// Boundary data at street transitions
    pub boundaries: BoundaryStore,

    /// Configuration used to generate this blueprint
    pub config: BlueprintConfig,

    /// Final exploitability achieved
    pub exploitability: f32,
}

#[derive(Clone, Debug, Encode, Decode)]
pub struct BlueprintConfig {
    /// Card abstraction settings
    pub abstraction: AbstractionConfig,

    /// Action abstraction (simplified bet sizing)
    pub action_abstraction: Option<ActionAbstractionConfig>,

    /// Number of DCFR iterations
    pub iterations: u32,

    /// Target exploitability (fraction of pot)
    pub target_exploitability: f32,
}

#[derive(Clone, Debug, Encode, Decode)]
pub struct ActionAbstractionConfig {
    /// Bet sizes as pot fractions (e.g., [0.33, 0.67, 1.0])
    pub bet_fractions: Vec<f64>,

    /// Raise multipliers (e.g., [2.5, 3.0])
    pub raise_multipliers: Vec<f64>,
}

impl Blueprint {
    /// Generate a blueprint for the given game configuration
    pub fn generate(
        card_config: CardConfig,
        tree_config: TreeConfig,
        config: BlueprintConfig,
        print_progress: bool,
    ) -> Result<Self, String> {
        // 1. Compute card abstraction
        let abstraction = AbstractionMapping::compute(
            &card_config.flop,
            &config.abstraction,
        );

        // 2. Build abstracted game tree
        let mut game = PostFlopGame::with_config(card_config, tree_config)?;
        game.allocate_memory(false)?;

        // 3. Solve with blueprint mode
        let exploitability = solve(
            &mut game,
            config.iterations,
            config.target_exploitability,
            print_progress,
            SolverMode::Blueprint(Arc::new(abstraction.clone())),
        );

        // 4. Extract boundaries
        let boundaries = BoundaryStore::extract(&game);

        Ok(Self {
            game,
            abstraction,
            boundaries,
            config,
            exploitability,
        })
    }
}
```

### 3.3 Boundary Data

```rust
// src/subgame/boundary.rs

/// Stores boundary information at all street transition points
#[derive(Clone, Default, Encode, Decode)]
pub struct BoundaryStore {
    /// Boundaries at flop→turn transitions
    /// Key: action history hash (compact representation of path from root)
    pub flop_to_turn: Vec<BoundaryData>,

    /// Boundaries at turn→river transitions
    /// Key: (flop_boundary_idx, turn_bucket)
    pub turn_to_river: Vec<Vec<BoundaryData>>,
}

/// Boundary data at a single street transition
#[derive(Clone, Debug, Encode, Decode)]
pub struct BoundaryData {
    /// Reaching probability for each hand [OOP hands, IP hands]
    pub ranges: [Vec<f32>; 2],

    /// Counterfactual values at this node (for safe solving)
    pub cfvalues: [Vec<f32>; 2],

    /// Expected values per hand (weighted by opponent range)
    pub expected_values: [Vec<f32>; 2],

    /// Current pot size (in chips)
    pub pot: i32,

    /// Remaining effective stack
    pub stack: i32,

    /// Action history to reach this point (for tree navigation)
    pub action_history: Vec<u16>,
}

impl BoundaryStore {
    /// Extract all boundary data from a solved game
    pub fn extract(game: &PostFlopGame) -> Self {
        let mut store = Self::default();

        // DFS traversal to find transition nodes
        Self::traverse_and_extract(
            game,
            game.root_index(),
            &game.initial_weights,
            &mut vec![],
            &mut store,
        );

        store
    }

    fn traverse_and_extract(
        game: &PostFlopGame,
        node_idx: usize,
        ranges: &[Vec<f32>; 2],
        history: &mut Vec<u16>,
        store: &mut Self,
    ) {
        let node = game.node(node_idx);

        // Check if this is a street transition (chance node dealing turn/river)
        if node.is_chance() {
            let street = game.street_at(node_idx);

            if street == Street::Flop {
                // Flop→Turn boundary
                let boundary = Self::extract_boundary(game, node_idx, ranges, history);
                store.flop_to_turn.push(boundary);
            } else if street == Street::Turn {
                // Turn→River boundary
                let boundary = Self::extract_boundary(game, node_idx, ranges, history);
                // Index by parent flop boundary + turn bucket
                store.turn_to_river.last_mut().unwrap().push(boundary);
            }
        }

        // Recurse into children
        for (action_idx, child_idx) in game.children(node_idx) {
            let new_ranges = game.compute_child_ranges(node_idx, action_idx, ranges);
            history.push(action_idx as u16);
            Self::traverse_and_extract(game, child_idx, &new_ranges, history, store);
            history.pop();
        }
    }

    fn extract_boundary(
        game: &PostFlopGame,
        node_idx: usize,
        ranges: &[Vec<f32>; 2],
        history: &[u16],
    ) -> BoundaryData {
        BoundaryData {
            ranges: ranges.clone(),
            cfvalues: game.cfvalues_at(node_idx),
            expected_values: game.expected_values_at(node_idx, ranges),
            pot: game.pot_at(node_idx),
            stack: game.stack_at(node_idx),
            action_history: history.to_vec(),
        }
    }
}
```

### 3.4 Subgame Solver

```rust
// src/subgame/subgame_solver.rs

/// Configuration for subgame refinement
#[derive(Clone, Debug, Encode, Decode)]
pub struct SubgameConfig {
    /// Number of DCFR iterations for subgame
    pub iterations: u32,

    /// Target exploitability
    pub target_exploitability: f32,

    /// Enable value compression (i16 storage)
    pub enable_compression: bool,

    /// Use safe solving with gift mechanism
    pub use_safe_solving: bool,

    /// Safety margin for gift (typically 0.0 to 0.05)
    pub safety_margin: f32,
}

impl Default for SubgameConfig {
    fn default() -> Self {
        Self {
            iterations: 500,
            target_exploitability: 0.001,
            enable_compression: true,
            use_safe_solving: true,
            safety_margin: 0.02,
        }
    }
}

/// A refined subgame for a specific Turn/River runout
pub struct Subgame {
    /// The refined game tree (Turn+River subtree)
    pub game: PostFlopGame,

    /// Boundary this subgame was initialized from
    pub boundary_idx: usize,

    /// Turn card for this subgame
    pub turn: Card,

    /// River card (None if Turn-only subgame)
    pub river: Option<Card>,

    /// Whether solving has completed
    pub is_solved: bool,

    /// Final exploitability
    pub exploitability: f32,
}

/// Safety constraint for safe subgame solving
pub struct SafetyConstraint {
    /// Minimum EV each hand must achieve (from blueprint)
    pub min_ev_guarantee: [Vec<f32>; 2],

    /// Minimum reach probability to prevent over-folding
    pub reach_floor: f32,
}

impl Subgame {
    /// Create a subgame from boundary data
    pub fn from_boundary(
        boundary: &BoundaryData,
        turn: Card,
        river: Option<Card>,
        tree_config: &TreeConfig,
    ) -> Result<Self, String> {
        // Build a new game tree rooted at the Turn/River
        let card_config = CardConfig {
            flop: boundary.board,
            turn: Some(turn),
            river,
            ..Default::default()
        };

        let mut game = PostFlopGame::with_config(card_config, tree_config.clone())?;

        // Initialize with boundary ranges (not default ranges)
        game.set_initial_weights(&boundary.ranges);
        game.set_pot(boundary.pot);
        game.set_stack(boundary.stack);

        game.allocate_memory(false)?;

        Ok(Self {
            game,
            boundary_idx: 0,
            turn,
            river,
            is_solved: false,
            exploitability: f32::MAX,
        })
    }

    /// Solve this subgame with optional safety constraints
    pub fn solve(&mut self, config: &SubgameConfig, constraint: Option<&SafetyConstraint>) {
        let mode = if config.use_safe_solving && constraint.is_some() {
            SolverMode::Subgame(constraint.unwrap().clone())
        } else {
            SolverMode::Standard
        };

        self.exploitability = solve(
            &mut self.game,
            config.iterations,
            config.target_exploitability,
            false,
            mode,
        );

        self.is_solved = true;
    }
}

/// Apply gift mechanism to enforce EV floors
pub fn apply_gift(
    game: &mut PostFlopGame,
    node_idx: usize,
    player: usize,
    hand_idx: usize,
    deficit: f32,
) {
    // The "gift" mechanism works by:
    // 1. Computing how much value the hand is losing vs blueprint
    // 2. Adjusting opponent's reaching probability to compensate
    // 3. This simulates opponent "giving away" value
    //
    // Implementation uses the Unsafe Subgame Solving approach from:
    // Burch et al. "Solving Imperfect Information Games Using Decomposition"

    let opponent = 1 - player;
    let opponent_reach = game.reach_probability(node_idx, opponent);

    // Compute gift amount to restore EV
    let gift = deficit / opponent_reach.max(1e-6);

    // Add gift to opponent's contribution at this node
    game.add_gift(node_idx, player, hand_idx, gift);
}
```

### 3.5 Stitched Game (Unified Access)

```rust
// src/subgame/stitching.rs

use std::sync::{Arc, RwLock};
use lru::LruCache;

/// Unified game access that transparently handles blueprint + subgames
pub struct StitchedGame {
    /// The base blueprint
    blueprint: Arc<Blueprint>,

    /// Archive reader for loading subgames on-demand
    archive: Option<RwLock<PfsArchive>>,

    /// Pre-solved subgames (for batch generation)
    precomputed_subgames: RwLock<BTreeMap<(Card, Option<Card>), Arc<Subgame>>>,

    /// LRU cache for recently accessed subgames
    subgame_cache: RwLock<LruCache<(Card, Option<Card>), Arc<Subgame>>>,

    /// Configuration for on-demand solving
    subgame_config: SubgameConfig,

    /// Maximum subgames to keep in memory
    max_cached: usize,
}

impl StitchedGame {
    /// Create from a blueprint (subgames solved on-demand)
    pub fn new(blueprint: Blueprint, config: SubgameConfig) -> Self {
        Self {
            blueprint: Arc::new(blueprint),
            archive: None,
            precomputed_subgames: RwLock::new(BTreeMap::new()),
            subgame_cache: RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(100).unwrap()
            )),
            subgame_config: config,
            max_cached: 100,
        }
    }

    /// Create from an archive file (.pfs)
    pub fn from_archive(path: &Path) -> Result<Self, String> {
        let archive = PfsArchive::open(path)?;
        let blueprint = archive.blueprint().clone();
        let config = archive.subgame_config().clone();

        Ok(Self {
            blueprint: Arc::new(blueprint),
            archive: Some(RwLock::new(archive)),
            precomputed_subgames: RwLock::new(BTreeMap::new()),
            subgame_cache: RwLock::new(LruCache::new(
                std::num::NonZeroUsize::new(100).unwrap()
            )),
            subgame_config: config,
            max_cached: 100,
        })
    }

    /// Get strategy at any node (handles subgame lookup transparently)
    pub fn strategy(&self, turn: Option<Card>, river: Option<Card>) -> &[f32] {
        match (turn, river) {
            (None, None) => {
                // Flop node - use blueprint directly
                self.blueprint.game.strategy()
            }
            (Some(t), r) => {
                // Turn or River node - need subgame
                let subgame = self.get_or_load_subgame(t, r);
                subgame.game.strategy()
            }
        }
    }

    fn get_or_load_subgame(&self, turn: Card, river: Option<Card>) -> Arc<Subgame> {
        let key = (turn, river);

        // 1. Check precomputed
        if let Some(sg) = self.precomputed_subgames.read().unwrap().get(&key) {
            return sg.clone();
        }

        // 2. Check cache
        if let Some(sg) = self.subgame_cache.write().unwrap().get(&key) {
            return sg.clone();
        }

        // 3. Load from archive or solve on-demand
        let subgame = if let Some(ref archive) = self.archive {
            archive.write().unwrap().load_subgame(turn, river)
        } else {
            self.solve_subgame_on_demand(turn, river)
        };

        // Cache result
        self.subgame_cache.write().unwrap().put(key, subgame.clone());

        subgame
    }

    fn solve_subgame_on_demand(&self, turn: Card, river: Option<Card>) -> Arc<Subgame> {
        let boundary = self.blueprint.boundaries.get_boundary(turn, river);
        let mut subgame = Subgame::from_boundary(
            boundary,
            turn,
            river,
            &self.blueprint.game.tree_config(),
        ).unwrap();

        let constraint = if self.subgame_config.use_safe_solving {
            Some(SafetyConstraint {
                min_ev_guarantee: boundary.expected_values.clone(),
                reach_floor: self.subgame_config.safety_margin,
            })
        } else {
            None
        };

        subgame.solve(&self.subgame_config, constraint.as_ref());

        Arc::new(subgame)
    }

    /// Pre-solve specific subgames (useful for batch generation)
    #[cfg(feature = "rayon")]
    pub fn precompute_subgames(&self, cards: &[(Card, Option<Card>)]) {
        use rayon::prelude::*;

        let results: Vec<_> = cards.par_iter()
            .map(|&(turn, river)| {
                let subgame = self.solve_subgame_on_demand(turn, river);
                ((turn, river), subgame)
            })
            .collect();

        let mut precomputed = self.precomputed_subgames.write().unwrap();
        for (key, subgame) in results {
            precomputed.insert(key, subgame);
        }
    }
}
```

---

## 4. Solver Integration

### 4.1 SolverMode Enum

```rust
// src/solver.rs

/// Mode of operation for the CFR solver
#[derive(Clone)]
pub enum SolverMode {
    /// Standard full-precision solving (current behavior)
    Standard,

    /// Blueprint mode with card abstraction
    Blueprint(Arc<AbstractionMapping>),

    /// Subgame mode with safety constraints
    Subgame(SafetyConstraint),
}

impl Default for SolverMode {
    fn default() -> Self {
        Self::Standard
    }
}
```

### 4.2 Modified solve() Function

```rust
// src/solver.rs

pub fn solve<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
    mode: SolverMode,  // NEW PARAMETER
) -> f32 {
    // ... existing setup code ...

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t);

        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
                &mode,  // Pass mode to recursive solver
            );
        }

        // ... existing progress/exploitability code ...
    }

    // ... existing finalization ...
}
```

### 4.3 Modified solve_recursive() for Blueprint Mode

```rust
// src/solver.rs (chance node handling)

// Inside solve_recursive, at chance node handling:
if node.is_chance() {
    match mode {
        SolverMode::Blueprint(ref abstraction) => {
            // Group cards by bucket, process each bucket once
            let street = game.current_street(&node);
            let num_buckets = if street == Street::Turn {
                abstraction.turn_buckets
            } else {
                abstraction.river_buckets
            };

            for bucket_id in 0..num_buckets {
                // Get representative card for this bucket
                let rep_card = abstraction.representative(bucket_id, street);

                // Get all cards in this bucket for weighting
                let bucket_cards = abstraction.cards_in_bucket(bucket_id, street);
                let bucket_weight = bucket_cards.len() as f32;

                // Process representative card
                let child_result = process_chance_child(game, node, rep_card, ...);

                // Weight by bucket size (all cards in bucket share this strategy)
                for card in bucket_cards {
                    cfvalues[card] = child_result * bucket_weight / total_weight;
                }
            }
        }
        SolverMode::Standard | SolverMode::Subgame(_) => {
            // Existing isomorphism-based handling
            // ... current code unchanged ...
        }
    }
}
```

### 4.4 Modified solve_recursive() for Safe Subgame Mode

```rust
// src/solver.rs (after CFV computation at decision nodes)

// After computing counterfactual values:
if let SolverMode::Subgame(ref constraint) = mode {
    // Enforce EV floors from blueprint
    for hand_idx in 0..num_hands {
        let current_ev = cfvalues[player][hand_idx];
        let min_ev = constraint.min_ev_guarantee[player][hand_idx];

        if current_ev < min_ev - constraint.reach_floor {
            // Apply gift to restore EV
            let deficit = min_ev - current_ev;
            apply_gift(game, node, player, hand_idx, deficit);

            // Recompute CFV with gift applied
            cfvalues[player][hand_idx] = min_ev;
        }
    }
}
```

---

## 5. File I/O Integration

### 5.1 Modified load_data_from_file

```rust
// src/file.rs

pub fn load_data_from_file<T: FileData>(
    path: &Path,
    max_memory_usage: Option<u64>,
) -> Result<T, String> {
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    match extension {
        "pfs" => {
            // New archive format with subgame support
            let stitched = StitchedGame::from_archive(path)?;
            // Return as PostFlopGame (StitchedGame implements Deref<Target=PostFlopGame>)
            Ok(stitched.into_game())
        }
        "bin" | _ => {
            // Legacy format - unchanged behavior
            load_legacy_format(path, max_memory_usage)
        }
    }
}
```

### 5.2 Archive Writer

```rust
// src/subgame/archive.rs

impl PfsArchive {
    /// Create a new archive from blueprint and subgames
    pub fn create(
        path: &Path,
        blueprint: &Blueprint,
        subgames: &[(Card, Option<Card>, Subgame)],
        compression_level: Option<i32>,
    ) -> Result<(), String> {
        let file = File::create(path)?;
        let mut zip = ZipWriter::new(file);

        let options = FileOptions::default()
            .compression_method(CompressionMethod::Zstd)
            .compression_level(compression_level);

        // Write manifest
        zip.start_file("manifest.json", options)?;
        let manifest = create_manifest(blueprint, subgames);
        serde_json::to_writer(&mut zip, &manifest)?;

        // Write blueprint
        zip.start_file("blueprint.bin", options)?;
        bincode::encode_into_std_write(blueprint, &mut zip, bincode::config::standard())?;

        // Write boundaries
        zip.start_file("boundaries.bin", options)?;
        bincode::encode_into_std_write(&blueprint.boundaries, &mut zip, bincode::config::standard())?;

        // Write abstraction
        zip.start_file("abstraction.bin", options)?;
        bincode::encode_into_std_write(&blueprint.abstraction, &mut zip, bincode::config::standard())?;

        // Write subgame index
        let index = build_subgame_index(subgames);
        zip.start_file("subgames/index.bin", options)?;
        bincode::encode_into_std_write(&index, &mut zip, bincode::config::standard())?;

        // Write individual subgames
        for (turn, river, subgame) in subgames {
            let filename = format!("subgames/t{:02}_r{:02}.bin",
                turn, river.unwrap_or(255));
            zip.start_file(&filename, options)?;
            bincode::encode_into_std_write(subgame, &mut zip, bincode::config::standard())?;
        }

        zip.finish()?;
        Ok(())
    }
}
```

---

## 6. Parallelization with Rayon

### 6.1 Parallel Blueprint Generation

```rust
// src/subgame/blueprint.rs

#[cfg(feature = "rayon")]
pub fn generate_blueprint_parallel(
    card_config: CardConfig,
    tree_config: TreeConfig,
    config: BlueprintConfig,
) -> Result<Blueprint, String> {
    use rayon::prelude::*;

    // Phase 1: Compute abstraction (parallelizable)
    let abstraction = AbstractionMapping::compute_parallel(
        &card_config.flop,
        &config.abstraction,
    );

    // Phase 2: Solve flop strategies (single game, uses internal parallelism)
    let mut game = PostFlopGame::with_config(card_config, tree_config)?;
    game.allocate_memory(false)?;

    let exploitability = solve(
        &mut game,
        config.iterations,
        config.target_exploitability,
        true,
        SolverMode::Blueprint(Arc::new(abstraction.clone())),
    );

    // Phase 3: Extract boundaries
    let boundaries = BoundaryStore::extract(&game);

    Ok(Blueprint {
        game,
        abstraction,
        boundaries,
        config,
        exploitability,
    })
}

impl AbstractionMapping {
    #[cfg(feature = "rayon")]
    pub fn compute_parallel(board: &[Card; 3], config: &AbstractionConfig) -> Self {
        use rayon::prelude::*;

        // Parallelize EHS2 computation across turn cards
        let turn_features: Vec<_> = (0..52u8)
            .into_par_iter()
            .filter(|&card| !board.contains(&card))
            .map(|turn| {
                let features = compute_turn_ehs2_for_card(board, turn);
                (turn, features)
            })
            .collect();

        // K-means clustering (fast enough to be single-threaded)
        let turn_clusters = kmeans_cluster(&turn_features, config.turn_buckets);

        // Parallelize river clustering per turn bucket
        let river_clusters: Vec<_> = turn_clusters
            .par_iter()
            .map(|bucket| {
                let river_features = compute_river_ehs2_for_bucket(board, bucket);
                kmeans_cluster(&river_features, config.river_buckets_per_turn)
            })
            .collect();

        Self::from_clusters(board, turn_clusters, river_clusters)
    }
}
```

### 6.2 Parallel Subgame Batch Solving

```rust
// src/subgame/mod.rs

/// Solve all subgames for a complete library
#[cfg(feature = "rayon")]
pub fn solve_all_subgames(
    blueprint: &Blueprint,
    config: &SubgameConfig,
) -> Vec<(Card, Option<Card>, Subgame)> {
    use rayon::prelude::*;

    // Generate all (turn, river) combinations
    let valid_cards: Vec<Card> = (0..52)
        .filter(|&c| !blueprint.game.board().contains(&c))
        .collect();

    let runouts: Vec<_> = valid_cards.iter()
        .flat_map(|&turn| {
            valid_cards.iter()
                .filter(move |&&river| river != turn)
                .map(move |&river| (turn, Some(river)))
        })
        .collect();

    // Solve in parallel
    runouts.par_iter()
        .map(|&(turn, river)| {
            let boundary = blueprint.boundaries.get_boundary(turn, river);
            let mut subgame = Subgame::from_boundary(
                boundary,
                turn,
                river,
                &blueprint.game.tree_config(),
            ).unwrap();

            let constraint = if config.use_safe_solving {
                Some(SafetyConstraint {
                    min_ev_guarantee: boundary.expected_values.clone(),
                    reach_floor: config.safety_margin,
                })
            } else {
                None
            };

            subgame.solve(config, constraint.as_ref());

            (turn, river, subgame)
        })
        .collect()
}
```

---

## 7. Files Summary

### Files to Modify

| File | Changes |
|------|---------|
| `src/solver.rs` | Add `SolverMode`, modify `solve()` and `solve_recursive()` |
| `src/file.rs` | Add `.pfs` detection and `StitchedGame` loading |
| `src/game/mod.rs` | Add `cfvalues_at()`, `expected_values_at()` helpers |
| `src/game/interpreter.rs` | Hook for transparent subgame access |
| `Cargo.toml` | Add `zip`, `lru` dependencies and `subgame` feature |

### New Files to Create

| File | Purpose |
|------|---------|
| `src/subgame/mod.rs` | Module exports, `solve_all_subgames()` |
| `src/subgame/abstraction.rs` | EHS2, k-means, `AbstractionMapping` |
| `src/subgame/blueprint.rs` | `Blueprint`, `BlueprintConfig` |
| `src/subgame/boundary.rs` | `BoundaryData`, `BoundaryStore` |
| `src/subgame/subgame_solver.rs` | `Subgame`, `SafetyConstraint`, gift mechanism |
| `src/subgame/stitching.rs` | `StitchedGame` facade |
| `src/subgame/archive.rs` | `.pfs` reader/writer |

---

## 8. Cargo.toml Changes

```toml
[dependencies]
# Existing
bincode = { version = "2.0.1", optional = true }
rayon = { version = "1.8.0", optional = true }
zstd = { version = "0.12.4", optional = true, default-features = false }

# New
zip = { version = "0.6", optional = true, default-features = false, features = ["deflate", "zstd"] }
lru = { version = "0.12", optional = true }
serde_json = { version = "1.0", optional = true }

[features]
default = ["bincode", "rayon"]
subgame = ["dep:zip", "dep:lru", "dep:serde_json", "bincode"]
subgame-safe = ["subgame"]    # Enable gift mechanism
```

---

## 9. Verification Plan

### Unit Tests
- [ ] EHS2 computation matches hand-calculated values
- [ ] K-means produces expected number of clusters
- [ ] Boundary extraction captures correct ranges at transition points
- [ ] Archive read/write round-trips correctly

### Integration Tests
- [ ] Blueprint solves to target exploitability
- [ ] Safe subgame EVs never drop below blueprint guarantees
- [ ] `StitchedGame` returns identical strategies whether from archive or on-demand solve
- [ ] Desktop client loads `.pfs` files without modification

### Regression Tests
- [ ] Full library exploitability ≤ blueprint exploitability + margin
- [ ] Memory usage stays within bounds with LRU cache
- [ ] Parallel subgame solving produces deterministic results

### Performance Benchmarks
- [ ] Blueprint generation: target <10% of full solve time
- [ ] Single subgame solve: target <0.5% of full solve time
- [ ] Archive load time: target <2s for complete library
- [ ] Memory: target <2GB for full 50bb library browsing

---

## 10. Implementation Phases

### Phase 1: Card Abstraction Foundation
1. Implement `compute_ehs2()` using existing hand evaluation
2. Implement k-means clustering
3. Create `AbstractionMapping` struct with lookup tables
4. Unit tests for abstraction correctness

### Phase 2: Blueprint Generation
1. Add `SolverMode` enum to solver.rs
2. Modify `solve_recursive()` for bucketed chance nodes
3. Implement `BoundaryStore::extract()`
4. Create `Blueprint` struct and generation function

### Phase 3: Subgame Solving
1. Implement `Subgame::from_boundary()`
2. Add `SolverMode::Subgame` with standard CFR
3. Implement safe solving with gift mechanism
4. Unit tests for safety constraint enforcement

### Phase 4: Archive Format & Stitching
1. Implement `.pfs` archive writer
2. Implement `.pfs` archive reader with random access
3. Create `StitchedGame` facade
4. Modify `file.rs` to detect and load `.pfs`

### Phase 5: Parallelization & Polish
1. Parallelize EHS2 computation
2. Parallelize subgame batch solving
3. Integration tests with desktop client
4. Performance benchmarking and optimization
