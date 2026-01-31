//! Card abstraction using Expected Hand Strength squared (EHS²).
//!
//! EHS² captures both the mean and variance of hand strength by squaring
//! equity values before averaging. This provides better clustering than
//! raw EHS because hands with similar EHS but different variance
//! (e.g., drawing hands vs made hands) are distinguished.
//!
//! # Implementation
//!
//! The abstraction process:
//! 1. Compute EHS² for all (turn card, hole cards) combinations
//! 2. K-means cluster turn cards into `turn_buckets` groups
//! 3. For each turn bucket, compute river EHS² and cluster into `river_buckets`
//! 4. Build O(1) lookup tables for runtime queries

use crate::hand::Hand;
use crate::Card;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Default number of Monte Carlo samples for EHS² computation.
/// Higher values = more accuracy but slower. 1000 is a good balance.
pub const DEFAULT_EHS2_SAMPLES: u32 = 1000;

/// Default number of k-means iterations.
pub const DEFAULT_KMEANS_ITERATIONS: u32 = 50;

/// Configuration for card abstraction.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct AbstractionConfig {
    /// Number of buckets for turn cards (typically 10).
    pub turn_buckets: u8,

    /// Number of buckets for river cards per turn bucket (typically 10).
    pub river_buckets: u8,

    /// Number of Monte Carlo samples for EHS² (default: 1000).
    pub num_samples: u32,

    /// Number of k-means iterations (default: 50).
    pub kmeans_iterations: u32,
}

impl AbstractionConfig {
    /// Create a new configuration with default samples and iterations.
    pub fn new(turn_buckets: u8, river_buckets: u8) -> Self {
        Self {
            turn_buckets,
            river_buckets,
            num_samples: DEFAULT_EHS2_SAMPLES,
            kmeans_iterations: DEFAULT_KMEANS_ITERATIONS,
        }
    }

    /// Create configuration with industry-standard 10 buckets per street.
    pub fn standard() -> Self {
        Self::new(10, 10)
    }
}

/// Precomputed abstraction mapping for O(1) bucket lookups.
///
/// After computing the abstraction, this struct provides instant
/// lookup of which bucket any card belongs to.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct AbstractionMapping {
    /// Flop board this abstraction was computed for.
    pub board: [Card; 3],

    /// Configuration used to generate this mapping.
    pub config: AbstractionConfig,

    /// Turn card → bucket_id (52 entries, invalid cards = 255).
    turn_to_bucket: [u8; 52],

    /// (Turn card, River card) → bucket_id.
    /// Indexed as: turn_card * 52 + river_card.
    /// Invalid combinations = 255.
    river_to_bucket: Vec<u8>,

    /// Representative card for each turn bucket (for display/debugging).
    turn_representatives: Vec<Card>,

    /// Representative card for each (turn_bucket, river_bucket).
    /// Indexed as: turn_bucket * river_buckets + river_bucket.
    river_representatives: Vec<Card>,

    /// EHS² centroid values for turn buckets (for analysis).
    turn_centroids: Vec<f32>,

    /// EHS² centroid values for river buckets per turn bucket.
    river_centroids: Vec<Vec<f32>>,
}

impl Default for AbstractionMapping {
    fn default() -> Self {
        Self {
            board: [255; 3],
            config: AbstractionConfig::default(),
            turn_to_bucket: [255; 52],
            river_to_bucket: Vec::new(),
            turn_representatives: Vec::new(),
            river_representatives: Vec::new(),
            turn_centroids: Vec::new(),
            river_centroids: Vec::new(),
        }
    }
}

impl AbstractionMapping {
    /// Compute abstraction for a given flop board.
    ///
    /// This is the main entry point for generating card abstraction.
    /// It computes EHS² values and performs k-means clustering.
    ///
    /// # Arguments
    /// * `board` - The 3-card flop
    /// * `config` - Abstraction configuration
    ///
    /// # Returns
    /// A fully computed AbstractionMapping ready for O(1) queries.
    pub fn compute(board: &[Card; 3], config: &AbstractionConfig) -> Self {
        // 1. Compute EHS² for all valid turn cards
        let turn_ehs2 = compute_turn_ehs2(board, config.num_samples);

        // 2. K-means cluster turn cards
        let (turn_assignments, turn_centroids) = kmeans_cluster(
            &turn_ehs2,
            config.turn_buckets as usize,
            config.kmeans_iterations,
        );

        // 3. Build turn lookup table
        let mut turn_to_bucket = [255u8; 52];
        let mut turn_representatives = vec![255u8; config.turn_buckets as usize];

        for (card, &bucket) in turn_assignments.iter().enumerate() {
            if bucket < 255 {
                turn_to_bucket[card] = bucket;
                // Use first card in each bucket as representative
                if turn_representatives[bucket as usize] == 255 {
                    turn_representatives[bucket as usize] = card as u8;
                }
            }
        }

        // 4. For each turn bucket, compute river EHS² and cluster
        let mut river_to_bucket = vec![255u8; 52 * 52];
        let mut river_representatives =
            vec![255u8; config.turn_buckets as usize * config.river_buckets as usize];
        let mut river_centroids = vec![Vec::new(); config.turn_buckets as usize];

        for turn_bucket in 0..config.turn_buckets as usize {
            // Get all turn cards in this bucket
            let bucket_turns: Vec<Card> = (0..52u8)
                .filter(|&c| turn_to_bucket[c as usize] == turn_bucket as u8)
                .collect();

            if bucket_turns.is_empty() {
                continue;
            }

            // Compute river EHS² for cards paired with turns in this bucket
            // We use the representative turn card for efficiency
            let rep_turn = bucket_turns[0];
            let river_ehs2 = compute_river_ehs2(board, rep_turn, config.num_samples);

            // K-means cluster river cards
            let (river_assignments, centroids) = kmeans_cluster(
                &river_ehs2,
                config.river_buckets as usize,
                config.kmeans_iterations,
            );

            river_centroids[turn_bucket] = centroids;

            // Store river bucket assignments for all turn cards in this bucket
            for &turn in &bucket_turns {
                for (river, &bucket) in river_assignments.iter().enumerate() {
                    if bucket < 255 {
                        let idx = turn as usize * 52 + river;
                        river_to_bucket[idx] = bucket;

                        // Set representative
                        let rep_idx = turn_bucket * config.river_buckets as usize + bucket as usize;
                        if river_representatives[rep_idx] == 255 {
                            river_representatives[rep_idx] = river as u8;
                        }
                    }
                }
            }
        }

        Self {
            board: *board,
            config: config.clone(),
            turn_to_bucket,
            river_to_bucket,
            turn_representatives,
            river_representatives,
            turn_centroids,
            river_centroids,
        }
    }

    /// Compute abstraction using parallel processing.
    #[cfg(feature = "rayon")]
    pub fn compute_parallel(board: &[Card; 3], config: &AbstractionConfig) -> Self {
        // 1. Compute EHS² for all valid turn cards in parallel
        let turn_ehs2 = compute_turn_ehs2_parallel(board, config.num_samples);

        // 2. K-means cluster turn cards
        let (turn_assignments, turn_centroids) = kmeans_cluster(
            &turn_ehs2,
            config.turn_buckets as usize,
            config.kmeans_iterations,
        );

        // 3. Build turn lookup table
        let mut turn_to_bucket = [255u8; 52];
        let mut turn_representatives = vec![255u8; config.turn_buckets as usize];

        for (card, &bucket) in turn_assignments.iter().enumerate() {
            if bucket < 255 {
                turn_to_bucket[card] = bucket;
                if turn_representatives[bucket as usize] == 255 {
                    turn_representatives[bucket as usize] = card as u8;
                }
            }
        }

        // 4. Parallel river clustering for each turn bucket
        let river_data: Vec<_> = (0..config.turn_buckets as usize)
            .into_par_iter()
            .map(|turn_bucket| {
                let bucket_turns: Vec<Card> = (0..52u8)
                    .filter(|&c| turn_to_bucket[c as usize] == turn_bucket as u8)
                    .collect();

                if bucket_turns.is_empty() {
                    return (vec![], Vec::new(), Vec::new());
                }

                let rep_turn = bucket_turns[0];
                let river_ehs2 = compute_river_ehs2_parallel(board, rep_turn, config.num_samples);

                let (river_assignments, centroids) = kmeans_cluster(
                    &river_ehs2,
                    config.river_buckets as usize,
                    config.kmeans_iterations,
                );

                (bucket_turns, river_assignments, centroids)
            })
            .collect();

        // 5. Merge results
        let mut river_to_bucket = vec![255u8; 52 * 52];
        let mut river_representatives =
            vec![255u8; config.turn_buckets as usize * config.river_buckets as usize];
        let mut river_centroids = vec![Vec::new(); config.turn_buckets as usize];

        for (turn_bucket, (bucket_turns, river_assignments, centroids)) in
            river_data.into_iter().enumerate()
        {
            river_centroids[turn_bucket] = centroids;

            for &turn in &bucket_turns {
                for (river, &bucket) in river_assignments.iter().enumerate() {
                    if bucket < 255 {
                        let idx = turn as usize * 52 + river;
                        river_to_bucket[idx] = bucket;

                        let rep_idx =
                            turn_bucket * config.river_buckets as usize + bucket as usize;
                        if river_representatives[rep_idx] == 255 {
                            river_representatives[rep_idx] = river as u8;
                        }
                    }
                }
            }
        }

        Self {
            board: *board,
            config: config.clone(),
            turn_to_bucket,
            river_to_bucket,
            turn_representatives,
            river_representatives,
            turn_centroids,
            river_centroids,
        }
    }

    /// Get the bucket ID for a turn card. O(1).
    #[inline]
    pub fn turn_bucket(&self, turn: Card) -> u8 {
        self.turn_to_bucket[turn as usize]
    }

    /// Get the bucket ID for a river card given a turn. O(1).
    #[inline]
    pub fn river_bucket(&self, turn: Card, river: Card) -> u8 {
        self.river_to_bucket[turn as usize * 52 + river as usize]
    }

    /// Get all turn cards in a given bucket.
    pub fn turns_in_bucket(&self, bucket: u8) -> Vec<Card> {
        (0..52u8)
            .filter(|&c| self.turn_to_bucket[c as usize] == bucket)
            .collect()
    }

    /// Get all river cards in a given bucket for a specific turn.
    pub fn rivers_in_bucket(&self, turn: Card, bucket: u8) -> Vec<Card> {
        (0..52u8)
            .filter(|&c| {
                c != turn
                    && !self.board.contains(&c)
                    && self.river_to_bucket[turn as usize * 52 + c as usize] == bucket
            })
            .collect()
    }

    /// Get the representative turn card for a bucket.
    #[inline]
    pub fn turn_representative(&self, bucket: u8) -> Card {
        self.turn_representatives[bucket as usize]
    }

    /// Get the representative river card for a (turn_bucket, river_bucket) pair.
    #[inline]
    pub fn river_representative(&self, turn_bucket: u8, river_bucket: u8) -> Card {
        let idx = turn_bucket as usize * self.config.river_buckets as usize + river_bucket as usize;
        self.river_representatives[idx]
    }

    /// Get the number of turn buckets.
    #[inline]
    pub fn num_turn_buckets(&self) -> u8 {
        self.config.turn_buckets
    }

    /// Get the number of river buckets per turn.
    #[inline]
    pub fn num_river_buckets(&self) -> u8 {
        self.config.river_buckets
    }

    /// Check if a turn card is valid (not on the board).
    #[inline]
    pub fn is_valid_turn(&self, turn: Card) -> bool {
        !self.board.contains(&turn) && turn < 52
    }

    /// Check if a river card is valid (not on the board or turn).
    #[inline]
    pub fn is_valid_river(&self, turn: Card, river: Card) -> bool {
        !self.board.contains(&river) && river != turn && river < 52
    }
}

/// Compute EHS² (Expected Hand Strength squared) for a turn card.
///
/// EHS² = E[equity²] where equity is computed against a random opponent hand.
/// This captures both the mean and variance of hand strength.
fn compute_turn_ehs2(board: &[Card; 3], num_samples: u32) -> Vec<f32> {
    let mut ehs2_values = vec![0.0f32; 52];

    // Create board mask for conflict detection
    let board_mask: u64 = (1u64 << board[0]) | (1u64 << board[1]) | (1u64 << board[2]);

    for turn in 0..52u8 {
        if board.contains(&turn) {
            continue;
        }

        let turn_mask = board_mask | (1u64 << turn);

        // Monte Carlo sampling of remaining cards
        let mut equity_sq_sum = 0.0f64;
        let mut count = 0u32;

        // Sample random hole cards and river
        let mut rng = SimpleRng::new(turn as u64 * 12345 + 67890);

        for _ in 0..num_samples {
            // Sample hero hand
            let (hero_c1, hero_c2) = sample_two_cards(&mut rng, turn_mask);
            let hero_mask = turn_mask | (1u64 << hero_c1) | (1u64 << hero_c2);

            // Sample villain hand
            let (villain_c1, villain_c2) = sample_two_cards(&mut rng, hero_mask);
            let villain_mask = hero_mask | (1u64 << villain_c1) | (1u64 << villain_c2);

            // Sample river
            let river = sample_one_card(&mut rng, villain_mask);

            // Evaluate hands
            let hero_hand = Hand::new()
                .add_card(board[0] as usize)
                .add_card(board[1] as usize)
                .add_card(board[2] as usize)
                .add_card(turn as usize)
                .add_card(river as usize)
                .add_card(hero_c1 as usize)
                .add_card(hero_c2 as usize);

            let villain_hand = Hand::new()
                .add_card(board[0] as usize)
                .add_card(board[1] as usize)
                .add_card(board[2] as usize)
                .add_card(turn as usize)
                .add_card(river as usize)
                .add_card(villain_c1 as usize)
                .add_card(villain_c2 as usize);

            let hero_strength = hero_hand.evaluate();
            let villain_strength = villain_hand.evaluate();

            // Higher value = stronger hand (0-7461)
            let equity = if hero_strength > villain_strength {
                1.0
            } else if hero_strength < villain_strength {
                0.0
            } else {
                0.5
            };

            equity_sq_sum += equity * equity;
            count += 1;
        }

        ehs2_values[turn as usize] = (equity_sq_sum / count as f64) as f32;
    }

    ehs2_values
}

/// Compute EHS² for river cards given a specific turn.
fn compute_river_ehs2(board: &[Card; 3], turn: Card, num_samples: u32) -> Vec<f32> {
    let mut ehs2_values = vec![0.0f32; 52];

    let board_mask: u64 =
        (1u64 << board[0]) | (1u64 << board[1]) | (1u64 << board[2]) | (1u64 << turn);

    for river in 0..52u8 {
        if board.contains(&river) || river == turn {
            continue;
        }

        let river_mask = board_mask | (1u64 << river);

        let mut equity_sq_sum = 0.0f64;
        let mut count = 0u32;

        let mut rng = SimpleRng::new(river as u64 * 54321 + turn as u64 * 98765);

        for _ in 0..num_samples {
            // Sample hero hand
            let (hero_c1, hero_c2) = sample_two_cards(&mut rng, river_mask);
            let hero_mask = river_mask | (1u64 << hero_c1) | (1u64 << hero_c2);

            // Sample villain hand
            let (villain_c1, villain_c2) = sample_two_cards(&mut rng, hero_mask);

            // Evaluate hands
            let hero_hand = Hand::new()
                .add_card(board[0] as usize)
                .add_card(board[1] as usize)
                .add_card(board[2] as usize)
                .add_card(turn as usize)
                .add_card(river as usize)
                .add_card(hero_c1 as usize)
                .add_card(hero_c2 as usize);

            let villain_hand = Hand::new()
                .add_card(board[0] as usize)
                .add_card(board[1] as usize)
                .add_card(board[2] as usize)
                .add_card(turn as usize)
                .add_card(river as usize)
                .add_card(villain_c1 as usize)
                .add_card(villain_c2 as usize);

            let hero_strength = hero_hand.evaluate();
            let villain_strength = villain_hand.evaluate();

            let equity = if hero_strength > villain_strength {
                1.0
            } else if hero_strength < villain_strength {
                0.0
            } else {
                0.5
            };

            equity_sq_sum += equity * equity;
            count += 1;
        }

        ehs2_values[river as usize] = (equity_sq_sum / count as f64) as f32;
    }

    ehs2_values
}

/// Parallel version of compute_turn_ehs2.
#[cfg(feature = "rayon")]
fn compute_turn_ehs2_parallel(board: &[Card; 3], num_samples: u32) -> Vec<f32> {
    let board_copy = *board;

    (0..52u8)
        .into_par_iter()
        .map(|turn| {
            if board_copy.contains(&turn) {
                return 0.0f32;
            }
            compute_single_turn_ehs2(&board_copy, turn, num_samples)
        })
        .collect()
}

/// Parallel version of compute_river_ehs2.
#[cfg(feature = "rayon")]
fn compute_river_ehs2_parallel(board: &[Card; 3], turn: Card, num_samples: u32) -> Vec<f32> {
    let board_copy = *board;

    (0..52u8)
        .into_par_iter()
        .map(|river| {
            if board_copy.contains(&river) || river == turn {
                return 0.0f32;
            }
            compute_single_river_ehs2(&board_copy, turn, river, num_samples)
        })
        .collect()
}

/// Compute EHS² for a single turn card.
fn compute_single_turn_ehs2(board: &[Card; 3], turn: Card, num_samples: u32) -> f32 {
    let board_mask: u64 = (1u64 << board[0]) | (1u64 << board[1]) | (1u64 << board[2]);
    let turn_mask = board_mask | (1u64 << turn);

    let mut equity_sq_sum = 0.0f64;
    let mut count = 0u32;
    let mut rng = SimpleRng::new(turn as u64 * 12345 + 67890);

    for _ in 0..num_samples {
        let (hero_c1, hero_c2) = sample_two_cards(&mut rng, turn_mask);
        let hero_mask = turn_mask | (1u64 << hero_c1) | (1u64 << hero_c2);

        let (villain_c1, villain_c2) = sample_two_cards(&mut rng, hero_mask);
        let villain_mask = hero_mask | (1u64 << villain_c1) | (1u64 << villain_c2);

        let river = sample_one_card(&mut rng, villain_mask);

        let hero_hand = Hand::new()
            .add_card(board[0] as usize)
            .add_card(board[1] as usize)
            .add_card(board[2] as usize)
            .add_card(turn as usize)
            .add_card(river as usize)
            .add_card(hero_c1 as usize)
            .add_card(hero_c2 as usize);

        let villain_hand = Hand::new()
            .add_card(board[0] as usize)
            .add_card(board[1] as usize)
            .add_card(board[2] as usize)
            .add_card(turn as usize)
            .add_card(river as usize)
            .add_card(villain_c1 as usize)
            .add_card(villain_c2 as usize);

        let hero_strength = hero_hand.evaluate();
        let villain_strength = villain_hand.evaluate();

        let equity = if hero_strength > villain_strength {
            1.0
        } else if hero_strength < villain_strength {
            0.0
        } else {
            0.5
        };

        equity_sq_sum += equity * equity;
        count += 1;
    }

    (equity_sq_sum / count as f64) as f32
}

/// Compute EHS² for a single river card.
fn compute_single_river_ehs2(board: &[Card; 3], turn: Card, river: Card, num_samples: u32) -> f32 {
    let board_mask: u64 =
        (1u64 << board[0]) | (1u64 << board[1]) | (1u64 << board[2]) | (1u64 << turn);
    let river_mask = board_mask | (1u64 << river);

    let mut equity_sq_sum = 0.0f64;
    let mut count = 0u32;
    let mut rng = SimpleRng::new(river as u64 * 54321 + turn as u64 * 98765);

    for _ in 0..num_samples {
        let (hero_c1, hero_c2) = sample_two_cards(&mut rng, river_mask);
        let hero_mask = river_mask | (1u64 << hero_c1) | (1u64 << hero_c2);

        let (villain_c1, villain_c2) = sample_two_cards(&mut rng, hero_mask);

        let hero_hand = Hand::new()
            .add_card(board[0] as usize)
            .add_card(board[1] as usize)
            .add_card(board[2] as usize)
            .add_card(turn as usize)
            .add_card(river as usize)
            .add_card(hero_c1 as usize)
            .add_card(hero_c2 as usize);

        let villain_hand = Hand::new()
            .add_card(board[0] as usize)
            .add_card(board[1] as usize)
            .add_card(board[2] as usize)
            .add_card(turn as usize)
            .add_card(river as usize)
            .add_card(villain_c1 as usize)
            .add_card(villain_c2 as usize);

        let hero_strength = hero_hand.evaluate();
        let villain_strength = villain_hand.evaluate();

        let equity = if hero_strength > villain_strength {
            1.0
        } else if hero_strength < villain_strength {
            0.0
        } else {
            0.5
        };

        equity_sq_sum += equity * equity;
        count += 1;
    }

    (equity_sq_sum / count as f64) as f32
}

/// K-means clustering algorithm.
///
/// Groups values into k clusters based on Euclidean distance.
/// Returns (assignments, centroids) where:
/// - assignments[i] = cluster ID for value i (255 if invalid/skipped)
/// - centroids[k] = centroid value for cluster k
fn kmeans_cluster(values: &[f32], k: usize, max_iterations: u32) -> (Vec<u8>, Vec<f32>) {
    if k == 0 {
        return (vec![255; values.len()], vec![]);
    }

    // Get valid (non-zero) values and their indices
    let valid_items: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i, v))
        .collect();

    if valid_items.is_empty() {
        return (vec![255; values.len()], vec![0.0; k]);
    }

    let n = valid_items.len();
    let actual_k = k.min(n);

    // Initialize centroids using k-means++ (deterministic variant)
    let mut centroids = initialize_centroids_kmeans_pp(&valid_items, actual_k);

    // Assignments for valid items only
    let mut item_assignments = vec![0u8; n];

    for _ in 0..max_iterations {
        // Assignment step: assign each point to nearest centroid
        let mut changed = false;
        for (item_idx, &(_, value)) in valid_items.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut best_cluster = 0u8;

            for (cluster_idx, &centroid) in centroids.iter().enumerate() {
                let dist = (value - centroid).abs();
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = cluster_idx as u8;
                }
            }

            if item_assignments[item_idx] != best_cluster {
                item_assignments[item_idx] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update step: recompute centroids
        let mut sums = vec![0.0f64; actual_k];
        let mut counts = vec![0u32; actual_k];

        for (item_idx, &(_, value)) in valid_items.iter().enumerate() {
            let cluster = item_assignments[item_idx] as usize;
            sums[cluster] += value as f64;
            counts[cluster] += 1;
        }

        for (i, centroid) in centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                *centroid = (sums[i] / counts[i] as f64) as f32;
            }
        }
    }

    // Build full assignment vector
    let mut assignments = vec![255u8; values.len()];
    for (item_idx, &(orig_idx, _)) in valid_items.iter().enumerate() {
        assignments[orig_idx] = item_assignments[item_idx];
    }

    // Pad centroids to k if needed
    while centroids.len() < k {
        centroids.push(0.0);
    }

    (assignments, centroids)
}

/// Initialize centroids using k-means++ algorithm (deterministic variant).
fn initialize_centroids_kmeans_pp(items: &[(usize, f32)], k: usize) -> Vec<f32> {
    if items.is_empty() || k == 0 {
        return vec![];
    }

    let mut centroids = Vec::with_capacity(k);

    // First centroid: middle value
    let mid_idx = items.len() / 2;
    centroids.push(items[mid_idx].1);

    // Remaining centroids: choose points furthest from existing centroids
    while centroids.len() < k {
        let mut max_min_dist = 0.0f32;
        let mut best_value = items[0].1;

        for &(_, value) in items {
            let min_dist = centroids
                .iter()
                .map(|&c| (value - c).abs())
                .fold(f32::MAX, f32::min);

            if min_dist > max_min_dist {
                max_min_dist = min_dist;
                best_value = value;
            }
        }

        centroids.push(best_value);
    }

    centroids
}

/// Simple pseudo-random number generator (xorshift64).
/// Used for deterministic Monte Carlo sampling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    #[inline]
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_bounded(&mut self, bound: u64) -> u64 {
        // Fast, slightly biased method (fine for our purposes)
        self.next() % bound
    }
}

/// Sample one card not in the mask.
#[inline]
fn sample_one_card(rng: &mut SimpleRng, mask: u64) -> Card {
    let available = 52 - mask.count_ones();
    let mut idx = rng.next_bounded(available as u64) as u32;

    for card in 0..52u8 {
        if (mask >> card) & 1 == 0 {
            if idx == 0 {
                return card;
            }
            idx -= 1;
        }
    }
    unreachable!()
}

/// Sample two distinct cards not in the mask.
#[inline]
fn sample_two_cards(rng: &mut SimpleRng, mask: u64) -> (Card, Card) {
    let c1 = sample_one_card(rng, mask);
    let c2 = sample_one_card(rng, mask | (1u64 << c1));
    (c1, c2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abstraction_config() {
        let config = AbstractionConfig::standard();
        assert_eq!(config.turn_buckets, 10);
        assert_eq!(config.river_buckets, 10);
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_sample_cards() {
        let mut rng = SimpleRng::new(12345);
        let mask: u64 = 0b111; // Cards 0, 1, 2 are blocked

        for _ in 0..100 {
            let card = sample_one_card(&mut rng, mask);
            assert!(card >= 3 && card < 52);
        }
    }

    #[test]
    fn test_sample_two_cards_distinct() {
        let mut rng = SimpleRng::new(12345);
        let mask: u64 = 0;

        for _ in 0..100 {
            let (c1, c2) = sample_two_cards(&mut rng, mask);
            assert_ne!(c1, c2);
            assert!(c1 < 52);
            assert!(c2 < 52);
        }
    }

    #[test]
    fn test_kmeans_basic() {
        // Test with clear clusters: 0.1, 0.2 should cluster together,
        // 0.8, 0.9 should cluster together
        let values = vec![0.1, 0.2, 0.0, 0.8, 0.9]; // index 2 is invalid (0.0)

        let (assignments, centroids) = kmeans_cluster(&values, 2, 50);

        // Check that index 2 is marked invalid
        assert_eq!(assignments[2], 255);

        // Check that 0.1, 0.2 are in the same cluster
        assert_eq!(assignments[0], assignments[1]);

        // Check that 0.8, 0.9 are in the same cluster
        assert_eq!(assignments[3], assignments[4]);

        // Check that low and high values are in different clusters
        assert_ne!(assignments[0], assignments[3]);

        // Check centroids are reasonable
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_compute_turn_ehs2() {
        let board = [0, 4, 8]; // 2c, 3c, 4c
        let ehs2 = compute_turn_ehs2(&board, 100);

        // Check that board cards have 0 EHS2
        assert_eq!(ehs2[0], 0.0);
        assert_eq!(ehs2[4], 0.0);
        assert_eq!(ehs2[8], 0.0);

        // Check that other cards have non-zero EHS2
        for card in 0..52u8 {
            if !board.contains(&card) {
                assert!(
                    ehs2[card as usize] > 0.0,
                    "Card {} has zero EHS2",
                    card
                );
                assert!(
                    ehs2[card as usize] <= 1.0,
                    "Card {} has EHS2 > 1.0",
                    card
                );
            }
        }
    }

    #[test]
    fn test_compute_river_ehs2() {
        let board = [0, 4, 8]; // 2c, 3c, 4c
        let turn = 12; // 5c
        let ehs2 = compute_river_ehs2(&board, turn, 100);

        // Check that board and turn cards have 0 EHS2
        assert_eq!(ehs2[0], 0.0);
        assert_eq!(ehs2[4], 0.0);
        assert_eq!(ehs2[8], 0.0);
        assert_eq!(ehs2[12], 0.0);

        // Check that other cards have non-zero EHS2
        for card in 0..52u8 {
            if !board.contains(&card) && card != turn {
                assert!(
                    ehs2[card as usize] > 0.0,
                    "Card {} has zero EHS2",
                    card
                );
            }
        }
    }

    #[test]
    fn test_abstraction_mapping_compute() {
        let board = [0, 4, 8]; // 2c, 3c, 4c
        let config = AbstractionConfig::new(3, 3); // Small for fast test

        let mapping = AbstractionMapping::compute(&board, &config);

        // Check that board cards are invalid
        assert_eq!(mapping.turn_bucket(0), 255);
        assert_eq!(mapping.turn_bucket(4), 255);
        assert_eq!(mapping.turn_bucket(8), 255);

        // Check that valid turn cards have valid buckets
        for card in 0..52u8 {
            if !board.contains(&card) {
                let bucket = mapping.turn_bucket(card);
                assert!(bucket < 3, "Turn card {} has invalid bucket {}", card, bucket);
            }
        }

        // Check river buckets
        let valid_turn = 12u8; // First valid turn
        for river in 0..52u8 {
            if !board.contains(&river) && river != valid_turn {
                let bucket = mapping.river_bucket(valid_turn, river);
                assert!(
                    bucket < 3 || bucket == 255,
                    "River card {} has invalid bucket {}",
                    river,
                    bucket
                );
            }
        }
    }

    #[test]
    fn test_turns_in_bucket() {
        let board = [0, 4, 8];
        let config = AbstractionConfig::new(3, 3);
        let mapping = AbstractionMapping::compute(&board, &config);

        let mut all_turns = Vec::new();
        for bucket in 0..3 {
            let turns = mapping.turns_in_bucket(bucket);
            assert!(!turns.is_empty(), "Bucket {} is empty", bucket);
            all_turns.extend(turns);
        }

        // All valid turns should be in exactly one bucket
        assert_eq!(all_turns.len(), 49); // 52 - 3 board cards
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_parallel_compute() {
        let board = [0, 4, 8];
        let config = AbstractionConfig::new(3, 3);

        // Results should be similar (not identical due to RNG seeding)
        let mapping_seq = AbstractionMapping::compute(&board, &config);
        let mapping_par = AbstractionMapping::compute_parallel(&board, &config);

        // Check same structure
        assert_eq!(
            mapping_seq.num_turn_buckets(),
            mapping_par.num_turn_buckets()
        );
        assert_eq!(
            mapping_seq.num_river_buckets(),
            mapping_par.num_river_buckets()
        );
    }
}
