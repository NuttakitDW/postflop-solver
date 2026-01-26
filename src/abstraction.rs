//! Hand abstraction via k-means clustering.
//!
//! This module provides k-means clustering for poker hands based on Expected Hand Strength (EHS).
//! Hands are grouped into buckets that behave similarly, reducing memory and computation.

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

use crate::card::{card_pair_to_index, Card, StrengthItem, NOT_DEALT};
use crate::interface::Game;

/// Helper function to process a single (turn, river) runout for EHS computation.
#[inline]
fn process_runout(
    card_lo: Card,
    card_hi: Card,
    flop_mask: u64,
    hand_strength: &[[Vec<StrengthItem>; 2]],
    private_cards: &[Vec<(Card, Card)>; 2],
    player: usize,
    opponent: usize,
    ehs_sum: &mut [f64],
    ehs_sq_sum: &mut [f64],
    runout_count: &mut [u32],
) {
    let board_mask = flop_mask | (1 << card_lo) | (1 << card_hi);
    let idx = card_pair_to_index(card_lo, card_hi);
    let strength_data = &hand_strength[idx];

    if strength_data[player].is_empty() {
        return;
    }

    let opp_strengths = &strength_data[opponent];
    let player_strengths = &strength_data[player];

    // Skip sentinel values (strength=0 for weak, strength=u16::MAX for strong)
    let opp_hands: Vec<_> = opp_strengths
        .iter()
        .filter(|s| s.strength != 0 && s.strength != u16::MAX)
        .collect();

    if opp_hands.is_empty() {
        return;
    }

    // For each player hand, compute equity against opponent range
    for item in player_strengths.iter() {
        if item.strength == 0 || item.strength == u16::MAX {
            continue; // Skip sentinels
        }

        let hand_idx = item.index as usize;
        let (c1, c2) = private_cards[player][hand_idx];
        let hand_mask: u64 = (1 << c1) | (1 << c2);

        // Skip if hand conflicts with board (should already be filtered, but be safe)
        if hand_mask & board_mask != 0 {
            continue;
        }

        // Count wins, ties against valid opponent hands
        let mut wins = 0u32;
        let mut ties = 0u32;
        let mut total = 0u32;

        for opp_item in &opp_hands {
            let opp_idx = opp_item.index as usize;
            let (oc1, oc2) = private_cards[opponent][opp_idx];
            let opp_mask: u64 = (1 << oc1) | (1 << oc2);

            // Skip card conflicts (opponent cards overlap with player cards)
            if opp_mask & hand_mask != 0 {
                continue;
            }

            total += 1;
            if item.strength > opp_item.strength {
                wins += 1;
            } else if item.strength == opp_item.strength {
                ties += 1;
            }
        }

        if total > 0 {
            let equity = (wins as f64 + 0.5 * ties as f64) / total as f64;
            ehs_sum[hand_idx] += equity;
            ehs_sq_sum[hand_idx] += equity * equity;
            runout_count[hand_idx] += 1;
        }
    }
}

/// Configuration for hand abstraction.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct AbstractionConfig {
    /// Number of buckets (clusters) per player.
    pub num_buckets: usize,
    /// Maximum k-means iterations.
    pub max_iterations: usize,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            num_buckets: 50,
            max_iterations: 100,
        }
    }
}

/// Features used for clustering a hand.
#[derive(Debug, Clone, Copy, Default)]
pub struct HandFeatures {
    /// Expected Hand Strength: average equity across all runouts.
    pub ehs: f32,
    /// EHS squared: used for potential-aware clustering.
    pub ehs_squared: f32,
}

impl HandFeatures {
    /// Euclidean distance squared to another feature vector.
    #[inline]
    pub fn distance_squared(&self, other: &HandFeatures) -> f32 {
        let d_ehs = self.ehs - other.ehs;
        let d_ehs2 = self.ehs_squared - other.ehs_squared;
        d_ehs * d_ehs + d_ehs2 * d_ehs2
    }
}

/// Compute EHS (Expected Hand Strength) for all hands of a player.
///
/// EHS is the average equity of a hand across all possible (turn, river) runouts.
/// For each runout, equity = (hands_beaten + 0.5 * hands_tied) / total_hands.
///
/// # Arguments
/// * `hand_strength` - Pre-computed hand strength data from `CardConfig::hand_strength()`
/// * `private_cards` - Private card combinations for each player
/// * `flop` - The flop cards
/// * `turn` - Turn card (or NOT_DEALT)
/// * `river` - River card (or NOT_DEALT)
/// * `player` - Which player (0 or 1)
///
/// # Returns
/// Vector of `HandFeatures` for each hand in the player's range.
pub(crate) fn compute_hand_features(
    hand_strength: &[[Vec<StrengthItem>; 2]],
    private_cards: &[Vec<(Card, Card)>; 2],
    flop: &[Card; 3],
    turn: Card,
    river: Card,
    player: usize,
) -> Vec<HandFeatures> {
    let num_hands = private_cards[player].len();
    let opponent = 1 - player;

    // Accumulators for each hand
    let mut ehs_sum = vec![0.0f64; num_hands];
    let mut ehs_sq_sum = vec![0.0f64; num_hands];
    let mut runout_count = vec![0u32; num_hands];

    let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

    // Iterate over all valid (turn, river) runouts
    // We need to handle three cases:
    // 1. Flop (turn=NOT_DEALT): iterate all (turn, river) pairs
    // 2. Turn (turn dealt, river=NOT_DEALT): iterate all rivers
    // 3. River (both dealt): single runout

    if turn != NOT_DEALT && river != NOT_DEALT {
        // Case 3: Single runout - both turn and river are known
        let (card_lo, card_hi) = if turn < river { (turn, river) } else { (river, turn) };
        process_runout(
            card_lo, card_hi, flop_mask, hand_strength, private_cards,
            player, opponent, &mut ehs_sum, &mut ehs_sq_sum, &mut runout_count,
        );
    } else if turn != NOT_DEALT {
        // Case 2: Turn is dealt, iterate all possible rivers
        for river_card in 0..52u8 {
            if (1 << river_card) & flop_mask != 0 || river_card == turn {
                continue;
            }
            let (card_lo, card_hi) = if turn < river_card { (turn, river_card) } else { (river_card, turn) };
            process_runout(
                card_lo, card_hi, flop_mask, hand_strength, private_cards,
                player, opponent, &mut ehs_sum, &mut ehs_sq_sum, &mut runout_count,
            );
        }
    } else {
        // Case 1: Flop - iterate all (turn, river) pairs
        for board1 in 0..52u8 {
            if (1 << board1) & flop_mask != 0 {
                continue;
            }
            for board2 in (board1 + 1)..52u8 {
                if (1 << board2) & flop_mask != 0 {
                    continue;
                }
                process_runout(
                    board1, board2, flop_mask, hand_strength, private_cards,
                    player, opponent, &mut ehs_sum, &mut ehs_sq_sum, &mut runout_count,
                );
            }
        }
    }

    // Convert to HandFeatures
    (0..num_hands)
        .map(|i| {
            if runout_count[i] > 0 {
                let count = runout_count[i] as f64;
                HandFeatures {
                    ehs: (ehs_sum[i] / count) as f32,
                    ehs_squared: (ehs_sq_sum[i] / count) as f32,
                }
            } else {
                HandFeatures::default()
            }
        })
        .collect()
}

/// K-means++ initialization: select k initial centroids.
///
/// Selects centroids with probability proportional to squared distance
/// from the nearest existing centroid.
pub fn kmeans_init(features: &[HandFeatures], k: usize, rng_seed: u64) -> Vec<HandFeatures> {
    if features.is_empty() || k == 0 {
        return vec![];
    }

    let n = features.len();
    let k = k.min(n);

    // Simple LCG random number generator
    let mut rng_state = rng_seed;
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as f64 / (1u64 << 31) as f64
    };

    let mut centroids = Vec::with_capacity(k);

    // First centroid: random
    let first_idx = (next_rand() * n as f64) as usize % n;
    centroids.push(features[first_idx]);

    // Distance from each point to nearest centroid
    let mut min_dist_sq = vec![f32::MAX; n];

    for _ in 1..k {
        // Update distances
        let last_centroid = centroids.last().unwrap();
        let mut total_dist: f64 = 0.0;

        for (i, feat) in features.iter().enumerate() {
            let d = feat.distance_squared(last_centroid);
            if d < min_dist_sq[i] {
                min_dist_sq[i] = d;
            }
            total_dist += min_dist_sq[i] as f64;
        }

        if total_dist == 0.0 {
            // All points are already centroids
            break;
        }

        // Select next centroid with probability proportional to distance²
        let threshold = next_rand() * total_dist;
        let mut cumulative = 0.0;
        let mut selected = 0;

        for (i, &d) in min_dist_sq.iter().enumerate() {
            cumulative += d as f64;
            if cumulative >= threshold {
                selected = i;
                break;
            }
        }

        centroids.push(features[selected]);
    }

    centroids
}

/// Assign each hand to the nearest centroid.
///
/// Returns a vector of cluster indices, one per hand.
pub fn assign_to_clusters(features: &[HandFeatures], centroids: &[HandFeatures]) -> Vec<u16> {
    features
        .iter()
        .map(|feat| {
            let mut best_cluster = 0u16;
            let mut best_dist = f32::MAX;

            for (c_idx, centroid) in centroids.iter().enumerate() {
                let d = feat.distance_squared(centroid);
                if d < best_dist {
                    best_dist = d;
                    best_cluster = c_idx as u16;
                }
            }

            best_cluster
        })
        .collect()
}

/// Recompute centroids as the mean of assigned hands.
///
/// Returns the new centroids and whether any changed significantly.
pub fn update_centroids(
    features: &[HandFeatures],
    assignments: &[u16],
    k: usize,
) -> (Vec<HandFeatures>, bool) {
    let mut sums = vec![(0.0f64, 0.0f64); k];
    let mut counts = vec![0usize; k];

    for (feat, &cluster) in features.iter().zip(assignments.iter()) {
        let c = cluster as usize;
        sums[c].0 += feat.ehs as f64;
        sums[c].1 += feat.ehs_squared as f64;
        counts[c] += 1;
    }

    let new_centroids: Vec<HandFeatures> = (0..k)
        .map(|c| {
            if counts[c] > 0 {
                HandFeatures {
                    ehs: (sums[c].0 / counts[c] as f64) as f32,
                    ehs_squared: (sums[c].1 / counts[c] as f64) as f32,
                }
            } else {
                HandFeatures::default()
            }
        })
        .collect();

    // Check for convergence (any centroid moved significantly)
    let converged = new_centroids.iter().zip(sums.iter()).all(|(c, _)| {
        // Consider converged if movement < epsilon
        c.ehs.is_finite() && c.ehs_squared.is_finite()
    });

    (new_centroids, converged)
}

/// Run full k-means clustering.
///
/// # Arguments
/// * `features` - Hand features to cluster
/// * `k` - Number of clusters
/// * `max_iter` - Maximum iterations
/// * `rng_seed` - Random seed for initialization
///
/// # Returns
/// Tuple of (cluster assignments, final centroids, iterations used)
pub fn kmeans(
    features: &[HandFeatures],
    k: usize,
    max_iter: usize,
    rng_seed: u64,
) -> (Vec<u16>, Vec<HandFeatures>, usize) {
    if features.is_empty() || k == 0 {
        return (vec![], vec![], 0);
    }

    let k = k.min(features.len());
    let mut centroids = kmeans_init(features, k, rng_seed);
    let mut assignments = assign_to_clusters(features, &centroids);

    for iter in 0..max_iter {
        let (new_centroids, _) = update_centroids(features, &assignments, k);
        let new_assignments = assign_to_clusters(features, &new_centroids);

        // Check convergence: no assignments changed
        let converged = assignments == new_assignments;

        centroids = new_centroids;
        assignments = new_assignments;

        if converged {
            return (assignments, centroids, iter + 1);
        }
    }

    (assignments, centroids, max_iter)
}

/// Get cluster sizes from assignments.
pub fn cluster_sizes(assignments: &[u16], k: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; k];
    for &c in assignments {
        if (c as usize) < k {
            sizes[c as usize] += 1;
        }
    }
    sizes
}

/// Result of clustering a player's hands.
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster assignment for each hand (index into centroids).
    pub assignments: Vec<u16>,
    /// Cluster centroids (feature values).
    pub centroids: Vec<HandFeatures>,
    /// Number of hands in each cluster.
    pub cluster_sizes: Vec<usize>,
    /// Number of iterations used.
    pub iterations: usize,
    /// Hand features used for clustering.
    pub features: Vec<HandFeatures>,
}

/// Precomputed abstraction data for bucket-based solving.
///
/// Contains hand-to-bucket mappings, bucket weights, and precomputed
/// bucket equity tables for all possible runouts.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct AbstractionData {
    /// Maps hand index to bucket index for each player.
    pub hand_to_bucket: [Vec<u16>; 2],
    /// Maps bucket index to list of hand indices for each player.
    pub bucket_to_hands: [Vec<Vec<u16>>; 2],
    /// Number of buckets for each player.
    pub num_buckets: [usize; 2],
    /// Sum of hand weights for each bucket for each player.
    pub bucket_weights: [Vec<f32>; 2],
    /// Precomputed bucket equity tables.
    /// Indexed as: bucket_equity[runout_index][oop_bucket][ip_bucket]
    /// Value is OOP's equity against IP in that bucket matchup.
    pub bucket_equity: Vec<Vec<Vec<f32>>>,
    /// Precomputed bucket matchup counts (non-blocking pairs).
    /// Indexed as: bucket_matchups[runout_index][oop_bucket][ip_bucket]
    /// Value is the number of non-blocking hand pairs between buckets.
    pub bucket_matchups: Vec<Vec<Vec<u32>>>,
    /// Number of valid runouts stored.
    pub num_runouts: usize,
    /// Maps (turn, river) card pair to runout index (-1 if invalid).
    runout_to_index: Vec<i32>,
}

impl AbstractionData {
    /// Compute abstraction data from a game and clustering configuration.
    ///
    /// This is the main factory method for creating abstraction data.
    /// It performs:
    /// 1. Hand clustering using k-means
    /// 2. Bucket weight computation
    /// 3. Bucket equity precomputation for all runouts
    pub fn compute(
        game: &crate::game::PostFlopGame,
        config: &AbstractionConfig,
    ) -> Self {
        // Cluster hands for both players
        let clustering_oop = cluster_hands(game, 0, config);
        let clustering_ip = cluster_hands(game, 1, config);

        // Build hand-to-bucket and bucket-to-hands mappings
        let hand_to_bucket = [
            clustering_oop.assignments.clone(),
            clustering_ip.assignments.clone(),
        ];

        let bucket_to_hands = [
            build_bucket_to_hands(&clustering_oop.assignments, clustering_oop.cluster_sizes.len()),
            build_bucket_to_hands(&clustering_ip.assignments, clustering_ip.cluster_sizes.len()),
        ];

        let num_buckets = [
            clustering_oop.cluster_sizes.len(),
            clustering_ip.cluster_sizes.len(),
        ];

        // Compute bucket weights
        let initial_weights_oop = game.initial_weights(0);
        let initial_weights_ip = game.initial_weights(1);

        let bucket_weights = [
            compute_bucket_weights(initial_weights_oop, &hand_to_bucket[0], num_buckets[0]),
            compute_bucket_weights(initial_weights_ip, &hand_to_bucket[1], num_buckets[1]),
        ];

        // Precompute bucket equity tables
        let (bucket_equity, bucket_matchups, runout_to_index, num_runouts) = precompute_bucket_equity(
            game,
            &hand_to_bucket,
            &bucket_to_hands,
            num_buckets,
        );

        Self {
            hand_to_bucket,
            bucket_to_hands,
            num_buckets,
            bucket_weights,
            bucket_equity,
            bucket_matchups,
            num_runouts,
            runout_to_index,
        }
    }

    /// Get the bucket index for a hand.
    #[inline]
    pub fn get_bucket(&self, player: usize, hand_idx: usize) -> u16 {
        self.hand_to_bucket[player][hand_idx]
    }

    /// Get the bucket equity for a specific runout.
    ///
    /// Returns the equity matrix for OOP vs IP buckets, or None if the runout is invalid.
    #[inline]
    pub fn get_bucket_equity(&self, turn: Card, river: Card) -> Option<&Vec<Vec<f32>>> {
        let (lo, hi) = if turn < river { (turn, river) } else { (river, turn) };
        let pair_idx = card_pair_to_index(lo, hi);
        let runout_idx = self.runout_to_index[pair_idx];
        if runout_idx < 0 {
            None
        } else {
            Some(&self.bucket_equity[runout_idx as usize])
        }
    }

    /// Get the bucket matchups for a specific runout.
    ///
    /// Returns the matchup count matrix for OOP vs IP buckets, or None if the runout is invalid.
    /// matchups[oop_bucket][ip_bucket] = number of non-blocking hand pairs.
    #[inline]
    pub fn get_bucket_matchups(&self, turn: Card, river: Card) -> Option<&Vec<Vec<u32>>> {
        let (lo, hi) = if turn < river { (turn, river) } else { (river, turn) };
        let pair_idx = card_pair_to_index(lo, hi);
        let runout_idx = self.runout_to_index[pair_idx];
        if runout_idx < 0 {
            None
        } else {
            Some(&self.bucket_matchups[runout_idx as usize])
        }
    }

    /// Get the weight of a bucket.
    #[inline]
    pub fn get_bucket_weight(&self, player: usize, bucket: usize) -> f32 {
        self.bucket_weights[player][bucket]
    }

    /// Get all hands in a bucket.
    #[inline]
    pub fn get_hands_in_bucket(&self, player: usize, bucket: usize) -> &[u16] {
        &self.bucket_to_hands[player][bucket]
    }

    /// Get the number of buckets for a player.
    #[inline]
    pub fn num_buckets(&self, player: usize) -> usize {
        self.num_buckets[player]
    }

    /// Memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let mut usage = 0usize;

        // hand_to_bucket
        for h2b in &self.hand_to_bucket {
            usage += h2b.len() * std::mem::size_of::<u16>();
        }

        // bucket_to_hands
        for b2h in &self.bucket_to_hands {
            for hands in b2h {
                usage += hands.len() * std::mem::size_of::<u16>();
            }
            usage += b2h.len() * std::mem::size_of::<Vec<u16>>();
        }

        // bucket_weights
        for bw in &self.bucket_weights {
            usage += bw.len() * std::mem::size_of::<f32>();
        }

        // bucket_equity
        for runout in &self.bucket_equity {
            for oop_bucket in runout {
                usage += oop_bucket.len() * std::mem::size_of::<f32>();
            }
            usage += runout.len() * std::mem::size_of::<Vec<f32>>();
        }
        usage += self.bucket_equity.len() * std::mem::size_of::<Vec<Vec<f32>>>();

        // bucket_matchups
        for runout in &self.bucket_matchups {
            for oop_bucket in runout {
                usage += oop_bucket.len() * std::mem::size_of::<u32>();
            }
            usage += runout.len() * std::mem::size_of::<Vec<u32>>();
        }
        usage += self.bucket_matchups.len() * std::mem::size_of::<Vec<Vec<u32>>>();

        // runout_to_index
        usage += self.runout_to_index.len() * std::mem::size_of::<i32>();

        usage
    }
}

/// Build bucket-to-hands mapping from hand-to-bucket assignments.
fn build_bucket_to_hands(assignments: &[u16], num_buckets: usize) -> Vec<Vec<u16>> {
    let mut result = vec![Vec::new(); num_buckets];
    for (hand_idx, &bucket) in assignments.iter().enumerate() {
        result[bucket as usize].push(hand_idx as u16);
    }
    result
}

/// Compute bucket weights by summing hand weights per bucket.
pub fn compute_bucket_weights(
    initial_weights: &[f32],
    hand_to_bucket: &[u16],
    num_buckets: usize,
) -> Vec<f32> {
    let mut weights = vec![0.0f32; num_buckets];
    for (hand_idx, &bucket) in hand_to_bucket.iter().enumerate() {
        weights[bucket as usize] += initial_weights[hand_idx];
    }
    weights
}

/// Compute equity and matchup counts between buckets for a single runout.
///
/// Returns:
/// - equity matrix where equity[oop_bucket][ip_bucket] is OOP's equity
/// - matchups matrix where matchups[oop_bucket][ip_bucket] is non-blocking pair count
fn compute_bucket_equity_for_runout(
    hand_strength: &[Vec<StrengthItem>; 2],
    private_cards: &[Vec<(Card, Card)>; 2],
    hand_to_bucket: &[Vec<u16>; 2],
    num_buckets: [usize; 2],
    board_mask: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<u32>>) {
    let oop_strength = &hand_strength[0];
    let ip_strength = &hand_strength[1];

    // Accumulators: wins[oop_bucket][ip_bucket], ties, total
    let mut wins = vec![vec![0u64; num_buckets[1]]; num_buckets[0]];
    let mut ties = vec![vec![0u64; num_buckets[1]]; num_buckets[0]];
    let mut total = vec![vec![0u64; num_buckets[1]]; num_buckets[0]];

    // Build valid hands list for each player (excluding sentinels and card conflicts)
    let valid_oop: Vec<_> = oop_strength
        .iter()
        .filter(|s| s.strength != 0 && s.strength != u16::MAX)
        .map(|s| {
            let hand_idx = s.index as usize;
            let (c1, c2) = private_cards[0][hand_idx];
            let hand_mask = (1u64 << c1) | (1u64 << c2);
            (s.strength, hand_idx, hand_mask)
        })
        .filter(|(_, _, mask)| mask & board_mask == 0)
        .collect();

    let valid_ip: Vec<_> = ip_strength
        .iter()
        .filter(|s| s.strength != 0 && s.strength != u16::MAX)
        .map(|s| {
            let hand_idx = s.index as usize;
            let (c1, c2) = private_cards[1][hand_idx];
            let hand_mask = (1u64 << c1) | (1u64 << c2);
            (s.strength, hand_idx, hand_mask)
        })
        .filter(|(_, _, mask)| mask & board_mask == 0)
        .collect();

    // Compare all OOP hands against all IP hands
    for &(oop_str, oop_hand, oop_mask) in &valid_oop {
        let oop_bucket = hand_to_bucket[0][oop_hand] as usize;

        for &(ip_str, ip_hand, ip_mask) in &valid_ip {
            // Skip if hands share cards
            if oop_mask & ip_mask != 0 {
                continue;
            }

            let ip_bucket = hand_to_bucket[1][ip_hand] as usize;
            total[oop_bucket][ip_bucket] += 1;

            if oop_str > ip_str {
                wins[oop_bucket][ip_bucket] += 1;
            } else if oop_str == ip_str {
                ties[oop_bucket][ip_bucket] += 1;
            }
        }
    }

    // Convert to equity
    let mut equity = vec![vec![0.0f32; num_buckets[1]]; num_buckets[0]];
    let mut matchups = vec![vec![0u32; num_buckets[1]]; num_buckets[0]];

    for oop_b in 0..num_buckets[0] {
        for ip_b in 0..num_buckets[1] {
            matchups[oop_b][ip_b] = total[oop_b][ip_b] as u32;
            if total[oop_b][ip_b] > 0 {
                let eq = (wins[oop_b][ip_b] as f64
                    + 0.5 * ties[oop_b][ip_b] as f64)
                    / total[oop_b][ip_b] as f64;
                equity[oop_b][ip_b] = eq as f32;
            } else {
                // No matchups possible, use 0.5 as default
                equity[oop_b][ip_b] = 0.5;
            }
        }
    }

    (equity, matchups)
}

/// Precompute bucket equity tables for all valid runouts.
///
/// Returns:
/// - Vec of equity matrices (one per runout)
/// - Vec of matchup count matrices (one per runout)
/// - Mapping from (turn, river) pair index to runout index (-1 if invalid)
/// - Number of valid runouts
fn precompute_bucket_equity(
    game: &crate::game::PostFlopGame,
    hand_to_bucket: &[Vec<u16>; 2],
    bucket_to_hands: &[Vec<Vec<u16>>; 2],
    num_buckets: [usize; 2],
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<u32>>>, Vec<i32>, usize) {
    let hand_strength = game.hand_strength();
    let card_config = game.card_config();
    let private_cards: [Vec<(Card, Card)>; 2] = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];

    let flop_mask: u64 = (1 << card_config.flop[0])
        | (1 << card_config.flop[1])
        | (1 << card_config.flop[2]);

    // Maximum pair index for (card, card) pairs
    let max_pair_index = card_pair_to_index(50, 51) + 1;
    let mut runout_to_index = vec![-1i32; max_pair_index];
    let mut bucket_equity = Vec::new();
    let mut bucket_matchups = Vec::new();

    // Determine which runouts to compute based on game state
    let turn_fixed = card_config.turn != NOT_DEALT;
    let river_fixed = card_config.river != NOT_DEALT;

    if turn_fixed && river_fixed {
        // Single runout
        let turn = card_config.turn;
        let river = card_config.river;
        let (lo, hi) = if turn < river { (turn, river) } else { (river, turn) };
        let pair_idx = card_pair_to_index(lo, hi);

        let board_mask = flop_mask | (1 << turn) | (1 << river);
        let strength_idx = card_pair_to_index(lo, hi);

        if !hand_strength[strength_idx][0].is_empty() {
            let (equity, matchups) = compute_bucket_equity_for_runout(
                &hand_strength[strength_idx],
                &private_cards,
                hand_to_bucket,
                num_buckets,
                board_mask,
            );
            runout_to_index[pair_idx] = 0;
            bucket_equity.push(equity);
            bucket_matchups.push(matchups);
        }
    } else if turn_fixed {
        // Turn is dealt, iterate all rivers
        let turn = card_config.turn;
        for river in 0..52u8 {
            if (1 << river) & flop_mask != 0 || river == turn {
                continue;
            }
            let (lo, hi) = if turn < river { (turn, river) } else { (river, turn) };
            let pair_idx = card_pair_to_index(lo, hi);

            let board_mask = flop_mask | (1 << turn) | (1 << river);
            let strength_idx = card_pair_to_index(lo, hi);

            if hand_strength[strength_idx][0].is_empty() {
                continue;
            }

            let (equity, matchups) = compute_bucket_equity_for_runout(
                &hand_strength[strength_idx],
                &private_cards,
                hand_to_bucket,
                num_buckets,
                board_mask,
            );
            runout_to_index[pair_idx] = bucket_equity.len() as i32;
            bucket_equity.push(equity);
            bucket_matchups.push(matchups);
        }
    } else {
        // Flop: iterate all (turn, river) pairs
        for turn in 0..52u8 {
            if (1 << turn) & flop_mask != 0 {
                continue;
            }
            for river in (turn + 1)..52u8 {
                if (1 << river) & flop_mask != 0 {
                    continue;
                }
                let pair_idx = card_pair_to_index(turn, river);

                let board_mask = flop_mask | (1 << turn) | (1 << river);
                let strength_idx = card_pair_to_index(turn, river);

                if hand_strength[strength_idx][0].is_empty() {
                    continue;
                }

                let (equity, matchups) = compute_bucket_equity_for_runout(
                    &hand_strength[strength_idx],
                    &private_cards,
                    hand_to_bucket,
                    num_buckets,
                    board_mask,
                );
                runout_to_index[pair_idx] = bucket_equity.len() as i32;
                bucket_equity.push(equity);
                bucket_matchups.push(matchups);
            }
        }
    }

    let num_runouts = bucket_equity.len();

    // Drop unused bucket_to_hands reference to avoid unused warning
    let _ = bucket_to_hands;

    (bucket_equity, bucket_matchups, runout_to_index, num_runouts)
}

/// Compute hand clustering for a PostFlopGame.
///
/// This is the main entry point for hand abstraction. It:
/// 1. Computes EHS features for all hands
/// 2. Runs k-means clustering
/// 3. Returns cluster assignments
///
/// Must be called after the game is initialized but before memory allocation.
pub fn cluster_hands(
    game: &crate::game::PostFlopGame,
    player: usize,
    config: &AbstractionConfig,
) -> ClusteringResult {
    let private_cards = game.private_cards(player);
    let private_cards_arr: [Vec<(Card, Card)>; 2] = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];
    let hand_strength = game.hand_strength();
    let card_config = game.card_config();

    // Compute features for all hands
    let features = compute_hand_features(
        hand_strength,
        &private_cards_arr,
        &card_config.flop,
        card_config.turn,
        card_config.river,
        player,
    );

    // Run k-means
    let k = config.num_buckets.min(private_cards.len());
    let (assignments, centroids, iterations) = kmeans(&features, k, config.max_iterations, 42);

    let sizes = cluster_sizes(&assignments, k);

    ClusteringResult {
        assignments,
        centroids,
        cluster_sizes: sizes,
        iterations,
        features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hand_features_distance() {
        let a = HandFeatures {
            ehs: 0.5,
            ehs_squared: 0.25,
        };
        let b = HandFeatures {
            ehs: 0.7,
            ehs_squared: 0.49,
        };

        let d = a.distance_squared(&b);
        // (0.2)^2 + (0.24)^2 = 0.04 + 0.0576 = 0.0976
        assert!((d - 0.0976).abs() < 0.001);

        // Distance to self should be 0
        assert_eq!(a.distance_squared(&a), 0.0);
    }

    #[test]
    fn test_kmeans_init() {
        let features: Vec<HandFeatures> = (0..100)
            .map(|i| HandFeatures {
                ehs: i as f32 / 100.0,
                ehs_squared: (i as f32 / 100.0).powi(2),
            })
            .collect();

        let centroids = kmeans_init(&features, 5, 42);
        assert_eq!(centroids.len(), 5);

        // All centroids should be distinct
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                assert!(centroids[i].distance_squared(&centroids[j]) > 0.0);
            }
        }
    }

    #[test]
    fn test_assign_to_clusters() {
        let features = vec![
            HandFeatures { ehs: 0.1, ehs_squared: 0.01 },
            HandFeatures { ehs: 0.15, ehs_squared: 0.02 },
            HandFeatures { ehs: 0.9, ehs_squared: 0.81 },
            HandFeatures { ehs: 0.85, ehs_squared: 0.72 },
        ];

        let centroids = vec![
            HandFeatures { ehs: 0.1, ehs_squared: 0.01 },
            HandFeatures { ehs: 0.9, ehs_squared: 0.81 },
        ];

        let assignments = assign_to_clusters(&features, &centroids);

        // First two hands should be cluster 0, last two cluster 1
        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 0);
        assert_eq!(assignments[2], 1);
        assert_eq!(assignments[3], 1);
    }

    #[test]
    fn test_kmeans_full() {
        // Create clearly separated clusters
        let mut features = Vec::new();
        for i in 0..50 {
            features.push(HandFeatures {
                ehs: 0.1 + (i as f32) * 0.005,
                ehs_squared: 0.01,
            });
        }
        for i in 0..50 {
            features.push(HandFeatures {
                ehs: 0.8 + (i as f32) * 0.003,
                ehs_squared: 0.64,
            });
        }

        let (assignments, centroids, iters) = kmeans(&features, 2, 100, 42);

        assert_eq!(assignments.len(), 100);
        assert_eq!(centroids.len(), 2);
        assert!(iters < 100, "Should converge before max iterations");

        // Check cluster sizes are roughly balanced
        let sizes = cluster_sizes(&assignments, 2);
        assert!(sizes[0] > 0 && sizes[1] > 0);
    }

    #[test]
    fn test_abstraction_config_default() {
        let config = AbstractionConfig::default();
        assert_eq!(config.num_buckets, 50);
        assert_eq!(config.max_iterations, 100);
    }
}
