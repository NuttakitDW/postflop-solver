//! Hand Clustering for Flop-Only Solving with Abstraction
//!
//! This module implements K-Means clustering on equity distributions to reduce
//! the number of information sets in the game tree.
//!
//! Based on Poker-AI abstraction techniques:
//! - Equity distribution histograms (not single values)
//! - K-Means clustering with Euclidean distance
//! - Pre-computed cluster assignments for fast runtime lookup

use crate::card::Card;

/// Number of bins for equity distribution histogram
pub const NUM_EQUITY_BINS: usize = 10;

/// Default number of clusters
pub const DEFAULT_NUM_CLUSTERS: usize = 50;

/// Equity distribution histogram for a hand
#[derive(Clone, Debug, Default)]
pub struct EquityDistribution {
    pub bins: [f32; NUM_EQUITY_BINS],
}

impl EquityDistribution {
    pub fn new() -> Self {
        Self { bins: [0.0; NUM_EQUITY_BINS] }
    }

    /// Add an equity sample to the histogram
    #[inline]
    pub fn add_sample(&mut self, equity: f32) {
        let bin_idx = ((equity * NUM_EQUITY_BINS as f32) as usize).min(NUM_EQUITY_BINS - 1);
        self.bins[bin_idx] += 1.0;
    }

    /// Normalize to sum to 1.0
    pub fn normalize(&mut self) {
        let sum: f32 = self.bins.iter().sum();
        if sum > 0.0 {
            for b in &mut self.bins {
                *b /= sum;
            }
        }
    }

    /// Euclidean distance to another distribution
    #[inline]
    pub fn distance(&self, other: &EquityDistribution) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..NUM_EQUITY_BINS {
            let diff = self.bins[i] - other.bins[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Get approximate mean equity from histogram
    pub fn mean_equity(&self) -> f32 {
        let mut sum = 0.0f32;
        for (i, &v) in self.bins.iter().enumerate() {
            sum += (i as f32 + 0.5) / NUM_EQUITY_BINS as f32 * v;
        }
        sum
    }
}

/// Cluster assignment data for a flop
#[derive(Clone, Debug)]
pub struct FlopClusterData {
    /// Number of clusters
    pub num_clusters: usize,
    /// Cluster centroids (equity distributions)
    pub centroids: Vec<EquityDistribution>,
    /// Cluster assignment for each hand index [player][hand_idx] -> cluster_id
    pub assignments: [Vec<u16>; 2],
    /// Number of hands in each cluster [player][cluster_id] -> count
    pub cluster_sizes: [Vec<usize>; 2],
    /// Mean equity per cluster [player][cluster_id] -> equity
    pub cluster_equities: [Vec<f32>; 2],
}

impl Default for FlopClusterData {
    fn default() -> Self {
        Self {
            num_clusters: 0,
            centroids: Vec::new(),
            assignments: [Vec::new(), Vec::new()],
            cluster_sizes: [Vec::new(), Vec::new()],
            cluster_equities: [Vec::new(), Vec::new()],
        }
    }
}

impl FlopClusterData {
    /// Create cluster data with specified number of clusters
    pub fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            centroids: Vec::with_capacity(num_clusters),
            assignments: [Vec::new(), Vec::new()],
            cluster_sizes: [vec![0; num_clusters], vec![0; num_clusters]],
            cluster_equities: [vec![0.0; num_clusters], vec![0.0; num_clusters]],
        }
    }

    /// Get cluster ID for a hand
    #[inline]
    pub fn get_cluster(&self, player: usize, hand_idx: usize) -> u16 {
        self.assignments[player][hand_idx]
    }

    /// Get number of hands in a cluster
    #[inline]
    pub fn get_cluster_size(&self, player: usize, cluster_id: usize) -> usize {
        self.cluster_sizes[player][cluster_id]
    }
}

/// K-Means clustering implementation
pub fn kmeans_cluster(
    distributions: &[EquityDistribution],
    k: usize,
    max_iterations: usize,
) -> (Vec<usize>, Vec<EquityDistribution>) {
    let n = distributions.len();
    if n == 0 || k == 0 {
        return (vec![], vec![]);
    }

    let k = k.min(n); // Can't have more clusters than samples

    // Initialize centroids using K-Means++ style
    let mut centroids: Vec<EquityDistribution> = Vec::with_capacity(k);

    // First centroid: first element (deterministic for reproducibility)
    centroids.push(distributions[0].clone());

    // Remaining centroids: choose points far from existing centroids
    for _ in 1..k {
        let mut max_dist = 0.0f32;
        let mut best_idx = 0;

        for (i, dist) in distributions.iter().enumerate() {
            let min_centroid_dist = centroids
                .iter()
                .map(|c| dist.distance(c))
                .fold(f32::MAX, |a, b| a.min(b));

            if min_centroid_dist > max_dist {
                max_dist = min_centroid_dist;
                best_idx = i;
            }
        }

        centroids.push(distributions[best_idx].clone());
    }

    // Cluster assignments
    let mut assignments = vec![0usize; n];

    for _iteration in 0..max_iterations {
        // Assign each point to nearest centroid
        let mut changed = false;
        for (i, dist) in distributions.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_distance = f32::MAX;

            for (j, centroid) in centroids.iter().enumerate() {
                let d = dist.distance(centroid);
                if d < best_distance {
                    best_distance = d;
                    best_cluster = j;
                }
            }

            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        for centroid in &mut centroids {
            centroid.bins = [0.0; NUM_EQUITY_BINS];
        }
        let mut counts = vec![0usize; k];

        for (i, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            for b in 0..NUM_EQUITY_BINS {
                centroids[cluster].bins[b] += distributions[i].bins[b];
            }
        }

        for (j, centroid) in centroids.iter_mut().enumerate() {
            if counts[j] > 0 {
                for b in 0..NUM_EQUITY_BINS {
                    centroid.bins[b] /= counts[j] as f32;
                }
            }
        }
    }

    (assignments, centroids)
}

/// Compute equity of a hand on a specific runout
/// Returns value in [0, 1] where 1 = always wins, 0 = always loses
pub fn compute_hand_equity_on_runout(
    hand: (Card, Card),
    board: &[Card], // 5 cards: flop + turn + river
    opponent_hands: &[(Card, Card)],
    hand_strength_fn: impl Fn(&[Card; 7]) -> u16,
) -> f32 {
    let player_cards: [Card; 7] = [
        hand.0, hand.1,
        board[0], board[1], board[2], board[3], board[4],
    ];
    let player_strength = hand_strength_fn(&player_cards);

    let board_mask: u64 = board.iter().map(|&c| 1u64 << c).sum();
    let hand_mask: u64 = (1u64 << hand.0) | (1u64 << hand.1);
    let blocked = board_mask | hand_mask;

    let mut wins = 0u32;
    let mut total = 0u32;

    for &(c1, c2) in opponent_hands {
        let opp_mask = (1u64 << c1) | (1u64 << c2);
        if opp_mask & blocked != 0 {
            continue;
        }

        let opp_cards: [Card; 7] = [
            c1, c2,
            board[0], board[1], board[2], board[3], board[4],
        ];
        let opp_strength = hand_strength_fn(&opp_cards);

        if player_strength > opp_strength {
            wins += 2;
        } else if player_strength == opp_strength {
            wins += 1;
        }
        total += 2;
    }

    if total > 0 {
        wins as f32 / total as f32
    } else {
        0.5
    }
}
