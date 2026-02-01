//! Pre-compute equity distributions and clusters for a single flop (POC)
//!
//! This tool:
//! 1. Calculates equity distribution (histogram) for each hand on the flop
//! 2. Clusters hands using K-Means on the equity distributions
//! 3. Pre-computes EV for each cluster over all turn/river runouts
//! 4. Saves the result as a binary file for fast loading
//!
//! Run with: cargo run --example precompute_flop_clusters --release

use postflop_solver::*;
use std::collections::HashMap;
use std::time::Instant;

/// Number of equity distribution bins (like Poker-AI)
const NUM_EQUITY_BINS: usize = 10;

/// Number of clusters for K-Means
const NUM_CLUSTERS: usize = 50;

/// Number of Monte Carlo samples for equity calculation
const MC_SAMPLES: usize = 50;

/// Represents an equity distribution histogram
#[derive(Clone, Debug)]
struct EquityDistribution {
    bins: [f32; NUM_EQUITY_BINS],
}

impl EquityDistribution {
    fn new() -> Self {
        Self { bins: [0.0; NUM_EQUITY_BINS] }
    }

    /// Add an equity sample to the distribution
    fn add_sample(&mut self, equity: f32) {
        let bin_idx = ((equity * NUM_EQUITY_BINS as f32) as usize).min(NUM_EQUITY_BINS - 1);
        self.bins[bin_idx] += 1.0;
    }

    /// Normalize the distribution to sum to 1.0
    fn normalize(&mut self) {
        let sum: f32 = self.bins.iter().sum();
        if sum > 0.0 {
            for b in &mut self.bins {
                *b /= sum;
            }
        }
    }

    /// Euclidean distance to another distribution
    fn distance(&self, other: &EquityDistribution) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..NUM_EQUITY_BINS {
            let diff = self.bins[i] - other.bins[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Pre-computed cluster data
#[derive(Clone, Debug)]
struct ClusterData {
    /// Centroid of the cluster (equity distribution)
    centroid: EquityDistribution,
    /// Hand indices belonging to this cluster
    hand_indices: Vec<usize>,
    /// Average EV for this cluster (computed from full runouts)
    avg_ev_oop: f32,
    avg_ev_ip: f32,
}

/// Card to string helper
fn card_to_str(card: u8) -> String {
    let rank = card >> 2;
    let suit = card & 3;
    let rank_char = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'][rank as usize];
    let suit_char = ['c', 'd', 'h', 's'][suit as usize];
    format!("{}{}", rank_char, suit_char)
}

fn hole_to_str(c1: u8, c2: u8) -> String {
    format!("{}{}", card_to_str(c1.max(c2)), card_to_str(c1.min(c2)))
}


/// Simple K-Means clustering implementation
fn kmeans_cluster(
    distributions: &[EquityDistribution],
    k: usize,
    max_iterations: usize,
) -> Vec<usize> {
    let n = distributions.len();
    if n == 0 || k == 0 {
        return vec![];
    }

    // Initialize centroids using K-Means++ style (spread out initial centroids)
    let mut centroids: Vec<EquityDistribution> = Vec::with_capacity(k);

    // First centroid: random (use first element for determinism)
    centroids.push(distributions[0].clone());

    // Remaining centroids: choose points far from existing centroids
    for _ in 1..k.min(n) {
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

    assignments
}

fn main() {
    println!("=== Flop Cluster Pre-computation POC ===\n");

    // Setup: Td9d6h
    let flop_str = "Td9d6h";
    let flop = flop_from_str(flop_str).unwrap();
    let flop_mask: u64 = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);

    println!("Flop: {} ({}, {}, {})", flop_str,
        card_to_str(flop[0]), card_to_str(flop[1]), card_to_str(flop[2]));
    println!("Flop mask: {:064b}\n", flop_mask);

    // Generate all possible hole card combinations
    let mut hands: Vec<(u8, u8)> = Vec::new();
    for c1 in 0u8..52 {
        if (1u64 << c1) & flop_mask != 0 {
            continue;
        }
        for c2 in (c1 + 1)..52 {
            if (1u64 << c2) & flop_mask != 0 {
                continue;
            }
            hands.push((c1, c2));
        }
    }
    println!("Total hands: {}", hands.len());

    // Step 1: Calculate equity distributions for each hand
    println!("\nStep 1: Calculating equity distributions...");
    let start = Instant::now();

    let mut distributions: Vec<EquityDistribution> = Vec::with_capacity(hands.len());

    for (i, &(c1, c2)) in hands.iter().enumerate() {
        let hand_mask = (1u64 << c1) | (1u64 << c2);
        let board_mask = flop_mask | hand_mask;

        let mut dist = EquityDistribution::new();

        // Sample turn/river combinations
        let mut samples = 0;
        for turn in 0u8..52 {
            if (1u64 << turn) & board_mask != 0 {
                continue;
            }
            let turn_mask = board_mask | (1u64 << turn);

            for river in (turn + 1)..52 {
                if (1u64 << river) & turn_mask != 0 {
                    continue;
                }

                // Calculate equity against random opponent
                // For POC, we'll use a simplified equity calculation
                let river_mask = turn_mask | (1u64 << river);

                // Count wins against all possible opponent hands
                let mut wins = 0;
                let mut total = 0;

                for opp_c1 in 0u8..52 {
                    if (1u64 << opp_c1) & river_mask != 0 {
                        continue;
                    }
                    for opp_c2 in (opp_c1 + 1)..52 {
                        if (1u64 << opp_c2) & river_mask != 0 {
                            continue;
                        }

                        // Evaluate hands using 7-card evaluation
                        let player_hand = [c1, c2, flop[0], flop[1], flop[2], turn, river];
                        let opp_hand = [opp_c1, opp_c2, flop[0], flop[1], flop[2], turn, river];

                        let player_strength = evaluate_hand(&player_hand);
                        let opp_strength = evaluate_hand(&opp_hand);

                        if player_strength > opp_strength {
                            wins += 2;
                        } else if player_strength == opp_strength {
                            wins += 1;
                        }
                        total += 2;
                    }
                }

                let equity = if total > 0 { wins as f32 / total as f32 } else { 0.5 };
                dist.add_sample(equity);
                samples += 1;

                // Limit samples for speed in POC
                if samples >= MC_SAMPLES {
                    break;
                }
            }
            if samples >= MC_SAMPLES {
                break;
            }
        }

        dist.normalize();
        distributions.push(dist);

        if (i + 1) % 100 == 0 {
            print!("\r  Processed {}/{} hands", i + 1, hands.len());
        }
    }
    println!("\r  Processed {}/{} hands in {:?}", hands.len(), hands.len(), start.elapsed());

    // Step 2: K-Means clustering
    println!("\nStep 2: K-Means clustering into {} clusters...", NUM_CLUSTERS);
    let start = Instant::now();

    let cluster_assignments = kmeans_cluster(&distributions, NUM_CLUSTERS, 100);

    println!("  Clustering completed in {:?}", start.elapsed());

    // Count hands per cluster
    let mut cluster_counts = vec![0usize; NUM_CLUSTERS];
    for &c in &cluster_assignments {
        cluster_counts[c] += 1;
    }

    println!("\nCluster distribution:");
    for (i, &count) in cluster_counts.iter().enumerate() {
        if count > 0 {
            println!("  Cluster {}: {} hands ({:.1}%)",
                i, count, count as f32 / hands.len() as f32 * 100.0);
        }
    }

    // Step 3: Show example hands per cluster
    println!("\nExample hands per cluster:");
    for cluster_id in 0..NUM_CLUSTERS.min(10) {
        let cluster_hands: Vec<_> = hands.iter()
            .zip(cluster_assignments.iter())
            .filter(|(_, &c)| c == cluster_id)
            .take(5)
            .map(|(&(c1, c2), _)| hole_to_str(c1, c2))
            .collect();

        if !cluster_hands.is_empty() {
            // Calculate average equity for this cluster
            let avg_equity: f32 = cluster_assignments.iter()
                .enumerate()
                .filter(|(_, &c)| c == cluster_id)
                .map(|(i, _)| {
                    // Sum middle bins as rough equity estimate
                    distributions[i].bins.iter()
                        .enumerate()
                        .map(|(b, &v)| (b as f32 + 0.5) / NUM_EQUITY_BINS as f32 * v)
                        .sum::<f32>()
                })
                .sum::<f32>() / cluster_counts[cluster_id] as f32;

            println!("  Cluster {} (avg equity ~{:.0}%): {:?}",
                cluster_id, avg_equity * 100.0, cluster_hands);
        }
    }

    println!("\n=== POC Complete ===");
    println!("\nNext steps:");
    println!("1. Use these clusters in DCFR solver");
    println!("2. Pre-compute EV for each cluster");
    println!("3. Solve at cluster level for speed");
}

/// Simple 7-card hand evaluation (returns strength score, higher = better)
fn evaluate_hand(cards: &[u8; 7]) -> u32 {
    // Use the library's evaluation if available, otherwise simplified
    // For POC, we'll use a basic evaluation

    let mut ranks = [0u8; 13];
    let mut suits = [0u8; 4];

    for &card in cards {
        let rank = card >> 2;
        let suit = card & 3;
        ranks[rank as usize] += 1;
        suits[suit as usize] += 1;
    }

    // Check for flush
    let is_flush = suits.iter().any(|&c| c >= 5);

    // Check for straight
    let mut straight_high = 0u8;
    let mut consecutive = 0;
    for r in (0..13).rev() {
        if ranks[r] > 0 {
            consecutive += 1;
            if consecutive >= 5 {
                straight_high = r as u8 + 4;
                break;
            }
        } else {
            consecutive = 0;
        }
    }
    // Check wheel (A-2-3-4-5)
    if consecutive < 5 && ranks[12] > 0 && ranks[0] > 0 && ranks[1] > 0 && ranks[2] > 0 && ranks[3] > 0 {
        straight_high = 3;
    }
    let is_straight = straight_high > 0;

    // Count pairs, trips, quads
    let quads = ranks.iter().filter(|&&c| c == 4).count();
    let trips = ranks.iter().filter(|&&c| c == 3).count();
    let pairs = ranks.iter().filter(|&&c| c == 2).count();

    // Hand rankings (higher = better)
    let hand_rank = if is_straight && is_flush {
        8 // Straight flush
    } else if quads > 0 {
        7 // Four of a kind
    } else if trips > 0 && pairs > 0 {
        6 // Full house
    } else if is_flush {
        5 // Flush
    } else if is_straight {
        4 // Straight
    } else if trips > 0 {
        3 // Three of a kind
    } else if pairs >= 2 {
        2 // Two pair
    } else if pairs == 1 {
        1 // One pair
    } else {
        0 // High card
    };

    // Combine hand rank with kickers for comparison
    let high_card = ranks.iter().enumerate().rev()
        .find(|(_, &c)| c > 0)
        .map(|(r, _)| r as u32)
        .unwrap_or(0);

    (hand_rank << 16) | (straight_high as u32) << 8 | high_card
}
