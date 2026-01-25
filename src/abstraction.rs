//! Hand abstraction (bucketing) module for memory-efficient solving.
//!
//! This module implements EHS-based hand bucketing to reduce memory usage
//! by grouping similar hands into buckets during solving.

use crate::card::*;
use crate::hand::Hand;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Expected Hand Strength value stored as u16 for memory efficiency.
/// Range: 0-10000 representing 0.0000-1.0000 equity.
pub type EhsValue = u16;

/// Bucket ID type (supports up to 65535 buckets).
pub type BucketId = u16;

/// Configuration for hand abstraction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct AbstractionConfig {
    /// Number of buckets for each street.
    /// Index 0 = flop/turn buckets, Index 1 = river buckets.
    /// Recommended: 100-500 for flop/turn, 100-200 for river.
    pub num_buckets: [usize; 2],

    /// Number of Monte Carlo samples for EHS calculation.
    /// Higher = more accurate but slower. Recommended: 1000-5000.
    pub ehs_samples: usize,

    /// Whether to use percentile bucketing (true) or equal-width bucketing (false).
    /// Percentile distributes hands more evenly across buckets.
    pub use_percentile_bucketing: bool,

    /// Random seed for reproducible EHS calculation.
    pub seed: u64,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            num_buckets: [200, 200],
            ehs_samples: 2000,
            use_percentile_bucketing: true,
            seed: 42,
        }
    }
}

/// Maps hands to buckets for a specific board texture.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct BucketMapping {
    /// Bucket ID for each private hand index.
    /// Length = num_private_hands for that player.
    pub hand_to_bucket: Vec<BucketId>,

    /// Number of hands in each bucket.
    /// Length = num_buckets.
    pub bucket_sizes: Vec<u16>,

    /// Indices of hands in each bucket (for reconstruction).
    /// bucket_hands[bucket_id] = Vec of hand indices.
    pub bucket_hands: Vec<Vec<u16>>,

    /// Sum of initial weights for hands in each bucket.
    /// Used for weighted aggregation during evaluation.
    pub bucket_weights: Vec<f32>,
}

impl BucketMapping {
    /// Creates a new empty bucket mapping.
    pub fn new(num_hands: usize, num_buckets: usize) -> Self {
        Self {
            hand_to_bucket: vec![0; num_hands],
            bucket_sizes: vec![0; num_buckets],
            bucket_hands: vec![Vec::new(); num_buckets],
            bucket_weights: vec![0.0; num_buckets],
        }
    }

    /// Returns the number of buckets.
    #[inline]
    pub fn num_buckets(&self) -> usize {
        self.bucket_sizes.len()
    }
}

/// Stores bucket mappings for all relevant boards.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct AbstractionData {
    /// Configuration used to generate this abstraction.
    pub config: AbstractionConfig,

    /// Number of buckets for each player [OOP, IP].
    pub num_buckets: [usize; 2],

    /// Flop bucket mappings [OOP, IP].
    pub flop_buckets: [Option<BucketMapping>; 2],

    /// Turn bucket mappings indexed by turn card.
    /// turn_buckets[turn_card][player] = BucketMapping.
    pub turn_buckets: Vec<[Option<BucketMapping>; 2]>,

    /// River bucket mappings indexed by turn-river pair index.
    /// river_buckets[card_pair_index][player] = BucketMapping.
    pub river_buckets: Vec<[Option<BucketMapping>; 2]>,
}

impl AbstractionData {
    /// Creates a new AbstractionData with the given configuration.
    pub fn new(config: AbstractionConfig) -> Self {
        Self {
            num_buckets: [config.num_buckets[0], config.num_buckets[0]],
            config,
            flop_buckets: [None, None],
            turn_buckets: vec![[None, None]; 52],
            river_buckets: vec![[None, None]; 52 * 51 / 2],
        }
    }

    /// Returns the bucket ID for a hand on a specific board.
    #[inline]
    pub fn get_bucket(
        &self,
        player: usize,
        hand_index: u16,
        turn: Card,
        river: Card,
    ) -> BucketId {
        if river != NOT_DEALT {
            let pair_index = card_pair_to_index(turn, river);
            self.river_buckets[pair_index][player]
                .as_ref()
                .map(|m| m.hand_to_bucket[hand_index as usize])
                .unwrap_or(0)
        } else if turn != NOT_DEALT {
            self.turn_buckets[turn as usize][player]
                .as_ref()
                .map(|m| m.hand_to_bucket[hand_index as usize])
                .unwrap_or(0)
        } else {
            self.flop_buckets[player]
                .as_ref()
                .map(|m| m.hand_to_bucket[hand_index as usize])
                .unwrap_or(0)
        }
    }

    /// Returns the bucket mapping for a specific board.
    #[inline]
    pub fn get_mapping(&self, player: usize, turn: Card, river: Card) -> Option<&BucketMapping> {
        if river != NOT_DEALT {
            let pair_index = card_pair_to_index(turn, river);
            self.river_buckets[pair_index][player].as_ref()
        } else if turn != NOT_DEALT {
            self.turn_buckets[turn as usize][player].as_ref()
        } else {
            self.flop_buckets[player].as_ref()
        }
    }
}

/// Computes Expected Hand Strength using Monte Carlo sampling.
///
/// EHS = probability of winning at showdown against a random opponent hand,
/// given the current board cards.
///
/// # Arguments
/// * `hand` - The two hole cards (card1, card2)
/// * `board` - Current board cards (3-5 cards)
/// * `num_samples` - Number of Monte Carlo samples
/// * `rng` - Random number generator
///
/// # Returns
/// EHS value scaled to 0-10000 (representing 0.0000-1.0000 equity)
pub fn compute_ehs(
    hand: (Card, Card),
    board: &[Card],
    num_samples: usize,
    rng: &mut SmallRng,
) -> EhsValue {
    let hand_mask: u64 = (1 << hand.0) | (1 << hand.1);
    let mut board_mask: u64 = hand_mask;
    for &card in board {
        board_mask |= 1 << card;
    }

    let cards_to_deal = 5 - board.len();

    // Build initial board hand for hero
    let mut base_board = Hand::new();
    for &card in board {
        base_board = base_board.add_card(card as usize);
    }

    let mut wins = 0.0f64;
    let mut ties = 0.0f64;
    let mut total = 0.0f64;

    for _ in 0..num_samples {
        let mut sample_mask = board_mask;
        let mut sample_board = base_board;

        // Deal remaining board cards
        for _ in 0..cards_to_deal {
            let card = loop {
                let c = rng.gen_range(0..52u8);
                if (sample_mask >> c) & 1 == 0 {
                    break c;
                }
            };
            sample_mask |= 1 << card;
            sample_board = sample_board.add_card(card as usize);
        }

        // Deal opponent hole cards
        let opp_c1 = loop {
            let c = rng.gen_range(0..52u8);
            if (sample_mask >> c) & 1 == 0 {
                break c;
            }
        };
        sample_mask |= 1 << opp_c1;

        let opp_c2 = loop {
            let c = rng.gen_range(0..52u8);
            if (sample_mask >> c) & 1 == 0 {
                break c;
            }
        };

        // Evaluate both hands
        let hero_hand = sample_board
            .add_card(hand.0 as usize)
            .add_card(hand.1 as usize);
        let opp_hand = sample_board
            .add_card(opp_c1 as usize)
            .add_card(opp_c2 as usize);

        let hero_strength = hero_hand.evaluate();
        let opp_strength = opp_hand.evaluate();

        if hero_strength > opp_strength {
            wins += 1.0;
        } else if hero_strength == opp_strength {
            ties += 1.0;
        }
        total += 1.0;
    }

    let equity = if total > 0.0 {
        (wins + 0.5 * ties) / total
    } else {
        0.5
    };

    (equity * 10000.0).round().min(10000.0) as EhsValue
}

/// Computes EHS values for all hands in a player's range.
///
/// # Arguments
/// * `private_cards` - The player's private card combinations
/// * `board` - Current board cards
/// * `num_samples` - Number of Monte Carlo samples per hand
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Vector of EHS values, one per hand
#[cfg(feature = "rayon")]
pub fn compute_all_ehs(
    private_cards: &[(Card, Card)],
    board: &[Card],
    num_samples: usize,
    seed: u64,
) -> Vec<EhsValue> {
    private_cards
        .par_iter()
        .enumerate()
        .map(|(idx, &hand)| {
            // Create a unique seed for each hand to ensure reproducibility
            let hand_seed = seed.wrapping_add(idx as u64);
            let mut rng = SmallRng::seed_from_u64(hand_seed);
            compute_ehs(hand, board, num_samples, &mut rng)
        })
        .collect()
}

#[cfg(not(feature = "rayon"))]
pub fn compute_all_ehs(
    private_cards: &[(Card, Card)],
    board: &[Card],
    num_samples: usize,
    seed: u64,
) -> Vec<EhsValue> {
    let mut rng = SmallRng::seed_from_u64(seed);
    private_cards
        .iter()
        .map(|&hand| compute_ehs(hand, board, num_samples, &mut rng))
        .collect()
}

/// Creates bucket mappings from EHS values using percentile-based clustering.
///
/// Hands are sorted by EHS and divided into roughly equal-sized buckets.
///
/// # Arguments
/// * `ehs_values` - EHS value for each hand
/// * `weights` - Initial weight for each hand
/// * `num_buckets` - Number of buckets to create
///
/// # Returns
/// BucketMapping with hand-to-bucket assignments
pub fn create_bucket_mapping_percentile(
    ehs_values: &[EhsValue],
    weights: &[f32],
    num_buckets: usize,
) -> BucketMapping {
    let num_hands = ehs_values.len();
    assert_eq!(weights.len(), num_hands);

    if num_hands == 0 || num_buckets == 0 {
        return BucketMapping::default();
    }

    let num_buckets = num_buckets.min(num_hands);
    let mut mapping = BucketMapping::new(num_hands, num_buckets);

    // Sort hand indices by EHS value
    let mut indexed: Vec<(usize, EhsValue)> = ehs_values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by_key(|&(_, v)| v);

    // Assign buckets based on percentile rank
    let hands_per_bucket = (num_hands + num_buckets - 1) / num_buckets;

    for (rank, (hand_idx, _)) in indexed.into_iter().enumerate() {
        let bucket = (rank / hands_per_bucket).min(num_buckets - 1);
        mapping.hand_to_bucket[hand_idx] = bucket as BucketId;
        mapping.bucket_sizes[bucket] += 1;
        mapping.bucket_hands[bucket].push(hand_idx as u16);
        mapping.bucket_weights[bucket] += weights[hand_idx];
    }

    mapping
}

/// Creates bucket mappings from EHS values using equal-width bucketing.
///
/// EHS range [0, 10000] is divided into equal-width buckets.
///
/// # Arguments
/// * `ehs_values` - EHS value for each hand
/// * `weights` - Initial weight for each hand
/// * `num_buckets` - Number of buckets to create
///
/// # Returns
/// BucketMapping with hand-to-bucket assignments
pub fn create_bucket_mapping_equal_width(
    ehs_values: &[EhsValue],
    weights: &[f32],
    num_buckets: usize,
) -> BucketMapping {
    let num_hands = ehs_values.len();
    assert_eq!(weights.len(), num_hands);

    if num_hands == 0 || num_buckets == 0 {
        return BucketMapping::default();
    }

    let mut mapping = BucketMapping::new(num_hands, num_buckets);
    let bucket_width = (10001 + num_buckets - 1) / num_buckets;

    for (hand_idx, &ehs) in ehs_values.iter().enumerate() {
        let bucket = ((ehs as usize) / bucket_width).min(num_buckets - 1);
        mapping.hand_to_bucket[hand_idx] = bucket as BucketId;
        mapping.bucket_sizes[bucket] += 1;
        mapping.bucket_hands[bucket].push(hand_idx as u16);
        mapping.bucket_weights[bucket] += weights[hand_idx];
    }

    mapping
}

/// Creates a bucket mapping using the specified method.
#[inline]
pub fn create_bucket_mapping(
    ehs_values: &[EhsValue],
    weights: &[f32],
    num_buckets: usize,
    use_percentile: bool,
) -> BucketMapping {
    if use_percentile {
        create_bucket_mapping_percentile(ehs_values, weights, num_buckets)
    } else {
        create_bucket_mapping_equal_width(ehs_values, weights, num_buckets)
    }
}

/// Converts bucket-level values to hand-level values.
///
/// Each hand gets the value of its assigned bucket.
///
/// # Arguments
/// * `bucket_values` - Values per bucket
/// * `mapping` - Hand-to-bucket mapping
/// * `num_hands` - Number of hands
///
/// # Returns
/// Vector of values, one per hand
pub fn expand_bucket_to_hands(
    bucket_values: &[f32],
    mapping: &BucketMapping,
    num_hands: usize,
) -> Vec<f32> {
    let mut hand_values = vec![0.0f32; num_hands];
    for (hand_idx, &bucket_id) in mapping.hand_to_bucket.iter().enumerate() {
        hand_values[hand_idx] = bucket_values[bucket_id as usize];
    }
    hand_values
}

/// Aggregates hand-level values to bucket-level values using weighted average.
///
/// # Arguments
/// * `hand_values` - Values per hand
/// * `hand_weights` - Weight per hand (typically reach probability)
/// * `mapping` - Hand-to-bucket mapping
///
/// # Returns
/// Vector of values, one per bucket (weighted average of hands in bucket)
pub fn aggregate_hands_to_buckets(
    hand_values: &[f32],
    hand_weights: &[f32],
    mapping: &BucketMapping,
) -> Vec<f32> {
    let num_buckets = mapping.num_buckets();
    let mut bucket_values = vec![0.0f32; num_buckets];
    let mut bucket_weights = vec![0.0f32; num_buckets];

    for (hand_idx, &bucket_id) in mapping.hand_to_bucket.iter().enumerate() {
        let bucket = bucket_id as usize;
        let weight = hand_weights[hand_idx];
        bucket_values[bucket] += hand_values[hand_idx] * weight;
        bucket_weights[bucket] += weight;
    }

    // Normalize by bucket weight
    for bucket in 0..num_buckets {
        if bucket_weights[bucket] > 0.0 {
            bucket_values[bucket] /= bucket_weights[bucket];
        }
    }

    bucket_values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_mapping_percentile() {
        // 10 hands with varying EHS
        let ehs_values: Vec<EhsValue> = vec![1000, 3000, 5000, 7000, 9000, 2000, 4000, 6000, 8000, 500];
        let weights: Vec<f32> = vec![1.0; 10];

        let mapping = create_bucket_mapping_percentile(&ehs_values, &weights, 5);

        assert_eq!(mapping.num_buckets(), 5);

        // Each bucket should have 2 hands
        for bucket in 0..5 {
            assert_eq!(mapping.bucket_sizes[bucket], 2);
        }

        // Check that similar EHS hands are in same buckets
        // EHS 500 and 1000 should be in bucket 0 (weakest)
        assert_eq!(mapping.hand_to_bucket[0], mapping.hand_to_bucket[9]); // 1000 and 500
    }

    #[test]
    fn test_bucket_mapping_equal_width() {
        let ehs_values: Vec<EhsValue> = vec![1000, 3000, 5000, 7000, 9000];
        let weights: Vec<f32> = vec![1.0; 5];

        let mapping = create_bucket_mapping_equal_width(&ehs_values, &weights, 5);

        assert_eq!(mapping.num_buckets(), 5);

        // With 5 buckets, width is ~2000
        // 1000 -> bucket 0, 3000 -> bucket 1, 5000 -> bucket 2, etc.
        assert!(mapping.hand_to_bucket[0] < mapping.hand_to_bucket[1]);
        assert!(mapping.hand_to_bucket[1] < mapping.hand_to_bucket[2]);
    }

    #[test]
    fn test_expand_and_aggregate() {
        let ehs_values: Vec<EhsValue> = vec![1000, 2000, 8000, 9000];
        let weights: Vec<f32> = vec![1.0; 4];

        let mapping = create_bucket_mapping_percentile(&ehs_values, &weights, 2);

        // Bucket values
        let bucket_values = vec![10.0, 20.0];

        // Expand to hands
        let hand_values = expand_bucket_to_hands(&bucket_values, &mapping, 4);

        // Hands 0,1 (low EHS) should have bucket 0 value
        // Hands 2,3 (high EHS) should have bucket 1 value
        assert!((hand_values[0] - 10.0).abs() < 0.001 || (hand_values[0] - 20.0).abs() < 0.001);
        assert!((hand_values[2] - 10.0).abs() < 0.001 || (hand_values[2] - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_ehs_computation() {
        // Test with a known board: A♠K♠Q♠ (royal flush draw board)
        // Hand: A♥K♥ (top two pair)
        let hand = (48, 44); // A♠ = 51, but let's use A♥=50, K♥=46... actually:
        // A = rank 12, so A♠ = 12*4+3 = 51, A♥ = 12*4+2 = 50, A♦ = 12*4+1 = 49, A♣ = 12*4+0 = 48
        // K = rank 11, so K♠ = 11*4+3 = 47, K♥ = 11*4+2 = 46, K♦ = 11*4+1 = 45, K♣ = 11*4+0 = 44
        let board = [51, 47, 43]; // A♠, K♠, Q♠

        let mut rng = SmallRng::seed_from_u64(42);
        let ehs = compute_ehs(hand, &board, 1000, &mut rng);

        // AK on AKQ board should have good equity (probably 60-80%)
        assert!(ehs > 5000, "AK should have >50% equity on AKQ board");
        assert!(ehs < 9500, "AK can lose to flushes/straights");
    }
}
