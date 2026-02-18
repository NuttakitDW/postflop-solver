use crate::card::*;
use crate::hand::*;

/// Sentinel bucket ID for hands blocked by the board.
pub const BLOCKED_BUCKET: u16 = u16::MAX;

/// Default number of clusters.
pub const DEFAULT_K: usize = 1000;

/// Result of hand bucketing: maps each of 1326 combos to a bucket ID.
pub struct BucketMapping {
    /// Maps combo index (0..1326) to bucket ID (0..k-1), or [`BLOCKED_BUCKET`] if blocked.
    pub hand_to_bucket: [u16; 1326],
    /// Number of clusters actually used (may be less than requested if K > num_valid).
    pub k: usize,
    /// Number of valid (non-blocked) hands.
    pub num_valid: usize,
}

/// Number of clustering features (matches `docs/feature_input.csv`).
const NUM_FEATURES: usize = 15;

/// Feature weights for weighted K-means L2 distance.
///
/// Order: equity, made_hand_rank, is_pair, is_suited, rank_gap,
///        suit_match_c1, suit_match_c2, pairs_with_board, overcards,
///        has_flush_draw, is_nut_flush_draw, has_made_flush,
///        straight_draw_outs, has_made_straight,
///        is_blocked (always 0 for valid hands — included for spec compliance)
const FEATURE_WEIGHTS: [f32; NUM_FEATURES] = [
    3.0, 2.0, 0.5, 0.5, 0.5, // equity, made_hand_rank, is_pair, is_suited, rank_gap
    0.5, 0.5, 0.5, 0.5,       // suit_match_c1, suit_match_c2, pairs_with_board, overcards
    1.5, 2.0, 2.0,             // has_flush_draw, is_nut_flush_draw, has_made_flush
    1.0, 2.0,                  // straight_draw_outs, has_made_straight
    0.0,                       // is_blocked
];

const MAX_KMEANS_ITER: usize = 50;

// ---------------------------------------------------------------------------
// Equity
// ---------------------------------------------------------------------------

/// Compute equity for all 1326 combos on a 4-card turn board.
///
/// Returns equity in \[0, 1\] for valid hands, 0.0 for blocked hands.
/// Equity is averaged over all valid river cards (ignoring card removal between opponents).
pub fn compute_equity(board: &[Card; 4]) -> [f32; 1326] {
    let mut equity = [0.0f32; 1326];
    let mut river_count = [0u16; 1326];
    let board_mask: u64 = board.iter().fold(0u64, |acc, &c| acc | (1u64 << c));

    let base_hand = Hand::new()
        .add_card(board[0] as usize)
        .add_card(board[1] as usize)
        .add_card(board[2] as usize)
        .add_card(board[3] as usize);

    for river in 0u8..52 {
        if board_mask & (1u64 << river) != 0 {
            continue;
        }

        let full_mask = board_mask | (1u64 << river);
        let river_hand = base_hand.add_card(river as usize);

        // Evaluate all valid combos for this river
        let mut hands: Vec<(u16, u16)> = Vec::with_capacity(1128);
        for combo_idx in 0..1326u16 {
            let (c1, c2) = index_to_card_pair(combo_idx as usize);
            let hand_mask: u64 = (1u64 << c1) | (1u64 << c2);
            if hand_mask & full_mask != 0 {
                continue;
            }
            let strength = river_hand
                .add_card(c1 as usize)
                .add_card(c2 as usize)
                .evaluate();
            hands.push((strength, combo_idx));
        }

        hands.sort_unstable_by_key(|&(s, _)| s);

        let n = hands.len();
        if n <= 1 {
            continue;
        }
        let denom = (n - 1) as f32;

        // Prefix-sum grouping: hands with equal strength tie with each other
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && hands[j].0 == hands[i].0 {
                j += 1;
            }
            let wins_below = i as f32;
            let ties = (j - i - 1) as f32;
            let eq = (wins_below + 0.5 * ties) / denom;

            for hand in hands.iter().take(j).skip(i) {
                let idx = hand.1 as usize;
                equity[idx] += eq;
                river_count[idx] += 1;
            }
            i = j;
        }
    }

    // Average over valid rivers
    for i in 0..1326 {
        if river_count[i] > 0 {
            equity[i] /= river_count[i] as f32;
        }
    }

    equity
}

// ---------------------------------------------------------------------------
// Hand-feature helpers (duplicated from oracle.rs to avoid #[cfg(feature = "onnx")])
// ---------------------------------------------------------------------------

fn check_straight(rank_bits: u16) -> bool {
    for start in 0..9 {
        let mask = 0b11111u16 << start;
        if rank_bits & mask == mask {
            return true;
        }
    }
    // Wheel: A-2-3-4-5
    let wheel = (1u16 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    rank_bits & wheel == wheel
}

fn count_straight_draw_outs(c1: Card, c2: Card, board: &[Card; 4]) -> u8 {
    let mut rank_bits = 0u16;
    for &c in [c1, c2].iter().chain(board.iter()) {
        rank_bits |= 1 << (c >> 2);
    }
    if check_straight(rank_bits) {
        return 0;
    }
    let mut outs = 0u8;
    for rank in 0u8..13 {
        if rank_bits & (1 << rank) != 0 {
            continue;
        }
        if check_straight(rank_bits | (1 << rank)) {
            outs += 4;
        }
    }
    outs
}

/// Returns (has_flush_draw, is_nut_flush_draw, has_made_flush).
/// Only considers flushes where the hand contributes at least one card.
fn compute_flush_features(c1: Card, c2: Card, board: &[Card; 4]) -> (bool, bool, bool) {
    let s1 = c1 & 3;
    let s2 = c2 & 3;

    let mut suit_counts = [0u8; 4];
    let mut hand_suit_counts = [0u8; 4];

    suit_counts[s1 as usize] += 1;
    suit_counts[s2 as usize] += 1;
    hand_suit_counts[s1 as usize] += 1;
    hand_suit_counts[s2 as usize] += 1;

    for &bc in board.iter() {
        suit_counts[(bc & 3) as usize] += 1;
    }

    let mut has_flush_draw = false;
    let mut has_made_flush = false;
    let mut is_nut_flush_draw = false;

    for suit in 0..4u8 {
        if hand_suit_counts[suit as usize] == 0 {
            continue;
        }
        let total = suit_counts[suit as usize];
        if total >= 5 {
            has_made_flush = true;
        } else if total == 4 {
            has_flush_draw = true;

            // Nut flush draw: hand holds the highest non-board card of this suit
            let hand_max_rank_in_suit = {
                let mut max_r = -1i8;
                if s1 == suit {
                    max_r = max_r.max((c1 >> 2) as i8);
                }
                if s2 == suit {
                    max_r = max_r.max((c2 >> 2) as i8);
                }
                max_r
            };

            let mut highest_non_board = -1i8;
            for r in (0u8..13).rev() {
                let card = (r << 2) | suit;
                if !board.contains(&card) {
                    highest_non_board = r as i8;
                    break;
                }
            }

            if hand_max_rank_in_suit == highest_non_board {
                is_nut_flush_draw = true;
            }
        }
    }

    (has_flush_draw, is_nut_flush_draw, has_made_flush)
}

fn compute_made_hand_rank(c1: Card, c2: Card, board: &[Card; 4]) -> f32 {
    let all_cards = [c1, c2, board[0], board[1], board[2], board[3]];

    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    let mut rank_bits = 0u16;

    for &c in &all_cards {
        rank_counts[(c >> 2) as usize] += 1;
        suit_counts[(c & 3) as usize] += 1;
        rank_bits |= 1 << (c >> 2);
    }

    let has_flush = suit_counts.iter().any(|&c| c >= 5);
    let has_straight = check_straight(rank_bits);

    if has_flush && has_straight {
        for suit in 0..4u8 {
            if suit_counts[suit as usize] >= 5 {
                let mut suit_rank_bits = 0u16;
                for &c in &all_cards {
                    if (c & 3) == suit {
                        suit_rank_bits |= 1 << (c >> 2);
                    }
                }
                if check_straight(suit_rank_bits) {
                    return 1.0;
                }
            }
        }
    }

    if rank_counts.iter().any(|&c| c >= 4) {
        return 0.9;
    }

    let trips_count = rank_counts.iter().filter(|&&c| c >= 3).count();
    let pair_or_better_count = rank_counts.iter().filter(|&&c| c >= 2).count();
    if trips_count >= 1 && pair_or_better_count >= 2 {
        return 0.8;
    }

    if has_flush {
        return 0.7;
    }
    if has_straight {
        return 0.6;
    }
    if trips_count >= 1 {
        return 0.5;
    }

    let pair_count = rank_counts.iter().filter(|&&c| c == 2).count();
    if pair_count >= 2 {
        return 0.4;
    }
    if pair_count >= 1 {
        return 0.2;
    }

    0.0 // high card
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

fn compute_features(
    board: &[Card; 4],
    equity: &[f32; 1326],
) -> (Vec<[f32; NUM_FEATURES]>, Vec<u16>) {
    let board_mask: u64 = board.iter().fold(0u64, |acc, &c| acc | (1u64 << c));
    let board_ranks: [u8; 4] = [board[0] >> 2, board[1] >> 2, board[2] >> 2, board[3] >> 2];
    let max_board_rank = *board_ranks.iter().max().unwrap();

    let mut features = Vec::with_capacity(1128);
    let mut valid_indices = Vec::with_capacity(1128);

    for combo_idx in 0..1326u16 {
        let (c1, c2) = index_to_card_pair(combo_idx as usize);
        let hand_mask: u64 = (1u64 << c1) | (1u64 << c2);
        if hand_mask & board_mask != 0 {
            continue;
        }

        let r1 = c1 >> 2;
        let r2 = c2 >> 2;
        let s1 = c1 & 3;
        let s2 = c2 & 3;

        // 0: equity
        let f_equity = equity[combo_idx as usize];

        // 1: made_hand_rank
        let f_made_rank = compute_made_hand_rank(c1, c2, board);

        // 2: is_pair
        let f_is_pair = if r1 == r2 { 1.0f32 } else { 0.0 };

        // 3: is_suited
        let f_is_suited = if s1 == s2 { 1.0f32 } else { 0.0 };

        // 4: rank_gap = |r1 - r2| / 12
        let gap = if r1 > r2 { r1 - r2 } else { r2 - r1 };
        let f_rank_gap = gap as f32 / 12.0;

        // 5: suit_match_c1 = board cards matching card1 suit / 4
        let count_s1 = board.iter().filter(|&&c| (c & 3) == s1).count();
        let f_suit_match_c1 = count_s1 as f32 / 4.0;

        // 6: suit_match_c2 = board cards matching card2 suit / 4
        let count_s2 = board.iter().filter(|&&c| (c & 3) == s2).count();
        let f_suit_match_c2 = count_s2 as f32 / 4.0;

        // 7: pairs_with_board = count of board cards pairing hand / 4
        let mut pair_count = 0u8;
        for &br in &board_ranks {
            if br == r1 || br == r2 {
                pair_count += 1;
            }
        }
        let f_pairs_with_board = pair_count as f32 / 4.0;

        // 8: overcards = hand cards above highest board card / 2
        let mut overcards_count = 0u8;
        if r1 > max_board_rank {
            overcards_count += 1;
        }
        if r2 > max_board_rank {
            overcards_count += 1;
        }
        let f_overcards = overcards_count as f32 / 2.0;

        // 9-11: flush features
        let (has_flush_draw, is_nut_flush_draw, has_made_flush) =
            compute_flush_features(c1, c2, board);
        let f_has_flush_draw = if has_flush_draw { 1.0f32 } else { 0.0 };
        let f_is_nut_flush_draw = if is_nut_flush_draw { 1.0f32 } else { 0.0 };
        let f_has_made_flush = if has_made_flush { 1.0f32 } else { 0.0 };

        // 12: straight_draw_outs / 8 capped at 1
        let straight_outs = count_straight_draw_outs(c1, c2, board);
        let f_straight_draw_outs = (straight_outs as f32 / 8.0).min(1.0);

        // 13: has_made_straight
        let f_has_made_straight = {
            let mut rank_bits = 0u16;
            for &c in [c1, c2].iter().chain(board.iter()) {
                rank_bits |= 1 << (c >> 2);
            }
            if check_straight(rank_bits) { 1.0f32 } else { 0.0 }
        };

        // 14: is_blocked (always 0 — blocked hands are excluded above)
        let f_is_blocked = 0.0f32;

        features.push([
            f_equity,
            f_made_rank,
            f_is_pair,
            f_is_suited,
            f_rank_gap,
            f_suit_match_c1,
            f_suit_match_c2,
            f_pairs_with_board,
            f_overcards,
            f_has_flush_draw,
            f_is_nut_flush_draw,
            f_has_made_flush,
            f_straight_draw_outs,
            f_has_made_straight,
            f_is_blocked,
        ]);
        valid_indices.push(combo_idx);
    }

    (features, valid_indices)
}

// ---------------------------------------------------------------------------
// K-means
// ---------------------------------------------------------------------------

fn board_seed(board: &[Card; 4]) -> u64 {
    let mut seed = 0u64;
    for &c in board {
        seed = seed.wrapping_mul(53).wrapping_add(c as u64 + 1);
    }
    if seed == 0 {
        seed = 1;
    }
    seed
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn weighted_l2_sq(a: &[f32; NUM_FEATURES], b: &[f32; NUM_FEATURES]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..NUM_FEATURES {
        let d = a[i] - b[i];
        sum += FEATURE_WEIGHTS[i] * d * d;
    }
    sum
}

fn kmeans_pp_init(features: &[[f32; NUM_FEATURES]], k: usize, rng: &mut u64) -> Vec<[f32; NUM_FEATURES]> {
    let n = features.len();
    let mut centroids = Vec::with_capacity(k);

    let idx = (xorshift64(rng) as usize) % n;
    centroids.push(features[idx]);

    let mut distances = vec![f32::MAX; n];

    for _ in 1..k {
        let last = centroids.last().unwrap();
        for i in 0..n {
            let d = weighted_l2_sq(&features[i], last);
            distances[i] = distances[i].min(d);
        }

        let total: f64 = distances.iter().map(|&d| d as f64).sum();
        if total <= 0.0 {
            let idx = (xorshift64(rng) as usize) % n;
            centroids.push(features[idx]);
            continue;
        }

        let threshold = (xorshift64(rng) as f64 / u64::MAX as f64) * total;
        let mut cumulative = 0.0f64;
        let mut chosen = n - 1;
        for i in 0..n {
            cumulative += distances[i] as f64;
            if cumulative >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(features[chosen]);
    }

    centroids
}

fn lloyd(features: &[[f32; NUM_FEATURES]], centroids: &mut [[f32; NUM_FEATURES]], max_iter: usize) -> Vec<u16> {
    let n = features.len();
    let k = centroids.len();
    let mut assignments = vec![0u16; n];

    for _ in 0..max_iter {
        let mut changed = false;

        for i in 0..n {
            let mut best = 0u16;
            let mut best_dist = f32::MAX;
            for j in 0..k {
                let d = weighted_l2_sq(&features[i], &centroids[j]);
                if d < best_dist {
                    best_dist = d;
                    best = j as u16;
                }
            }
            if assignments[i] != best {
                changed = true;
                assignments[i] = best;
            }
        }

        if !changed {
            break;
        }

        // Recompute centroids
        let mut sums = vec![[0.0f64; NUM_FEATURES]; k];
        let mut counts = vec![0usize; k];

        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for f in 0..NUM_FEATURES {
                sums[c][f] += features[i][f] as f64;
            }
        }

        for j in 0..k {
            if counts[j] > 0 {
                for f in 0..NUM_FEATURES {
                    centroids[j][f] = (sums[j][f] / counts[j] as f64) as f32;
                }
            }
        }
    }

    assignments
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute bucket assignments for all 1326 combos on a 4-card turn board.
///
/// Hands blocked by the board get [`BLOCKED_BUCKET`].
/// If `k` exceeds the number of valid hands, it is capped automatically.
pub fn compute_buckets(board: &[Card; 4], k: usize) -> BucketMapping {
    let equity = compute_equity(board);
    let (features, valid_indices) = compute_features(board, &equity);
    let num_valid = valid_indices.len();
    let actual_k = k.min(num_valid);

    let mut hand_to_bucket = [BLOCKED_BUCKET; 1326];

    if actual_k == 0 {
        return BucketMapping {
            hand_to_bucket,
            k: 0,
            num_valid: 0,
        };
    }

    if actual_k >= num_valid {
        // Each hand gets its own bucket
        for (i, &combo_idx) in valid_indices.iter().enumerate() {
            hand_to_bucket[combo_idx as usize] = i as u16;
        }
        return BucketMapping {
            hand_to_bucket,
            k: num_valid,
            num_valid,
        };
    }

    // K-means++ init + Lloyd's algorithm
    let mut rng = board_seed(board);
    let mut centroids = kmeans_pp_init(&features, actual_k, &mut rng);
    let assignments = lloyd(&features, &mut centroids, MAX_KMEANS_ITER);

    for (i, &combo_idx) in valid_indices.iter().enumerate() {
        hand_to_bucket[combo_idx as usize] = assignments[i];
    }

    BucketMapping {
        hand_to_bucket,
        k: actual_k,
        num_valid,
    }
}

// ---------------------------------------------------------------------------
// Projection: 1326 combos → K buckets
// ---------------------------------------------------------------------------

/// Project a 1326-element reach vector into K buckets, then normalize so the
/// vector sums to 1.0.  Blocked hands ([`BLOCKED_BUCKET`]) are skipped.
/// If all reaches are zero the returned vector is all-zero.
pub fn project_range_to_buckets(reach: &[f32; 1326], mapping: &BucketMapping) -> Vec<f32> {
    let mut buckets = vec![0.0f32; mapping.k];
    for combo_idx in 0..1326 {
        let b = mapping.hand_to_bucket[combo_idx];
        if b == BLOCKED_BUCKET {
            continue;
        }
        buckets[b as usize] += reach[combo_idx];
    }
    let total: f64 = buckets.iter().map(|&x| x as f64).sum();
    if total > 0.0 {
        for v in buckets.iter_mut() {
            *v = (*v as f64 / total) as f32;
        }
    }
    buckets
}

/// Project 1326-element CFVs into K buckets using reach-weighted averaging.
///
/// `cfv_bucket[k] = Σ(reach[h] · cfv[h]) / Σ(reach[h])` for all hands h in bucket k.
/// If a bucket has zero total reach its CFV is set to 0.0.
pub fn project_cfv_to_buckets(
    cfv: &[f32; 1326],
    reach: &[f32; 1326],
    mapping: &BucketMapping,
) -> Vec<f32> {
    let mut weighted_sum = vec![0.0f64; mapping.k];
    let mut reach_sum = vec![0.0f64; mapping.k];
    for combo_idx in 0..1326 {
        let b = mapping.hand_to_bucket[combo_idx];
        if b == BLOCKED_BUCKET {
            continue;
        }
        let r = reach[combo_idx] as f64;
        weighted_sum[b as usize] += r * cfv[combo_idx] as f64;
        reach_sum[b as usize] += r;
    }
    let mut result = vec![0.0f32; mapping.k];
    for k in 0..mapping.k {
        if reach_sum[k] > 0.0 {
            result[k] = (weighted_sum[k] / reach_sum[k]) as f32;
        }
    }
    result
}

/// Broadcast bucket-level CFVs back to 1326 per-combo CFVs.
///
/// Each combo receives the CFV of its assigned bucket.
/// Blocked hands ([`BLOCKED_BUCKET`]) get 0.0.
pub fn expand_cfv_from_buckets(
    bucket_cfv: &[f32],
    mapping: &BucketMapping,
) -> [f32; 1326] {
    let mut result = [0.0f32; 1326];
    for combo_idx in 0..1326 {
        let b = mapping.hand_to_bucket[combo_idx];
        if b != BLOCKED_BUCKET && (b as usize) < bucket_cfv.len() {
            result[combo_idx] = bucket_cfv[b as usize];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Board features (15-dim, for the bucketed value network)
// ---------------------------------------------------------------------------

/// Compute the 15 board features for the bucketed value network.
///
/// Layout (matches `docs/feature_input.csv`):
///  0-3  board ranks sorted descending, / 12
///  4-7  suit counts for suits 0–3, / 4
///  8    board_paired  (any rank appears 2+ times)
///  9    board_trips   (any rank appears 3+ times)
///  10   monotone      (3+ cards of same suit)
///  11   connectivity  = 1 − rank_span / 12
///  12   log_spr       = min(1, ln(1 + stack/pot) / ln(22))
///  13   pot_fraction  = pot / (pot + stack)
///  14   stack_fraction = stack / (pot + stack)
pub fn compute_board_features(board: &[Card; 4], pot: f32, stack: f32) -> [f32; 15] {
    let mut features = [0.0f32; 15];

    // Ranks and suits
    let mut ranks: [u8; 4] = [board[0] >> 2, board[1] >> 2, board[2] >> 2, board[3] >> 2];
    ranks.sort_unstable_by(|a, b| b.cmp(a)); // descending

    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    for &c in board {
        rank_counts[(c >> 2) as usize] += 1;
        suit_counts[(c & 3) as usize] += 1;
    }

    // 0-3: board ranks descending, normalized
    for i in 0..4 {
        features[i] = ranks[i] as f32 / 12.0;
    }

    // 4-7: suit counts normalized
    for i in 0..4 {
        features[4 + i] = suit_counts[i] as f32 / 4.0;
    }

    // 8: board_paired
    features[8] = if rank_counts.iter().any(|&c| c >= 2) {
        1.0
    } else {
        0.0
    };

    // 9: board_trips
    features[9] = if rank_counts.iter().any(|&c| c >= 3) {
        1.0
    } else {
        0.0
    };

    // 10: monotone (3+ same suit)
    features[10] = if suit_counts.iter().any(|&c| c >= 3) {
        1.0
    } else {
        0.0
    };

    // 11: connectivity = 1 - rank_span / 12
    let max_rank = ranks[0];
    let min_rank = ranks[3];
    let span = max_rank - min_rank;
    features[11] = 1.0 - span as f32 / 12.0;

    // 12: log_spr
    let spr = if pot > 0.0 { stack / pot } else { 0.0 };
    features[12] = ((1.0 + spr).ln() / 22.0f32.ln()).min(1.0);

    // 13-14: pot/stack fractions
    let total = pot + stack;
    if total > 0.0 {
        features[13] = pot / total;
        features[14] = stack / total;
    }

    features
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Board: Td 9d 6h Qc
    // Td = 4*8+1 = 33, 9d = 4*7+1 = 29, 6h = 4*4+2 = 18, Qc = 4*10+0 = 40
    const TEST_BOARD: [Card; 4] = [33, 29, 18, 40];

    #[test]
    fn test_equity_known_board() {
        let equity = compute_equity(&TEST_BOARD);

        // AA: AhAs = combo index 1325 (highest combo). AA is a strong overpair.
        let aa_equity = equity[1325];
        assert!(
            aa_equity > 0.7,
            "AA equity should be > 0.7, got {}",
            aa_equity
        );

        // 72o: 2d7c = card_pair_to_index(1, 20) = 69. Very weak hand.
        let seven_two_equity = equity[69];
        assert!(
            seven_two_equity < 0.3,
            "72o equity should be < 0.3, got {}",
            seven_two_equity
        );
    }

    #[test]
    fn test_blocked_hands_excluded() {
        let equity = compute_equity(&TEST_BOARD);
        let buckets = compute_buckets(&TEST_BOARD, 10);

        // Td (card 33) is on the board. Any combo containing card 33 is blocked.
        for combo_idx in 0..1326u16 {
            let (c1, c2) = index_to_card_pair(combo_idx as usize);
            let blocked = TEST_BOARD.contains(&c1) || TEST_BOARD.contains(&c2);
            if blocked {
                assert_eq!(
                    buckets.hand_to_bucket[combo_idx as usize],
                    BLOCKED_BUCKET,
                    "Combo {} (cards {},{}) should be BLOCKED_BUCKET",
                    combo_idx,
                    c1,
                    c2
                );
                assert_eq!(
                    equity[combo_idx as usize], 0.0,
                    "Blocked combo {} should have equity 0.0",
                    combo_idx
                );
            }
        }
    }

    #[test]
    fn test_bucket_count() {
        let buckets = compute_buckets(&TEST_BOARD, 10);
        assert_eq!(buckets.k, 10);

        // All valid hands should be assigned to buckets 0..9
        for combo_idx in 0..1326 {
            let b = buckets.hand_to_bucket[combo_idx];
            if b != BLOCKED_BUCKET {
                assert!(
                    b < 10,
                    "Combo {} has bucket {}, expected < 10",
                    combo_idx,
                    b
                );
            }
        }

        // Check all 10 buckets are used
        let mut used = [false; 10];
        for &b in &buckets.hand_to_bucket {
            if b != BLOCKED_BUCKET {
                used[b as usize] = true;
            }
        }
        assert!(
            used.iter().all(|&u| u),
            "Not all 10 buckets are used: {:?}",
            used
        );
    }

    #[test]
    fn test_similar_hands_same_bucket() {
        let buckets = compute_buckets(&TEST_BOARD, 10);

        // All 6 AA combos (none blocked by this board) should land in the same bucket.
        // AA combo indices: 1320, 1321, 1322, 1323, 1324, 1325
        let aa_indices: [usize; 6] = [1320, 1321, 1322, 1323, 1324, 1325];
        let aa_bucket = buckets.hand_to_bucket[aa_indices[0]];
        assert_ne!(aa_bucket, BLOCKED_BUCKET);
        for &idx in &aa_indices[1..] {
            assert_eq!(
                buckets.hand_to_bucket[idx], aa_bucket,
                "AA combo {} has bucket {}, expected {}",
                idx, buckets.hand_to_bucket[idx], aa_bucket
            );
        }
    }

    #[test]
    fn test_determinism() {
        let b1 = compute_buckets(&TEST_BOARD, 50);
        let b2 = compute_buckets(&TEST_BOARD, 50);
        assert_eq!(b1.k, b2.k);
        assert_eq!(b1.num_valid, b2.num_valid);
        assert_eq!(b1.hand_to_bucket, b2.hand_to_bucket);
    }

    #[test]
    fn test_k_larger_than_valid() {
        // There are 1128 valid hands; requesting 2000 should not panic
        let buckets = compute_buckets(&TEST_BOARD, 2000);
        assert_eq!(buckets.k, 1128);
        assert_eq!(buckets.num_valid, 1128);

        // Each hand should get a unique bucket 0..1127
        let mut seen = vec![false; 1128];
        for &b in &buckets.hand_to_bucket {
            if b != BLOCKED_BUCKET {
                assert!((b as usize) < 1128);
                seen[b as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn test_equity_all_valid_counted() {
        let equity = compute_equity(&TEST_BOARD);

        // Exactly C(48,2) = 1128 hands are not blocked by the 4-card board
        let valid_count = equity.iter().filter(|&&e| e > 0.0).count();
        assert_eq!(
            valid_count, 1128,
            "Expected 1128 valid hands, got {}",
            valid_count
        );
    }

    // -----------------------------------------------------------------------
    // Projection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_range_sums_to_one() {
        let buckets = compute_buckets(&TEST_BOARD, 10);

        // Give every valid hand reach 1.0
        let mut reach = [0.0f32; 1326];
        for combo_idx in 0..1326 {
            if buckets.hand_to_bucket[combo_idx] != BLOCKED_BUCKET {
                reach[combo_idx] = 1.0;
            }
        }

        let projected = project_range_to_buckets(&reach, &buckets);
        assert_eq!(projected.len(), 10);

        let sum: f64 = projected.iter().map(|&x| x as f64).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Projected range should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_project_range_zero_reach() {
        let buckets = compute_buckets(&TEST_BOARD, 10);
        let reach = [0.0f32; 1326];

        let projected = project_range_to_buckets(&reach, &buckets);
        assert!(
            projected.iter().all(|&x| x == 0.0),
            "All-zero reach should produce all-zero projected range"
        );
    }

    #[test]
    fn test_project_cfv_basic() {
        let buckets = compute_buckets(&TEST_BOARD, 10);

        // Uniform reach = 1.0 for all valid hands, CFV = equity
        let equity = compute_equity(&TEST_BOARD);
        let mut reach = [0.0f32; 1326];
        for combo_idx in 0..1326 {
            if buckets.hand_to_bucket[combo_idx] != BLOCKED_BUCKET {
                reach[combo_idx] = 1.0;
            }
        }

        let projected = project_cfv_to_buckets(&equity, &reach, &buckets);
        assert_eq!(projected.len(), 10);

        // Each bucket's CFV should be the average equity of its hands
        for k in 0..10 {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for combo_idx in 0..1326 {
                if buckets.hand_to_bucket[combo_idx] == k as u16 {
                    sum += equity[combo_idx] as f64;
                    count += 1;
                }
            }
            if count > 0 {
                let expected = (sum / count as f64) as f32;
                assert!(
                    (projected[k] - expected).abs() < 1e-5,
                    "Bucket {} CFV: got {}, expected {}",
                    k,
                    projected[k],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_project_cfv_zero_bucket() {
        let buckets = compute_buckets(&TEST_BOARD, 10);

        // Only give reach to hands in bucket 0
        let mut reach = [0.0f32; 1326];
        let mut cfv = [0.0f32; 1326];
        for combo_idx in 0..1326 {
            if buckets.hand_to_bucket[combo_idx] == 0 {
                reach[combo_idx] = 1.0;
                cfv[combo_idx] = 0.5;
            }
        }

        let projected = project_cfv_to_buckets(&cfv, &reach, &buckets);

        // Bucket 0 should have CFV ~0.5, all others 0.0
        assert!((projected[0] - 0.5).abs() < 1e-5);
        for k in 1..10 {
            assert_eq!(
                projected[k], 0.0,
                "Bucket {} with zero reach should have CFV 0.0",
                k
            );
        }
    }

    // -----------------------------------------------------------------------
    // Board feature tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_board_features_range_and_shape() {
        // Td(33) 9d(29) 6h(18) Qc(40): pot=1000, stack=2000
        let features = compute_board_features(&TEST_BOARD, 1000.0, 2000.0);

        // All features should be in [0, 1]
        for (i, &f) in features.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&f),
                "Feature {} = {} is out of [0,1]",
                i,
                f
            );
        }

        // Ranks sorted desc: Q(10), T(8), 9(7), 6(4) → /12 = 0.833, 0.667, 0.583, 0.333
        assert!((features[0] - 10.0 / 12.0).abs() < 1e-5); // Q
        assert!((features[1] - 8.0 / 12.0).abs() < 1e-5); // T
        assert!((features[2] - 7.0 / 12.0).abs() < 1e-5); // 9
        assert!((features[3] - 4.0 / 12.0).abs() < 1e-5); // 6

        // Suits: Qc(0), Td(1), 9d(1), 6h(2) → counts: c=1, d=2, h=1, s=0
        assert!((features[4] - 0.25).abs() < 1e-5); // suit 0 (clubs) = 1/4
        assert!((features[5] - 0.50).abs() < 1e-5); // suit 1 (diamonds) = 2/4
        assert!((features[6] - 0.25).abs() < 1e-5); // suit 2 (hearts) = 1/4
        assert!((features[7] - 0.0).abs() < 1e-5); // suit 3 (spades) = 0/4

        // Not paired, not trips
        assert_eq!(features[8], 0.0);
        assert_eq!(features[9], 0.0);

        // Not monotone (max suit count = 2)
        assert_eq!(features[10], 0.0);

        // Connectivity: span = Q(10) - 6(4) = 6, conn = 1 - 6/12 = 0.5
        assert!((features[11] - 0.5).abs() < 1e-5);

        // SPR = 2000/1000 = 2, log_spr = ln(3)/ln(22) ≈ 0.355
        assert!((features[12] - (3.0f32.ln() / 22.0f32.ln())).abs() < 1e-4);

        // pot_frac = 1000/3000 ≈ 0.333, stack_frac = 2000/3000 ≈ 0.667
        assert!((features[13] - 1.0 / 3.0).abs() < 1e-5);
        assert!((features[14] - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_board_features_monotone() {
        // 3 diamonds: 2d(1) 5d(9) Td(33) Qs(42)
        let board: [Card; 4] = [1, 9, 33, 42];
        let features = compute_board_features(&board, 500.0, 500.0);
        assert_eq!(features[10], 1.0, "3 diamonds should be monotone");
    }

    #[test]
    fn test_board_features_paired() {
        // Board with a pair: 2c(0) 2d(1) 5h(10) Ks(47)
        let board: [Card; 4] = [0, 1, 10, 47];
        let features = compute_board_features(&board, 500.0, 500.0);
        assert_eq!(features[8], 1.0, "Paired board should have board_paired=1");
        assert_eq!(features[9], 0.0, "No trips");
    }
}
