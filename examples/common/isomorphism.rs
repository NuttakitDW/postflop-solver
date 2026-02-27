//! Suit isomorphism utilities for oracle reuse across isomorphic boards.
//!
//! When two flop boards differ only by a suit permutation (e.g., As3s4s vs Ad3d4d),
//! their game trees are structurally identical. However, the private hand ordering
//! differs because card IDs are suit-dependent. This module computes the permutation
//! needed to remap oracle CFV vectors from one board to another.

use postflop_solver::*;
use std::collections::HashMap;

/// Compute the suit permutation that maps oracle_flop to actual_flop.
///
/// Returns `suit_map` where `suit_map[oracle_suit] = actual_suit`.
/// Returns `None` if the boards have different rank patterns (not isomorphic).
///
/// Both flop arrays must be sorted (as returned by `flop_from_str`).
pub fn compute_suit_permutation(
    oracle_flop: &[Card; 3],
    actual_flop: &[Card; 3],
) -> Option<[u8; 4]> {
    // Check ranks match (sorted flop → same position = same rank required)
    for i in 0..3 {
        if oracle_flop[i] >> 2 != actual_flop[i] >> 2 {
            return None;
        }
    }

    // Build partial suit mapping from flop cards
    let mut suit_map: [Option<u8>; 4] = [None; 4];
    let mut reverse_map: [Option<u8>; 4] = [None; 4];

    for i in 0..3 {
        let oracle_suit = oracle_flop[i] & 3;
        let actual_suit = actual_flop[i] & 3;

        match suit_map[oracle_suit as usize] {
            Some(mapped) => {
                if mapped != actual_suit {
                    return None; // Inconsistent: same oracle suit maps to two different actual suits
                }
            }
            None => {
                // Check reverse consistency
                if let Some(mapped_from) = reverse_map[actual_suit as usize] {
                    if mapped_from != oracle_suit {
                        return None; // Two oracle suits map to the same actual suit
                    }
                }
                suit_map[oracle_suit as usize] = Some(actual_suit);
                reverse_map[actual_suit as usize] = Some(oracle_suit);
            }
        }
    }

    // Fill in remaining (unmapped) suits — order doesn't matter for suit-agnostic ranges
    let unmapped_oracle: Vec<u8> = (0..4u8).filter(|&s| suit_map[s as usize].is_none()).collect();
    let unmapped_actual: Vec<u8> =
        (0..4u8).filter(|&s| reverse_map[s as usize].is_none()).collect();
    for (&o, &a) in unmapped_oracle.iter().zip(unmapped_actual.iter()) {
        suit_map[o as usize] = Some(a);
    }

    Some([
        suit_map[0].unwrap(),
        suit_map[1].unwrap(),
        suit_map[2].unwrap(),
        suit_map[3].unwrap(),
    ])
}

/// Compute the hand index permutation for one player.
///
/// Returns `permutation` where `permutation[actual_idx] = oracle_idx`.
/// This means: `actual_cfv[i] = oracle_cfv[permutation[i]]`.
///
/// `suit_map[oracle_suit] = actual_suit` (as returned by `compute_suit_permutation`).
pub fn compute_hand_permutation(
    oracle_flop: &[Card; 3],
    actual_flop: &[Card; 3],
    range: &Range,
    suit_map: &[u8; 4],
) -> Vec<usize> {
    let oracle_mask: u64 = oracle_flop.iter().fold(0u64, |m, &c| m | (1u64 << c));
    let actual_mask: u64 = actual_flop.iter().fold(0u64, |m, &c| m | (1u64 << c));

    let (oracle_hands, _) = range.get_hands_weights(oracle_mask);
    let (actual_hands, _) = range.get_hands_weights(actual_mask);

    assert_eq!(
        oracle_hands.len(),
        actual_hands.len(),
        "Hand counts differ: oracle={} vs actual={} — boards are not isomorphic for this range",
        oracle_hands.len(),
        actual_hands.len(),
    );

    // Invert suit_map: inv[actual_suit] = oracle_suit
    let mut inv_suit_map = [0u8; 4];
    for (oracle_s, &actual_s) in suit_map.iter().enumerate() {
        inv_suit_map[actual_s as usize] = oracle_s as u8;
    }

    // Build oracle hand → index lookup
    let oracle_index: HashMap<(Card, Card), usize> = oracle_hands
        .iter()
        .enumerate()
        .map(|(idx, &hand)| (hand, idx))
        .collect();

    // For each actual hand, map it back to oracle suits and find the oracle index
    actual_hands
        .iter()
        .map(|&(c1, c2)| {
            let oc1 = (c1 >> 2) * 4 + inv_suit_map[(c1 & 3) as usize];
            let oc2 = (c2 >> 2) * 4 + inv_suit_map[(c2 & 3) as usize];
            let (lo, hi) = if oc1 < oc2 { (oc1, oc2) } else { (oc2, oc1) };
            *oracle_index
                .get(&(lo, hi))
                .unwrap_or_else(|| panic!("Hand ({}, {}) not found in oracle hands", lo, hi))
        })
        .collect()
}

/// Apply a permutation to a CFV vector.
///
/// `permutation[actual_idx] = oracle_idx`, so:
/// `result[actual_idx] = original[permutation[actual_idx]]`
pub fn permute_cfv(original: &[f32], permutation: &[usize]) -> Vec<f32> {
    permutation.iter().map(|&oracle_idx| original[oracle_idx]).collect()
}

/// Convert a flop (sorted card array) to a display string like "As3s4s".
pub fn flop_to_string(flop: &[Card; 3]) -> String {
    flop.iter()
        .map(|&c| card_to_string(c).unwrap())
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suit_permutation_identity() {
        let flop = flop_from_str("As3s4s").unwrap();
        let perm = compute_suit_permutation(&flop, &flop).unwrap();
        assert_eq!(perm, [0, 1, 2, 3]);
    }

    #[test]
    fn test_suit_permutation_spade_to_diamond() {
        let oracle = flop_from_str("As3s4s").unwrap();
        let actual = flop_from_str("Ad3d4d").unwrap();
        let perm = compute_suit_permutation(&oracle, &actual).unwrap();
        // s(3) → d(1), remaining {c(0),d(1),h(2)} → {c(0),h(2),s(3)}
        assert_eq!(perm[3], 1); // s → d
        // Check it's a valid permutation
        let mut sorted = perm.to_vec();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_suit_permutation_different_ranks() {
        let a = flop_from_str("As3s4s").unwrap();
        let b = flop_from_str("Ad3d5d").unwrap();
        assert!(compute_suit_permutation(&a, &b).is_none());
    }

    #[test]
    fn test_suit_permutation_paired_board() {
        // 6c6d9s → 6h6d9c: c→h, d→d, s→c
        let oracle = flop_from_str("6c6d9s").unwrap();
        let actual = flop_from_str("6h6d9c").unwrap();
        let perm = compute_suit_permutation(&oracle, &actual).unwrap();
        assert_eq!(perm[0], 2); // c → h
        assert_eq!(perm[1], 1); // d → d
        assert_eq!(perm[3], 0); // s → c
    }
}
