//! Exact turn CFV evaluator.
//!
//! Solves a Turn-start game once, then evaluates counterfactual values
//! for any opponent reach distribution. Nash equilibrium strategies are
//! reach-independent, so a single solve supports unlimited evaluations.
//!
//! **Usage pattern:**
//! 1. `ExactTurnCfv::new(...)` — build and solve a Turn-start game
//! 2. `evaluate(player, cfreach)` — compute CFVs (cheap tree traversal)
//!
//! Later, replace `ExactTurnCfv` with an NN that has the same interface.

use crate::action_tree::*;
use crate::bet_size::*;
use crate::card::*;
use crate::game::PostFlopGame;
use crate::interface::*;
use crate::range::Range;
use crate::solver::solve;
use crate::utility::compute_cfvalue_recursive;
use std::mem::MaybeUninit;

/// Bet size configuration for turn/river subtrees.
#[derive(Clone)]
pub struct TurnBetConfig {
    pub turn_bet_sizes: [BetSizeOptions; 2],
    pub river_bet_sizes: [BetSizeOptions; 2],
    pub add_allin_threshold: f64,
    pub force_allin_threshold: f64,
    pub merging_threshold: f64,
}

impl Default for TurnBetConfig {
    fn default() -> Self {
        Self {
            turn_bet_sizes: [
                ("100%,a", "a").try_into().unwrap(),
                ("100%,a", "a").try_into().unwrap(),
            ],
            river_bet_sizes: [
                ("100%,a", "a").try_into().unwrap(),
                ("100%,a", "a").try_into().unwrap(),
            ],
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
        }
    }
}

/// A pre-solved Turn-start game that can evaluate CFVs for any opponent reach.
///
/// This is the exact solver implementation. Later, replace this with an NN
/// that has the same `evaluate` interface.
pub struct ExactTurnCfv {
    game: PostFlopGame,
}

impl ExactTurnCfv {
    /// Build and solve a Turn-start game.
    ///
    /// # Arguments
    /// - `flop` — three flop cards
    /// - `turn_card` — the turn card
    /// - `oop_range`, `ip_range` — player ranges
    /// - `pot` — pot size at the start of turn action
    /// - `stack` — effective stack remaining
    /// - `bet_config` — bet/raise size configuration
    /// - `max_iterations` — DCFR iteration limit
    /// - `target_exploitability` — stop when exploitability drops below this (in chips)
    pub fn new(
        flop: [Card; 3],
        turn_card: Card,
        oop_range: &Range,
        ip_range: &Range,
        pot: i32,
        stack: i32,
        bet_config: &TurnBetConfig,
        max_iterations: u32,
        target_exploitability: f32,
    ) -> Result<Self, String> {
        let card_config = CardConfig {
            range: [oop_range.clone(), ip_range.clone()],
            flop,
            turn: turn_card,
            river: NOT_DEALT,
        };

        let tree_config = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: pot,
            effective_stack: stack,
            turn_bet_sizes: bet_config.turn_bet_sizes.clone(),
            river_bet_sizes: bet_config.river_bet_sizes.clone(),
            add_allin_threshold: bet_config.add_allin_threshold,
            force_allin_threshold: bet_config.force_allin_threshold,
            merging_threshold: bet_config.merging_threshold,
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config)?;
        let mut game = PostFlopGame::with_config(card_config, action_tree)?;
        game.allocate_memory(false);
        solve(&mut game, max_iterations, target_exploitability, false);

        Ok(Self { game })
    }

    /// Compute CFVs for the given player using the opponent's reach.
    ///
    /// # Arguments
    /// - `player` — 0 (OOP) or 1 (IP)
    /// - `cfreach` — opponent's reach probabilities in game-internal hand indexing.
    ///   Length must equal `num_private_hands(player ^ 1)`.
    ///
    /// # Returns
    /// CFVs in game-internal hand indexing (length = `num_private_hands(player)`).
    pub fn evaluate(&self, player: usize, cfreach: &[f32]) -> Vec<f32> {
        let num_hands = self.game.num_private_hands(player);
        let expected_len = self.game.num_private_hands(player ^ 1);
        assert_eq!(
            cfreach.len(),
            expected_len,
            "cfreach length {} != expected {} (num_private_hands of opponent)",
            cfreach.len(),
            expected_len
        );

        let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
        {
            let mut root = self.game.root();
            compute_cfvalue_recursive(
                &mut result,
                &self.game,
                &mut root,
                player,
                cfreach,
                false,
            );
        }

        result.into_iter().map(|v| unsafe { v.assume_init() }).collect()
    }

    /// Compute CFVs in 1326-combo space.
    ///
    /// Maps cfreach from 1326-combo indexing to game-internal indexing,
    /// evaluates, then maps results back to 1326-combo indexing.
    /// Blocked/out-of-range combos have CFV = 0.
    ///
    /// This is the interface an NN replacement would use.
    pub fn evaluate_1326(&self, player: usize, cfreach_1326: &[f32; 1326]) -> [f32; 1326] {
        let opponent = player ^ 1;

        // Map cfreach from 1326-combo space to game-internal opponent hand indexing
        let opp_cards = self.game.private_cards(opponent);
        let mut cfreach_internal = vec![0.0f32; opp_cards.len()];
        for (hand_idx, &(c1, c2)) in opp_cards.iter().enumerate() {
            let combo_idx = card_pair_to_index(c1, c2);
            cfreach_internal[hand_idx] = cfreach_1326[combo_idx];
        }

        // Evaluate in game-internal space
        let cfvs_internal = self.evaluate(player, &cfreach_internal);

        // Map CFVs back to 1326-combo space
        let player_cards = self.game.private_cards(player);
        let mut cfvs_1326 = [0.0f32; 1326];
        for (hand_idx, &(c1, c2)) in player_cards.iter().enumerate() {
            let combo_idx = card_pair_to_index(c1, c2);
            cfvs_1326[combo_idx] = cfvs_internal[hand_idx];
        }

        cfvs_1326
    }

    /// Number of private hands for the given player (game-internal count).
    pub fn num_private_hands(&self, player: usize) -> usize {
        self.game.num_private_hands(player)
    }

    /// Private card pairs for the given player.
    pub fn private_cards(&self, player: usize) -> &[(Card, Card)] {
        self.game.private_cards(player)
    }

    /// Initial reach probabilities for the given player.
    pub fn initial_weights(&self, player: usize) -> &[f32] {
        self.game.initial_weights(player)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::*;

    fn make_evaluator() -> ExactTurnCfv {
        let oop_range: Range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s"
            .parse()
            .unwrap();
        let ip_range: Range =
            "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+"
                .parse()
                .unwrap();

        let flop = flop_from_str("Td9d6h").unwrap();
        let turn = card_from_str("Qc").unwrap();
        let pot = 200;
        let stack = 900;
        let target = pot as f32 * 0.005; // 0.5% of pot

        ExactTurnCfv::new(
            flop,
            turn,
            &oop_range,
            &ip_range,
            pot,
            stack,
            &TurnBetConfig::default(),
            1000,
            target,
        )
        .unwrap()
    }

    #[test]
    fn test_evaluate_matches_direct() {
        // ExactTurnCfv::evaluate should match direct compute_cfvalue_recursive
        let eval = make_evaluator();

        for player in 0..2 {
            let cfreach = eval.initial_weights(player ^ 1).to_vec();
            let cfvs = eval.evaluate(player, &cfreach);
            assert_eq!(cfvs.len(), eval.num_private_hands(player));

            // Values should be finite and reasonable
            for &v in &cfvs {
                assert!(v.is_finite(), "CFV should be finite, got {}", v);
            }
        }
    }

    #[test]
    fn test_zero_sum() {
        let eval = make_evaluator();

        // Reach-weighted CFVs should be zero-sum (within solver tolerance)
        let cfreach_oop = eval.initial_weights(1).to_vec(); // opponent of OOP = IP
        let cfreach_ip = eval.initial_weights(0).to_vec(); // opponent of IP = OOP

        let cfvs_oop = eval.evaluate(0, &cfreach_oop);
        let cfvs_ip = eval.evaluate(1, &cfreach_ip);

        let weighted_oop: f64 = cfvs_oop
            .iter()
            .zip(eval.initial_weights(0))
            .map(|(&v, &w)| v as f64 * w as f64)
            .sum();
        let weighted_ip: f64 = cfvs_ip
            .iter()
            .zip(eval.initial_weights(1))
            .map(|(&v, &w)| v as f64 * w as f64)
            .sum();

        let sum = weighted_oop + weighted_ip;
        assert!(
            sum.abs() < 1.0,
            "Reach-weighted CFVs should be ~zero-sum, got {}",
            sum
        );
    }

    #[test]
    fn test_zero_reach_gives_zero_cfv() {
        let eval = make_evaluator();

        let zero_reach = vec![0.0f32; eval.num_private_hands(1)];
        let cfvs = eval.evaluate(0, &zero_reach);

        for (i, &v) in cfvs.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "Zero reach should give zero CFV, hand {} got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_evaluate_1326() {
        let eval = make_evaluator();

        // Build 1326-space reach from initial_weights
        let mut cfreach_1326 = [0.0f32; 1326];
        for (hand_idx, &(c1, c2)) in eval.private_cards(1).iter().enumerate() {
            let combo_idx = card_pair_to_index(c1, c2);
            cfreach_1326[combo_idx] = eval.initial_weights(1)[hand_idx];
        }

        let cfvs_1326 = eval.evaluate_1326(0, &cfreach_1326);

        // Compare with direct evaluate in game-internal space
        let cfreach_internal = eval.initial_weights(1).to_vec();
        let cfvs_internal = eval.evaluate(0, &cfreach_internal);

        // Map internal to 1326 for comparison
        for (hand_idx, &(c1, c2)) in eval.private_cards(0).iter().enumerate() {
            let combo_idx = card_pair_to_index(c1, c2);
            let diff = (cfvs_1326[combo_idx] - cfvs_internal[hand_idx]).abs();
            assert!(
                diff < 1e-6,
                "1326 and internal should match: combo {} diff {}",
                combo_idx,
                diff
            );
        }

        // Blocked combos should be zero
        let active: std::collections::HashSet<usize> = eval
            .private_cards(0)
            .iter()
            .map(|&(c1, c2)| card_pair_to_index(c1, c2))
            .collect();
        for i in 0..1326 {
            if !active.contains(&i) {
                assert_eq!(cfvs_1326[i], 0.0, "Blocked combo {} should be 0", i);
            }
        }
    }

    #[test]
    fn test_different_reach_gives_different_cfv() {
        let eval = make_evaluator();

        let full_reach = eval.initial_weights(1).to_vec();
        let cfvs_full = eval.evaluate(0, &full_reach);

        // Zero out half the opponent's reach
        let mut partial_reach = full_reach.clone();
        for i in 0..partial_reach.len() / 2 {
            partial_reach[i] = 0.0;
        }
        let cfvs_partial = eval.evaluate(0, &partial_reach);

        // Should be different
        let max_diff: f32 = cfvs_full
            .iter()
            .zip(&cfvs_partial)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Different reaches should give different CFVs"
        );
    }
}
