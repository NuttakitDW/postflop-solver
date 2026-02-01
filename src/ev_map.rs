//! EV Map: Precomputed hand EVs for leaf evaluation with Range-Agnostic transfer.
//!
//! This module provides "perfect information" leaf evaluation by using
//! precomputed EVs from a full-tree solve. The EVs capture the true value
//! of each hand including all future street betting.
//!
//! **Range-Agnostic Transfer**: To handle different ranges than the precompute,
//! we use Equity-Ratio Scaling:
//!   EV_final = Precomputed_EV × (Current_Equity / Precomputed_Equity)
//!
//! This allows a 100bb SRP precompute to work for 3-bet pots, different stack sizes, etc.
//!
//! Workflow:
//! 1. Solve full tree (flop → turn → river) with WIDE ranges (100bb SRP)
//! 2. Extract EVs AND equities at flop-terminal nodes
//! 3. Save as ev_map.bin
//! 4. Use ev_map for instant flop-only solving with ANY bet sizes and ranges

use crate::card::Card;
use crate::game::PostFlopGame;
use crate::interface::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// EV data for a single flop-terminal node
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub struct FlopTerminalEv {
    /// Hash of the action path to reach this terminal (e.g., check-check, bet-call)
    pub path_hash: u64,
    /// Pot size at this terminal (for normalization)
    pub pot_size: i32,
    /// OOP EVs: (card1, card2) → (EV, Equity) - both needed for range transfer
    pub oop_data: HashMap<(Card, Card), (f32, f32)>,
    /// IP EVs: (card1, card2) → (EV, Equity)
    pub ip_data: HashMap<(Card, Card), (f32, f32)>,
}

/// Complete EV map for a flop
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub struct EvMap {
    /// Version for compatibility
    pub version: String,
    /// Flop cards (sorted)
    pub flop: [Card; 3],
    /// All flop-terminal EVs indexed by path_hash
    pub terminals: HashMap<u64, FlopTerminalEv>,
}

impl EvMap {
    const VERSION: &'static str = "ev-map-v2";

    /// Create empty EV map for a flop
    pub fn new(flop: [Card; 3]) -> Self {
        let mut sorted_flop = flop;
        sorted_flop.sort();
        Self {
            version: Self::VERSION.to_string(),
            flop: sorted_flop,
            terminals: HashMap::new(),
        }
    }

    /// Extract EV map from a SOLVED full-tree game.
    /// This captures the EV at each point where flop action ends.
    ///
    /// IMPORTANT: The game should be solved with WIDE ranges (100bb SRP recommended)
    /// to ensure all hand combinations are covered.
    pub fn from_solved_game(game: &PostFlopGame) -> Result<Self, String> {
        if !game.is_solved() {
            return Err("Game must be solved to extract EV map".to_string());
        }

        let flop = game.card_config().flop;
        let mut ev_map = Self::new(flop);

        // Traverse tree and find flop-terminal nodes
        ev_map.extract_terminals_recursive(game, 0, 0)?;

        println!("Extracted {} flop-terminal EV sets", ev_map.terminals.len());
        Ok(ev_map)
    }

    /// Recursively find flop-terminal nodes and extract EVs
    fn extract_terminals_recursive(
        &mut self,
        game: &PostFlopGame,
        node_index: usize,
        _path_hash: u64,  // Unused - we use pot_size as key instead
    ) -> Result<(), String> {
        let node = game.node_arena()[node_index].lock();

        // Check if this is a flop-level node (turn not dealt yet)
        let is_flop_level = node.turn() == crate::card::NOT_DEALT;

        if node.is_terminal() {
            // Fold at flop level - this is a flop terminal
            if is_flop_level {
                // Use pot size as the hash key (must match evaluation.rs compute_path_hash_simple)
                let pot_size = game.tree_config().starting_pot + 2 * node.amount();
                let pot_hash = pot_size as u64;
                let terminal_ev = self.extract_fold_terminal(game, node_index, pot_hash)?;
                self.terminals.insert(pot_hash, terminal_ev);
            }
            return Ok(());
        }

        if node.is_chance() {
            // Chance node at flop level = deal turn = flop action is complete
            // This is a flop terminal - extract EVs
            if is_flop_level {
                // Use pot size as the hash key (must match evaluation.rs compute_path_hash_simple)
                let pot_size = game.tree_config().starting_pot + 2 * node.amount();
                let pot_hash = pot_size as u64;
                let terminal_ev = self.extract_showdown_terminal(game, node_index, pot_hash)?;
                self.terminals.insert(pot_hash, terminal_ev);
            }
            // Don't recurse into turn/river - we only care about flop terminals
            return Ok(());
        }

        // Action node - recurse into children
        let num_actions = node.num_actions();
        let children_offset = node.children_offset() as usize;
        drop(node);

        for i in 0..num_actions {
            let child_index = node_index + children_offset + i;
            // Continue recursion (path_hash not used, we use pot_size at terminals)
            self.extract_terminals_recursive(game, child_index, 0)?;
        }

        Ok(())
    }

    /// Extract EVs at a fold terminal (one player folds)
    fn extract_fold_terminal(
        &self,
        game: &PostFlopGame,
        node_index: usize,
        path_hash: u64,
    ) -> Result<FlopTerminalEv, String> {
        let node = game.node_arena()[node_index].lock();
        let pot_size = game.tree_config().starting_pot + 2 * node.amount();
        let folder = node.player() & 0x03; // PLAYER_MASK - who folded

        let mut oop_data = HashMap::new();
        let mut ip_data = HashMap::new();

        // At fold terminals, one player wins the pot
        // Winner gets: pot_size - their_contribution (which is node.amount())
        // Folder gets: -their_contribution

        // For equity at fold, the winner has 100% equity, folder has 0%
        // But we store the "realized" equity based on the fold

        // OOP hands
        for &(c1, c2) in game.private_cards(0) {
            let key = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let (ev, equity) = if folder == 0 {
                // OOP folded - they lose their contribution, equity = 0
                (-(node.amount() as f32), 0.0)
            } else {
                // IP folded - OOP wins full pot, equity = 1
                ((pot_size - node.amount()) as f32, 1.0)
            };
            oop_data.insert(key, (ev, equity));
        }

        // IP hands
        for &(c1, c2) in game.private_cards(1) {
            let key = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let (ev, equity) = if folder == 1 {
                // IP folded - they lose their contribution, equity = 0
                (-(node.amount() as f32), 0.0)
            } else {
                // OOP folded - IP wins full pot, equity = 1
                ((pot_size - node.amount()) as f32, 1.0)
            };
            ip_data.insert(key, (ev, equity));
        }

        Ok(FlopTerminalEv {
            path_hash,
            pot_size,
            oop_data,
            ip_data,
        })
    }

    /// Extract EVs at a chance node (turn deal) - this is the key function!
    /// The EV here represents the expected value INCLUDING all future betting.
    fn extract_showdown_terminal(
        &self,
        game: &PostFlopGame,
        node_index: usize,
        path_hash: u64,
    ) -> Result<FlopTerminalEv, String> {
        let is_compressed = game.is_compression_enabled();

        // First pass: get info and extract cfvalues if stored directly
        let (pot_size, cfvalue_player, oop_direct, ip_direct, has_ip_cfvalues) = {
            let node = game.node_arena()[node_index].lock();
            let pot_size = game.tree_config().starting_pot + 2 * node.amount();
            let cfvalue_player = node.cfvalue_storage_player();

            // Get OOP cfvalues if stored directly
            let oop_direct: Option<Vec<f32>> = if cfvalue_player == Some(0) {
                Some(if is_compressed {
                    let scale = node.cfvalue_chance_scale();
                    node.cfvalues_chance_compressed()
                        .iter()
                        .map(|&v| v as f32 * scale / i16::MAX as f32)
                        .collect()
                } else {
                    node.cfvalues_chance().to_vec()
                })
            } else {
                None
            };

            // Get IP cfvalues if stored directly
            let ip_direct: Option<Vec<f32>> = if cfvalue_player == Some(1) {
                Some(if is_compressed {
                    let scale = node.cfvalue_chance_scale();
                    node.cfvalues_chance_compressed()
                        .iter()
                        .map(|&v| v as f32 * scale / i16::MAX as f32)
                        .collect()
                } else {
                    node.cfvalues_chance().to_vec()
                })
            } else if node.has_cfvalues_ip() {
                Some(if is_compressed {
                    let scale = node.cfvalue_ip_scale();
                    node.cfvalues_ip_compressed()
                        .iter()
                        .map(|&v| v as f32 * scale / i16::MAX as f32)
                        .collect()
                } else {
                    node.cfvalues_ip().to_vec()
                })
            } else {
                None
            };

            let has_ip = node.has_cfvalues_ip();
            (pot_size, cfvalue_player, oop_direct, ip_direct, has_ip)
        };

        let mut oop_data = HashMap::new();
        let mut ip_data = HashMap::new();

        // Track if we need to compute by traversal
        let oop_needs_traversal = oop_direct.is_none();
        let ip_needs_traversal = ip_direct.is_none();

        // Get OOP cfvalues (direct or by traversal)
        let oop_cfvalues: Vec<f32> = if let Some(evs) = oop_direct {
            evs
        } else {
            // Need to compute OOP cfvalues by traversing turn/river subtrees
            self.compute_player_evs_by_traversal(game, node_index, 0)
        };

        // Get IP cfvalues (direct or by traversal)
        let ip_cfvalues: Vec<f32> = if let Some(evs) = ip_direct {
            evs
        } else {
            // Need to compute IP cfvalues by traversing turn/river subtrees
            self.compute_player_evs_by_traversal(game, node_index, 1)
        };

        // Debug: log if we had to compute by traversal
        if oop_needs_traversal || ip_needs_traversal {
            println!(
                "[EV_MAP] Computed EVs by traversal: OOP={}, IP={} (cfvalue_player={:?}, has_ip_cfvalues={})",
                oop_needs_traversal,
                ip_needs_traversal,
                cfvalue_player,
                has_ip_cfvalues
            );
        }

        // Compute equities for range transfer
        // We need flop equity for each hand against opponent's range
        let oop_equities = self.compute_flop_equities(game, 0);
        let ip_equities = self.compute_flop_equities(game, 1);

        // Store OOP data
        for (hand_idx, &(c1, c2)) in game.private_cards(0).iter().enumerate() {
            let key = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let ev = if hand_idx < oop_cfvalues.len() {
                oop_cfvalues[hand_idx]
            } else {
                0.0
            };
            let equity = if hand_idx < oop_equities.len() {
                oop_equities[hand_idx]
            } else {
                0.5
            };
            oop_data.insert(key, (ev, equity));
        }

        // Store IP data
        for (hand_idx, &(c1, c2)) in game.private_cards(1).iter().enumerate() {
            let key = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let ev = if hand_idx < ip_cfvalues.len() {
                ip_cfvalues[hand_idx]
            } else {
                0.0
            };
            let equity = if hand_idx < ip_equities.len() {
                ip_equities[hand_idx]
            } else {
                0.5
            };
            ip_data.insert(key, (ev, equity));
        }

        Ok(FlopTerminalEv {
            path_hash,
            pot_size,
            oop_data,
            ip_data,
        })
    }

    /// Compute EVs for a player by traversing turn/river subtrees from a chance node.
    /// This is used when cfvalues are not stored for this player at the chance node.
    fn compute_player_evs_by_traversal(
        &self,
        game: &PostFlopGame,
        chance_node_index: usize,
        player: usize,
    ) -> Vec<f32> {
        let num_hands = game.private_cards(player).len();
        let mut evs = vec![0.0f32; num_hands];
        let mut counts = vec![0.0f32; num_hands];

        let node = game.node_arena()[chance_node_index].lock();
        let num_children = node.num_actions();
        let children_offset = node.children_offset() as usize;
        drop(node);

        // Traverse all turn children
        for child_idx in 0..num_children {
            let child_node_index = chance_node_index + children_offset + child_idx;

            // Get EVs from this turn subtree
            let child_evs = self.compute_subtree_evs(game, child_node_index, player);

            // Accumulate EVs (each turn card contributes equally)
            for (hand_idx, &ev) in child_evs.iter().enumerate() {
                if hand_idx < num_hands && !ev.is_nan() {
                    evs[hand_idx] += ev;
                    counts[hand_idx] += 1.0;
                }
            }
        }

        // Average over all turn cards
        for i in 0..num_hands {
            if counts[i] > 0.0 {
                evs[i] /= counts[i];
            }
        }

        evs
    }

    /// Recursively compute EVs for a player at a subtree rooted at node_index.
    /// Returns EVs for each hand in the player's range.
    fn compute_subtree_evs(
        &self,
        game: &PostFlopGame,
        node_index: usize,
        player: usize,
    ) -> Vec<f32> {
        let node = game.node_arena()[node_index].lock();
        let num_hands = game.private_cards(player).len();
        let is_compressed = game.is_compression_enabled();

        if node.is_terminal() {
            // Terminal node: use cfvalues if available, or compute from payoff
            let folder = (node.player() & 0x03) as usize;
            let amount = node.amount();
            let pot_size = game.tree_config().starting_pot + 2 * amount;
            drop(node);

            // At fold terminal, one player wins the pot
            let mut evs = vec![0.0f32; num_hands];
            for (hand_idx, _) in game.private_cards(player).iter().enumerate() {
                if folder == player {
                    // This player folded - loses contribution
                    evs[hand_idx] = -(amount as f32);
                } else {
                    // Other player folded - wins pot
                    evs[hand_idx] = (pot_size - amount) as f32;
                }
            }
            return evs;
        }

        if node.is_chance() {
            // Chance node (river deal): try to get cfvalues, or recurse into children
            let cfvalue_player = node.cfvalue_storage_player();

            if cfvalue_player == Some(player) {
                // EVs stored directly for this player
                let evs = if is_compressed {
                    let scale = node.cfvalue_chance_scale();
                    node.cfvalues_chance_compressed()
                        .iter()
                        .map(|&v| v as f32 * scale / i16::MAX as f32)
                        .collect()
                } else {
                    node.cfvalues_chance().to_vec()
                };
                drop(node);
                return evs;
            }

            // Need to traverse river children
            let num_children = node.num_actions();
            let children_offset = node.children_offset() as usize;
            drop(node);

            let mut evs = vec![0.0f32; num_hands];
            let mut counts = vec![0.0f32; num_hands];

            for child_idx in 0..num_children {
                let child_node_index = node_index + children_offset + child_idx;
                let child_evs = self.compute_subtree_evs(game, child_node_index, player);

                for (hand_idx, &ev) in child_evs.iter().enumerate() {
                    if hand_idx < num_hands && !ev.is_nan() {
                        evs[hand_idx] += ev;
                        counts[hand_idx] += 1.0;
                    }
                }
            }

            for i in 0..num_hands {
                if counts[i] > 0.0 {
                    evs[i] /= counts[i];
                }
            }

            return evs;
        }

        // Action node: compute weighted average of children based on strategies
        let acting_player = (node.player() & 0x03) as usize;
        let num_actions = node.num_actions();
        let children_offset = node.children_offset() as usize;

        // Try to get strategy probabilities
        let strategy: Vec<Vec<f32>> = if acting_player == player {
            // Get this player's strategy at this node
            self.get_node_strategy(game, &node, num_actions, num_hands, is_compressed)
        } else {
            // For opponent's node, we need their strategy
            let opp_hands = game.private_cards(acting_player).len();
            self.get_node_strategy(game, &node, num_actions, opp_hands, is_compressed)
        };

        drop(node);

        // Compute EVs for each action
        let mut action_evs: Vec<Vec<f32>> = Vec::new();
        for action_idx in 0..num_actions {
            let child_node_index = node_index + children_offset + action_idx;
            action_evs.push(self.compute_subtree_evs(game, child_node_index, player));
        }

        // Compute expected EV based on strategy
        let mut evs = vec![0.0f32; num_hands];

        if acting_player == player {
            // This player is acting - use their strategy to weight EVs
            for hand_idx in 0..num_hands {
                let mut ev = 0.0f32;
                for action_idx in 0..num_actions {
                    if action_idx < strategy.len() && hand_idx < strategy[action_idx].len() {
                        let prob = strategy[action_idx][hand_idx];
                        if action_idx < action_evs.len() && hand_idx < action_evs[action_idx].len() {
                            ev += prob * action_evs[action_idx][hand_idx];
                        }
                    }
                }
                evs[hand_idx] = ev;
            }
        } else {
            // Opponent is acting - average over opponent's strategy (simplified)
            // This is an approximation - proper would weight by opponent reach
            for hand_idx in 0..num_hands {
                let mut ev = 0.0f32;
                let mut total = 0.0f32;
                for action_idx in 0..num_actions {
                    if action_idx < action_evs.len() && hand_idx < action_evs[action_idx].len() {
                        // Use uniform weighting as approximation
                        ev += action_evs[action_idx][hand_idx];
                        total += 1.0;
                    }
                }
                evs[hand_idx] = if total > 0.0 { ev / total } else { 0.0 };
            }
        }

        evs
    }

    /// Get the strategy at a node (probability of each action for each hand)
    fn get_node_strategy(
        &self,
        _game: &PostFlopGame,
        node: &crate::game::PostFlopNode,
        num_actions: usize,
        num_hands: usize,
        is_compressed: bool,
    ) -> Vec<Vec<f32>> {
        let mut strategy = vec![vec![0.0f32; num_hands]; num_actions];

        // Get raw strategy values from node
        // The strategy is stored as regret-matched probabilities
        for action_idx in 0..num_actions {
            for hand_idx in 0..num_hands {
                let idx = action_idx * num_hands + hand_idx;
                let val = if is_compressed {
                    let slice = node.strategy_compressed();
                    if idx < slice.len() {
                        let scale = node.strategy_scale();
                        slice[idx] as f32 * scale / i16::MAX as f32
                    } else {
                        1.0 / num_actions as f32 // Default uniform
                    }
                } else {
                    let slice = node.strategy();
                    if idx < slice.len() {
                        slice[idx]
                    } else {
                        1.0 / num_actions as f32
                    }
                };
                strategy[action_idx][hand_idx] = val;
            }
        }

        // Normalize to ensure probabilities sum to 1 for each hand
        for hand_idx in 0..num_hands {
            let mut sum = 0.0f32;
            for action_idx in 0..num_actions {
                sum += strategy[action_idx][hand_idx].max(0.0);
            }
            if sum > 0.0 {
                for action_idx in 0..num_actions {
                    strategy[action_idx][hand_idx] = strategy[action_idx][hand_idx].max(0.0) / sum;
                }
            } else {
                // Uniform if all zeros
                for action_idx in 0..num_actions {
                    strategy[action_idx][hand_idx] = 1.0 / num_actions as f32;
                }
            }
        }

        strategy
    }

    /// Compute flop equities for each hand against opponent's range
    fn compute_flop_equities(&self, game: &PostFlopGame, player: usize) -> Vec<f32> {
        let private_cards = game.private_cards(player);
        let opp_cards = game.private_cards(player ^ 1);
        let opp_weights = game.initial_weights(player ^ 1);

        let flop = game.card_config().flop;
        let mut board_mask: u64 = 0;
        for &card in &flop {
            board_mask |= 1u64 << card;
        }

        let mut equities = Vec::with_capacity(private_cards.len());

        for &(c1, c2) in private_cards {
            let hand_mask = (1u64 << c1) | (1u64 << c2);

            // Skip if hand conflicts with board
            if hand_mask & board_mask != 0 {
                equities.push(0.0);
                continue;
            }

            let mut total_weight = 0.0f64;
            let mut win_weight = 0.0f64;

            for (opp_idx, &(oc1, oc2)) in opp_cards.iter().enumerate() {
                let opp_mask = (1u64 << oc1) | (1u64 << oc2);

                // Skip if opponent hand conflicts with our hand or board
                if opp_mask & (hand_mask | board_mask) != 0 {
                    continue;
                }

                let weight = opp_weights[opp_idx] as f64;
                if weight <= 0.0 {
                    continue;
                }

                total_weight += weight;

                // Simple equity calculation based on hand rankings
                // This is an approximation - proper implementation would enumerate runouts
                let our_strength = self.simple_hand_strength(c1, c2, &flop);
                let opp_strength = self.simple_hand_strength(oc1, oc2, &flop);

                if our_strength > opp_strength {
                    win_weight += weight;
                } else if our_strength == opp_strength {
                    win_weight += weight * 0.5;
                }
            }

            let equity = if total_weight > 0.0 {
                (win_weight / total_weight) as f32
            } else {
                0.5
            };
            equities.push(equity);
        }

        equities
    }

    /// Simple hand strength on flop (approximation)
    fn simple_hand_strength(&self, c1: Card, c2: Card, flop: &[Card; 3]) -> u32 {
        let rank1 = c1 >> 2;
        let rank2 = c2 >> 2;
        let suit1 = c1 & 3;
        let suit2 = c2 & 3;

        let flop_ranks: Vec<u8> = flop.iter().map(|&c| c >> 2).collect();
        let flop_suits: Vec<u8> = flop.iter().map(|&c| c & 3).collect();

        let high = rank1.max(rank2);
        let low = rank1.min(rank2);
        let paired = rank1 == rank2;

        // Check for trips
        let trips = flop_ranks.iter().filter(|&&r| r == rank1 || r == rank2).count();
        if trips >= 2 && paired {
            return 7000 + high as u32; // Full house potential
        }
        if trips >= 2 {
            return 6000 + high as u32; // Trips
        }

        // Check for two pair
        let board_pairs: Vec<u8> = flop_ranks.iter().filter(|&&r| r == rank1 || r == rank2).cloned().collect();
        if board_pairs.len() == 2 && !paired {
            return 5000 + high as u32 * 15 + low as u32; // Two pair
        }

        // Check for pair
        if paired {
            // Pocket pair
            let overpair = flop_ranks.iter().all(|&r| rank1 > r);
            if overpair {
                return 4500 + high as u32 * 15; // Overpair
            }
            return 4000 + high as u32 * 15; // Pocket pair
        }

        if board_pairs.len() == 1 {
            // One pair with board
            let pair_rank = board_pairs[0];
            let kicker = if pair_rank == rank1 { rank2 } else { rank1 };
            return 3000 + pair_rank as u32 * 15 + kicker as u32; // Top/middle/bottom pair
        }

        // Flush draw
        let flush_suit = if suit1 == suit2 { Some(suit1) } else { None };
        let flush_draw = flush_suit.map_or(false, |s| flop_suits.iter().filter(|&&fs| fs == s).count() >= 2);

        // Straight draw (simplified)
        let all_ranks: Vec<u8> = vec![rank1, rank2, flop_ranks[0], flop_ranks[1], flop_ranks[2]];
        let straight_potential = self.count_straight_outs(&all_ranks);

        // High card with draws
        let base = high as u32 * 15 + low as u32;
        if flush_draw && straight_potential >= 4 {
            return 2500 + base; // Combo draw
        }
        if flush_draw {
            return 2000 + base; // Flush draw
        }
        if straight_potential >= 4 {
            return 1500 + base; // OESD
        }
        if straight_potential >= 3 {
            return 1000 + base; // Gutshot
        }

        // Pure high card
        base
    }

    fn count_straight_outs(&self, ranks: &[u8]) -> u8 {
        let mut rank_set: u64 = 0;
        for &r in ranks {
            rank_set |= 1u64 << r;
            if r == 12 { // Ace can be low
                rank_set |= 1u64 << 0; // Pseudo-low ace
            }
        }

        // Count 4-card sequences (potential OESD)
        let mut max_connected = 0u8;
        for start in 0..=9 {
            let mask = 0b11111u64 << start;
            let count = (rank_set & mask).count_ones() as u8;
            max_connected = max_connected.max(count);
        }
        max_connected
    }

    /// Look up EV for a hand at a given path with Range-Agnostic transfer
    ///
    /// Formula: EV_final = Precomputed_EV × (Current_Equity / Precomputed_Equity)
    pub fn get_ev_with_transfer(
        &self,
        path_hash: u64,
        player: usize,
        hand: (Card, Card),
        current_equity: f32,
    ) -> Option<f32> {
        let key = if hand.0 < hand.1 { hand } else { (hand.1, hand.0) };

        if let Some(terminal) = self.terminals.get(&path_hash) {
            let data = if player == 0 { &terminal.oop_data } else { &terminal.ip_data };
            if let Some(&(precomputed_ev, precomputed_equity)) = data.get(&key) {
                // Range-Agnostic transfer: scale EV by equity ratio
                if precomputed_equity > 0.001 {
                    let ev_final = precomputed_ev * (current_equity / precomputed_equity);
                    return Some(ev_final);
                } else if current_equity < 0.001 {
                    // Both near zero - use precomputed directly
                    return Some(precomputed_ev);
                }
            }
        }
        None
    }

    /// Look up raw EV (no transfer) for debugging
    pub fn get_raw_ev(&self, path_hash: u64, player: usize, hand: (Card, Card)) -> Option<(f32, f32)> {
        let key = if hand.0 < hand.1 { hand } else { (hand.1, hand.0) };

        if let Some(terminal) = self.terminals.get(&path_hash) {
            let data = if player == 0 { &terminal.oop_data } else { &terminal.ip_data };
            return data.get(&key).copied();
        }
        None
    }

    /// Get the pot size at a terminal for normalization
    pub fn get_pot_size(&self, path_hash: u64) -> Option<i32> {
        self.terminals.get(&path_hash).map(|t| t.pot_size)
    }

    /// Save EV map to file
    #[cfg(feature = "bincode")]
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        let config = bincode::config::standard();
        bincode::encode_into_std_write(self, &mut writer, config)
            .map_err(|e| format!("Failed to encode: {}", e))?;

        writer.flush().map_err(|e| format!("Failed to flush: {}", e))?;

        println!("Saved EV map with {} terminals to {}", self.terminals.len(), path);
        Ok(())
    }

    /// Load EV map from file
    #[cfg(feature = "bincode")]
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);

        let config = bincode::config::standard();
        let ev_map: Self = bincode::decode_from_std_read(&mut std::io::BufReader::new(reader), config)
            .map_err(|e| format!("Failed to decode: {}", e))?;

        if !ev_map.version.starts_with("ev-map-") {
            return Err(format!(
                "Version mismatch: expected 'ev-map-*', got '{}'",
                ev_map.version
            ));
        }

        println!("Loaded EV map with {} terminals from {}", ev_map.terminals.len(), path);
        Ok(ev_map)
    }
}

/// Trait for providing static EV values during DCFR iteration.
/// This is the "Perfect Leaf Evaluator" for flop-only solving.
pub trait StaticEvProvider: Send + Sync {
    /// Get the EV for a hand at a given action path.
    /// Returns (EV, should_use_static) - if should_use_static is false, use normal evaluation.
    fn get_static_ev(
        &self,
        path_hash: u64,
        player: usize,
        hand: (Card, Card),
        current_equity: f32,
    ) -> Option<f32>;

    /// Check if this provider has data for the given path
    fn has_path(&self, path_hash: u64) -> bool;
}

impl StaticEvProvider for EvMap {
    fn get_static_ev(
        &self,
        path_hash: u64,
        player: usize,
        hand: (Card, Card),
        current_equity: f32,
    ) -> Option<f32> {
        self.get_ev_with_transfer(path_hash, player, hand, current_equity)
    }

    fn has_path(&self, path_hash: u64) -> bool {
        self.terminals.contains_key(&path_hash)
    }
}

/// A no-op provider that always returns None (use normal evaluation)
pub struct NoOpEvProvider;

impl StaticEvProvider for NoOpEvProvider {
    fn get_static_ev(
        &self,
        _path_hash: u64,
        _player: usize,
        _hand: (Card, Card),
        _current_equity: f32,
    ) -> Option<f32> {
        None
    }

    fn has_path(&self, _path_hash: u64) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ev_map_creation() {
        let flop = [0, 4, 8]; // 2c 3c 4c
        let ev_map = EvMap::new(flop);
        assert_eq!(ev_map.version, "ev-map-v2");
        assert!(ev_map.terminals.is_empty());
    }

    #[test]
    fn test_range_transfer() {
        let flop = [0, 4, 8];
        let mut ev_map = EvMap::new(flop);

        // Add a fake terminal
        let mut oop_data = HashMap::new();
        let mut ip_data = HashMap::new();

        // Hand with EV=100, Equity=0.6
        oop_data.insert((0, 4), (100.0, 0.6));
        ip_data.insert((8, 12), (50.0, 0.4));

        ev_map.terminals.insert(0, FlopTerminalEv {
            path_hash: 0,
            pot_size: 100,
            oop_data,
            ip_data,
        });

        // Test equity-ratio scaling
        // If current equity is 0.8 (better than 0.6), EV should scale up
        let ev = ev_map.get_ev_with_transfer(0, 0, (0, 4), 0.8);
        assert!(ev.is_some());
        let scaled_ev = ev.unwrap();
        // EV = 100 * (0.8 / 0.6) = 133.33
        assert!((scaled_ev - 133.33).abs() < 1.0);

        // If current equity is 0.3 (worse than 0.6), EV should scale down
        let ev = ev_map.get_ev_with_transfer(0, 0, (0, 4), 0.3);
        let scaled_ev = ev.unwrap();
        // EV = 100 * (0.3 / 0.6) = 50
        assert!((scaled_ev - 50.0).abs() < 1.0);
    }
}
