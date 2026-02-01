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

        // Debug: Print number of hands and first 5 for each player
        let oop_hands = game.private_cards(0);
        let ip_hands = game.private_cards(1);
        println!("\n=== EV MAP EXTRACTION DEBUG ===");
        println!("OOP hands: {} total", oop_hands.len());
        println!("First 5 OOP hands: {:?}", oop_hands.iter().take(5).collect::<Vec<_>>());
        println!("IP hands: {} total", ip_hands.len());
        println!("First 5 IP hands: {:?}", ip_hands.iter().take(5).collect::<Vec<_>>());
        println!("================================\n");

        // Traverse tree and find flop-terminal nodes
        ev_map.extract_terminals_recursive(game, 0)?;

        println!("Extracted {} flop-terminal EV sets", ev_map.terminals.len());

        // Validate the extracted EVs
        ev_map.validate_extraction()?;

        Ok(ev_map)
    }

    /// Validate that extracted EVs are reasonable (non-zero, proper magnitudes)
    fn validate_extraction(&self) -> Result<(), String> {
        println!("\n=== EV Map Validation ===");
        println!("Total terminals: {}", self.terminals.len());
        if self.terminals.is_empty() {
            return Err("No terminals extracted! Check if chance nodes exist at flop level.".to_string());
        }

        for (_path_hash, terminal) in &self.terminals {
            let mut oop_zeros = 0;
            let mut oop_total = 0;
            let mut oop_sum = 0.0f64;
            let mut ip_zeros = 0;
            let mut ip_total = 0;
            let mut ip_sum = 0.0f64;

            for &(ev, _equity) in terminal.oop_data.values() {
                oop_total += 1;
                oop_sum += ev as f64;
                if ev.abs() < 0.001 {
                    oop_zeros += 1;
                }
            }

            for &(ev, _equity) in terminal.ip_data.values() {
                ip_total += 1;
                ip_sum += ev as f64;
                if ev.abs() < 0.001 {
                    ip_zeros += 1;
                }
            }

            let oop_avg = if oop_total > 0 { oop_sum / oop_total as f64 } else { 0.0 };
            let ip_avg = if ip_total > 0 { ip_sum / ip_total as f64 } else { 0.0 };

            println!(
                "Terminal pot={}: OOP avg_ev={:.2}, zeros={}/{} ({:.1}%) | IP avg_ev={:.2}, zeros={}/{} ({:.1}%)",
                terminal.pot_size,
                oop_avg,
                oop_zeros,
                oop_total,
                100.0 * oop_zeros as f64 / oop_total.max(1) as f64,
                ip_avg,
                ip_zeros,
                ip_total,
                100.0 * ip_zeros as f64 / ip_total.max(1) as f64
            );

            // Check for suspicious patterns
            let oop_zero_pct = oop_zeros as f64 / oop_total.max(1) as f64;
            let ip_zero_pct = ip_zeros as f64 / ip_total.max(1) as f64;

            // Detect fold terminals: one player has EV ≈ pot (winner), other has EV ≈ 0 (folder with no chips in)
            // This is legitimate when folder's amount=0, so their EV for folding is 0
            let is_oop_fold_terminal = oop_avg.abs() < 0.01 && (ip_avg - terminal.pot_size as f64).abs() < 1.0;
            let is_ip_fold_terminal = ip_avg.abs() < 0.01 && (oop_avg - terminal.pot_size as f64).abs() < 1.0;

            if oop_zero_pct > 0.9 && !is_oop_fold_terminal {
                return Err(format!(
                    "VALIDATION FAILED: Terminal pot={} has {}% zero OOP EVs - extraction likely failed!",
                    terminal.pot_size,
                    (oop_zero_pct * 100.0) as i32
                ));
            }

            if ip_zero_pct > 0.9 && !is_ip_fold_terminal {
                return Err(format!(
                    "VALIDATION FAILED: Terminal pot={} has {}% zero IP EVs - extraction likely failed!",
                    terminal.pot_size,
                    (ip_zero_pct * 100.0) as i32
                ));
            }

            // Log detected fold terminals
            if is_oop_fold_terminal || is_ip_fold_terminal {
                println!("  (Detected as fold terminal - {} folds with amount=0)",
                    if is_oop_fold_terminal { "OOP" } else { "IP" });
            }
        }

        println!("=== Validation PASSED ===\n");
        Ok(())
    }

    /// Recursively find flop-terminal nodes and extract EVs
    ///
    /// We extract EVs for CHANCE nodes only (where the turn is about to be dealt).
    /// Fold terminals are handled by standard evaluation logic.
    ///
    /// Key format: Compound key from (pot_size, prev_action_type)
    /// This uniquely identifies chance nodes because:
    /// - pot_size captures the total betting that occurred
    /// - prev_action distinguishes check-check from bet-call sequences
    fn extract_terminals_recursive(
        &mut self,
        game: &PostFlopGame,
        node_index: usize,
    ) -> Result<(), String> {
        let node = game.node_arena()[node_index].lock();

        // Check if this is a flop-level node (turn not dealt yet)
        let is_flop_level = node.turn() == crate::card::NOT_DEALT;

        if node.is_terminal() {
            // Fold terminals are handled by standard evaluation logic (not EV map)
            return Ok(());
        }

        if node.is_chance() {
            // Chance node at flop level = deal turn = flop action is complete
            if is_flop_level {
                let pot_size = game.tree_config().starting_pot + 2 * node.amount();
                let prev_action = node.prev_action();
                let path_hash = Self::compute_compound_key(pot_size, prev_action);
                println!("[EV_MAP] Extracting chance node: pot={} prev_action={:?} hash={}",
                    pot_size, prev_action, path_hash);
                let terminal_ev = self.extract_showdown_terminal(game, node_index, path_hash)?;
                self.terminals.insert(path_hash, terminal_ev);
            }
            // Don't recurse into turn/river - we only care about flop terminals
            return Ok(());
        }

        // Action node - recurse into children
        let num_actions = node.num_actions();
        let children_offset = node.children_offset() as usize;
        drop(node);

        for action_idx in 0..num_actions {
            let child_index = node_index + children_offset + action_idx;
            self.extract_terminals_recursive(game, child_index)?;
        }

        Ok(())
    }

    /// Compute a compound key from (pot_size, prev_action)
    /// This should uniquely identify chance nodes at flop level
    pub fn compute_compound_key(pot_size: i32, prev_action: crate::action_tree::Action) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        pot_size.hash(&mut hasher);
        // Hash the action type and amount (if any)
        std::mem::discriminant(&prev_action).hash(&mut hasher);
        match prev_action {
            crate::action_tree::Action::Bet(amt) |
            crate::action_tree::Action::Raise(amt) |
            crate::action_tree::Action::AllIn(amt) => amt.hash(&mut hasher),
            _ => {}
        }
        hasher.finish()
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
        let pot_size = game.tree_config().starting_pot + 2 * node.amount();
        drop(node);

        // For debugging: track if we're getting all zeros
        let mut total_non_zero = 0usize;

        // Traverse all turn children
        for child_idx in 0..num_children {
            let child_node_index = chance_node_index + children_offset + child_idx;

            // Get EVs from this turn subtree
            let child_evs = self.compute_subtree_evs(game, child_node_index, player, 0);

            // Accumulate EVs (each turn card contributes equally)
            for (hand_idx, &ev) in child_evs.iter().enumerate() {
                if hand_idx < num_hands && !ev.is_nan() {
                    evs[hand_idx] += ev;
                    counts[hand_idx] += 1.0;
                    if ev.abs() > 0.001 {
                        total_non_zero += 1;
                    }
                }
            }
        }

        // Average over all turn cards
        for i in 0..num_hands {
            if counts[i] > 0.0 {
                evs[i] /= counts[i];
            }
        }

        // Debug: warn if all zeros
        if total_non_zero == 0 {
            eprintln!("[EV_MAP DEBUG] Traversal for player {} at pot={} returned all zeros! num_children={}",
                player, pot_size, num_children);
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
        depth: usize,
    ) -> Vec<f32> {
        use crate::action_tree::PLAYER_FOLD_FLAG;

        let node = game.node_arena()[node_index].lock();
        let num_hands = game.private_cards(player).len();
        let is_compressed = game.is_compression_enabled();

        if node.is_terminal() {
            let player_byte = node.player() as u8;
            let amount = node.amount();
            let pot_size = game.tree_config().starting_pot + 2 * amount;

            // Check if this is a FOLD terminal (PLAYER_FOLD_FLAG = 24 = 0x18)
            let is_fold = (player_byte & PLAYER_FOLD_FLAG) == PLAYER_FOLD_FLAG;

            if is_fold {
                // Fold terminal: one player folded
                let folder = (player_byte & 0x03) as usize;
                drop(node);

                let mut evs = vec![0.0f32; num_hands];
                for hand_idx in 0..num_hands {
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

            // Showdown terminal: compute EVs using hand strength
            let turn_card = node.turn();
            let river_card = node.river();
            drop(node);

            // If we don't have complete board, return zeros (shouldn't happen at river)
            if turn_card == crate::card::NOT_DEALT || river_card == crate::card::NOT_DEALT {
                return vec![0.0f32; num_hands];
            }

            // Compute showdown EVs using hand strengths
            return self.compute_showdown_evs(game, player, turn_card, river_card, pot_size);
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
                let child_evs = self.compute_subtree_evs(game, child_node_index, player, depth + 1);

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

        // Action node: try to use stored cfvalues first
        let acting_player = (node.player() & 0x03) as usize;
        let num_actions = node.num_actions();
        let children_offset = node.children_offset() as usize;

        // Try to use stored cfvalues directly at this action node
        // At action nodes: cfvalues = acting player's CF values per action
        //                  cfvalues_ip = IP's CF values when OOP acts
        let stored_cfvalues: Option<Vec<f32>> = if acting_player == player {
            // Acting player's cfvalues are in storage2
            let cf_len = num_actions * num_hands;
            if is_compressed {
                let slice = node.cfvalues_compressed();
                if slice.len() >= cf_len {
                    let scale = node.cfvalue_scale();
                    Some(slice[..cf_len].iter().map(|&v| v as f32 * scale / i16::MAX as f32).collect())
                } else {
                    None
                }
            } else {
                let slice = node.cfvalues();
                if slice.len() >= cf_len {
                    Some(slice[..cf_len].to_vec())
                } else {
                    None
                }
            }
        } else if player == 1 && acting_player == 0 && node.has_cfvalues_ip() {
            // IP wants EVs, OOP is acting: use cfvalues_ip
            let cf_len = num_actions * num_hands;
            if is_compressed {
                let slice = node.cfvalues_ip_compressed();
                if slice.len() >= cf_len {
                    let scale = node.cfvalue_ip_scale();
                    Some(slice[..cf_len].iter().map(|&v| v as f32 * scale / i16::MAX as f32).collect())
                } else {
                    None
                }
            } else {
                let slice = node.cfvalues_ip();
                if slice.len() >= cf_len {
                    Some(slice[..cf_len].to_vec())
                } else {
                    None
                }
            }
        } else {
            None
        };

        // If we have stored cfvalues, use them with strategy to compute expected EV
        if let Some(cfv) = stored_cfvalues {
            // Get strategy for the acting player
            let strategy = self.get_node_strategy(game, &node, num_actions, num_hands, is_compressed);
            drop(node);

            // Compute expected EV: sum(strategy[action][hand] * cfvalue[action][hand])
            let mut evs = vec![0.0f32; num_hands];
            for hand_idx in 0..num_hands {
                let mut ev = 0.0f32;
                for action_idx in 0..num_actions {
                    let cf_idx = action_idx * num_hands + hand_idx;
                    if cf_idx < cfv.len() && action_idx < strategy.len() && hand_idx < strategy[action_idx].len() {
                        ev += strategy[action_idx][hand_idx] * cfv[cf_idx];
                    }
                }
                evs[hand_idx] = ev;
            }
            return evs;
        }

        // Fallback: compute EVs by recursing into children
        let strategy: Vec<Vec<f32>> = if acting_player == player {
            self.get_node_strategy(game, &node, num_actions, num_hands, is_compressed)
        } else {
            let opp_hands = game.private_cards(acting_player).len();
            self.get_node_strategy(game, &node, num_actions, opp_hands, is_compressed)
        };

        drop(node);

        // Compute EVs for each action by recursing
        let mut action_evs: Vec<Vec<f32>> = Vec::new();
        for action_idx in 0..num_actions {
            let child_node_index = node_index + children_offset + action_idx;
            action_evs.push(self.compute_subtree_evs(game, child_node_index, player, depth + 1));
        }

        // Compute expected EV based on strategy
        let mut evs = vec![0.0f32; num_hands];

        if acting_player == player {
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
            // Opponent is acting - weight by opponent's average strategy
            let opp_weights = game.initial_weights(acting_player);
            let mut avg_strategy = vec![0.0f32; num_actions];
            let mut total_weight = 0.0f32;

            for (opp_hand_idx, &weight) in opp_weights.iter().enumerate() {
                if weight <= 0.0 {
                    continue;
                }
                total_weight += weight;
                for action_idx in 0..num_actions {
                    if action_idx < strategy.len() && opp_hand_idx < strategy[action_idx].len() {
                        avg_strategy[action_idx] += weight * strategy[action_idx][opp_hand_idx];
                    }
                }
            }

            if total_weight > 0.0 {
                for action_idx in 0..num_actions {
                    avg_strategy[action_idx] /= total_weight;
                }
            } else {
                for action_idx in 0..num_actions {
                    avg_strategy[action_idx] = 1.0 / num_actions as f32;
                }
            }

            for hand_idx in 0..num_hands {
                let mut ev = 0.0f32;
                for action_idx in 0..num_actions {
                    if action_idx < action_evs.len() && hand_idx < action_evs[action_idx].len() {
                        ev += avg_strategy[action_idx] * action_evs[action_idx][hand_idx];
                    }
                }
                evs[hand_idx] = ev;
            }
        }

        evs
    }

    /// Compute showdown EVs at a river terminal for a player.
    /// Uses hand strength data to determine win/lose/tie outcomes.
    fn compute_showdown_evs(
        &self,
        game: &PostFlopGame,
        player: usize,
        turn: crate::card::Card,
        river: crate::card::Card,
        pot_size: i32,
    ) -> Vec<f32> {
        use crate::card::card_pair_to_index;
        use crate::card::StrengthItem;

        let num_hands = game.private_cards(player).len();
        let mut evs = vec![0.0f32; num_hands];

        let pair_index = card_pair_to_index(turn, river);
        let hand_strength = game.hand_strength();

        if pair_index >= hand_strength.len() {
            return evs;
        }

        let player_strength = &hand_strength[pair_index][player];
        let opponent_strength = &hand_strength[pair_index][player ^ 1];

        if player_strength.len() < 3 || opponent_strength.len() < 3 {
            return evs;
        }

        let opp_weights = game.initial_weights(player ^ 1);
        let half_pot = pot_size as f32 / 2.0;

        // For each of our hands, compute showdown EV against opponent range
        // This is a simplified version - assumes uniform opponent reach
        let valid_player_strength = &player_strength[1..player_strength.len() - 1];
        let valid_opponent_strength = &opponent_strength[1..opponent_strength.len() - 1];

        // Count total opponent combinations and cumulative weights by strength
        let mut total_opp_weight = 0.0f64;
        for &StrengthItem { index, .. } in valid_opponent_strength {
            let weight = opp_weights.get(index as usize).copied().unwrap_or(0.0) as f64;
            total_opp_weight += weight;
        }

        if total_opp_weight <= 0.0 {
            return evs;
        }

        // For each player hand, compute EV based on equity vs opponent range
        for &StrengthItem { strength: player_str, index: player_idx } in valid_player_strength {
            let mut win_weight = 0.0f64;
            let mut tie_weight = 0.0f64;

            for &StrengthItem { strength: opp_str, index: opp_idx } in valid_opponent_strength {
                let weight = opp_weights.get(opp_idx as usize).copied().unwrap_or(0.0) as f64;
                if weight <= 0.0 {
                    continue;
                }

                if player_str > opp_str {
                    win_weight += weight;
                } else if player_str == opp_str {
                    tie_weight += weight;
                }
            }

            // EV = (win_prob * pot) + (tie_prob * 0) - (lose_prob * contribution)
            // Simplified: EV = equity * pot - half_pot (our contribution)
            let equity = (win_weight + 0.5 * tie_weight) / total_opp_weight;
            let ev = (equity as f32) * (pot_size as f32) - half_pot;

            if (player_idx as usize) < num_hands {
                evs[player_idx as usize] = ev;
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
    ///
    /// Note: The EV map only contains chance nodes (not fold terminals).
    /// Fold terminals are handled by standard evaluation logic.
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
                } else {
                    // Very low precomputed equity - use EV directly to avoid division issues
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
