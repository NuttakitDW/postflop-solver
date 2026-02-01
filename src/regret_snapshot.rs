//! Regret Snapshot: Save and load precomputed regrets for warm-starting solvers.
//!
//! This allows solving a board once, then reusing the regrets to quickly solve
//! the same board with different ranges/settings.

use crate::game::PostFlopGame;
use crate::interface::*;
use crate::card::Card;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

/// A snapshot of regrets for a solved game, organized for transfer to other games.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub struct RegretSnapshot {
    /// Version string for compatibility checking
    pub version: String,
    /// Flop cards (sorted for canonical form)
    pub flop: [Card; 3],
    /// Regrets organized by node path and player
    /// Key: (node_path_hash, player) -> HashMap<(card1, card2), regrets_per_action>
    pub node_regrets: Vec<NodeRegrets>,
}

/// Regrets for a single node
#[derive(Clone, Debug)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub struct NodeRegrets {
    /// Hash of the path to this node (actions taken to reach it)
    pub path_hash: u64,
    /// Player who acts at this node (0 = OOP, 1 = IP)
    pub player: u8,
    /// Number of actions at this node
    pub num_actions: usize,
    /// Turn card (NOT_DEALT if flop node)
    pub turn: Card,
    /// River card (NOT_DEALT if flop/turn node)
    pub river: Card,
    /// Regrets per hand: (card1, card2) -> [regret_action0, regret_action1, ...]
    pub hand_regrets: HashMap<(Card, Card), Vec<f32>>,
}

impl RegretSnapshot {
    /// Current version string
    const VERSION: &'static str = "regret-snapshot-v1";

    /// Create a new empty snapshot for a given flop
    pub fn new(flop: [Card; 3]) -> Self {
        let mut sorted_flop = flop;
        sorted_flop.sort();
        Self {
            version: Self::VERSION.to_string(),
            flop: sorted_flop,
            node_regrets: Vec::new(),
        }
    }

    /// Extract regrets from a solved game
    pub fn from_game(game: &PostFlopGame) -> Result<Self, String> {
        if !game.is_solved() {
            return Err("Game must be solved before extracting regrets".to_string());
        }

        let flop = game.card_config().flop;
        let mut snapshot = Self::new(flop);

        // Traverse the tree and extract regrets from each node
        snapshot.extract_regrets_recursive(game, 0, 0)?;

        Ok(snapshot)
    }

    /// Recursively extract regrets from the game tree
    fn extract_regrets_recursive(
        &mut self,
        game: &PostFlopGame,
        node_index: usize,
        path_hash: u64,
    ) -> Result<(), String> {
        let node = game.node_arena()[node_index].lock();

        if node.is_terminal() {
            return Ok(());
        }

        if node.is_chance() {
            // Recurse into all chance children
            let num_children = node.num_actions();
            let children_offset = node.children_offset() as usize;
            drop(node);

            for i in 0..num_children {
                let child_index = node_index + children_offset + i;
                let child_hash = path_hash.wrapping_mul(31).wrapping_add(i as u64 + 1000);
                self.extract_regrets_recursive(game, child_index, child_hash)?;
            }
            return Ok(());
        }

        // Action node - extract regrets
        let player = node.player() as u8;
        let num_actions = node.num_actions();
        let turn = node.turn();
        let river = node.river();

        // Get regrets from the node
        let regrets: Vec<f32> = if game.is_compression_enabled() {
            let scale = node.regret_scale();
            node.regrets_compressed()
                .iter()
                .map(|&r| r as f32 * scale / i16::MAX as f32)
                .collect()
        } else {
            node.regrets().to_vec()
        };

        let num_hands = game.num_private_hands(player as usize);
        let private_cards = game.private_cards(player as usize);

        // Build hand -> regrets mapping
        let mut hand_regrets: HashMap<(Card, Card), Vec<f32>> = HashMap::new();

        for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
            let mut action_regrets = Vec::with_capacity(num_actions);
            for action in 0..num_actions {
                let regret_idx = action * num_hands + hand_idx;
                if regret_idx < regrets.len() {
                    action_regrets.push(regrets[regret_idx]);
                } else {
                    action_regrets.push(0.0);
                }
            }
            // Store with canonical card order (lower card first)
            let key = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            hand_regrets.insert(key, action_regrets);
        }

        let node_regrets = NodeRegrets {
            path_hash,
            player,
            num_actions,
            turn,
            river,
            hand_regrets,
        };

        self.node_regrets.push(node_regrets);

        // Recurse into children
        let children_offset = node.children_offset() as usize;
        drop(node);

        for i in 0..num_actions {
            let child_index = node_index + children_offset + i;
            let child_hash = path_hash.wrapping_mul(31).wrapping_add(i as u64);
            self.extract_regrets_recursive(game, child_index, child_hash)?;
        }

        Ok(())
    }

    /// Apply regrets to a game (warm start)
    pub fn apply_to_game(&self, game: &mut PostFlopGame) -> Result<usize, String> {
        // Verify flop matches
        let mut game_flop = game.card_config().flop;
        game_flop.sort();
        if game_flop != self.flop {
            return Err(format!(
                "Flop mismatch: snapshot has {:?}, game has {:?}",
                self.flop, game_flop
            ));
        }

        // Build lookup from path_hash to NodeRegrets
        let regret_lookup: HashMap<u64, &NodeRegrets> = self
            .node_regrets
            .iter()
            .map(|nr| (nr.path_hash, nr))
            .collect();

        // Apply regrets recursively
        let applied = self.apply_regrets_recursive(game, 0, 0, &regret_lookup)?;

        Ok(applied)
    }

    /// Recursively apply regrets to game nodes
    fn apply_regrets_recursive(
        &self,
        game: &mut PostFlopGame,
        node_index: usize,
        path_hash: u64,
        regret_lookup: &HashMap<u64, &NodeRegrets>,
    ) -> Result<usize, String> {
        let mut applied = 0;

        let (is_terminal, is_chance, player, num_actions, children_offset, turn, river) = {
            let node = game.node_arena()[node_index].lock();
            (
                node.is_terminal(),
                node.is_chance(),
                node.player() as u8,
                node.num_actions(),
                node.children_offset() as usize,
                node.turn(),
                node.river(),
            )
        };

        if is_terminal {
            return Ok(0);
        }

        if is_chance {
            // Recurse into all chance children
            for i in 0..num_actions {
                let child_index = node_index + children_offset + i;
                let child_hash = path_hash.wrapping_mul(31).wrapping_add(i as u64 + 1000);
                applied += self.apply_regrets_recursive(game, child_index, child_hash, regret_lookup)?;
            }
            return Ok(applied);
        }

        // Try to find matching regrets for this node
        if let Some(node_regrets) = regret_lookup.get(&path_hash) {
            // Verify node properties match
            if node_regrets.player == player
                && node_regrets.num_actions == num_actions
                && node_regrets.turn == turn
                && node_regrets.river == river
            {
                // Apply regrets
                let num_hands = game.num_private_hands(player as usize);
                let private_cards = game.private_cards(player as usize).to_vec();

                let mut new_regrets = vec![0.0f32; num_actions * num_hands];

                for (hand_idx, &(c1, c2)) in private_cards.iter().enumerate() {
                    let key = if c1 < c2 { (c1, c2) } else { (c2, c1) };
                    if let Some(action_regrets) = node_regrets.hand_regrets.get(&key) {
                        for (action, &regret) in action_regrets.iter().enumerate() {
                            if action < num_actions {
                                new_regrets[action * num_hands + hand_idx] = regret;
                            }
                        }
                    }
                }

                // Write regrets to node
                let mut node = game.node_arena()[node_index].lock();
                if game.is_compression_enabled() {
                    let max_abs = new_regrets.iter().map(|r| r.abs()).fold(0.0f32, f32::max);
                    let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
                    node.set_regret_scale(scale);
                    let regrets = node.regrets_compressed_mut();
                    for (i, &r) in new_regrets.iter().enumerate() {
                        regrets[i] = (r / scale * i16::MAX as f32).round() as i16;
                    }
                } else {
                    let regrets = node.regrets_mut();
                    regrets.copy_from_slice(&new_regrets);
                }

                applied += 1;
            }
        }

        // Recurse into children
        for i in 0..num_actions {
            let child_index = node_index + children_offset + i;
            let child_hash = path_hash.wrapping_mul(31).wrapping_add(i as u64);
            applied += self.apply_regrets_recursive(game, child_index, child_hash, regret_lookup)?;
        }

        Ok(applied)
    }

    /// Save snapshot to file
    #[cfg(feature = "bincode")]
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        let mut writer = BufWriter::new(file);

        let config = bincode::config::standard();
        bincode::encode_into_std_write(self, &mut writer, config)
            .map_err(|e| format!("Failed to encode: {}", e))?;

        writer.flush().map_err(|e| format!("Failed to flush: {}", e))?;
        Ok(())
    }

    /// Load snapshot from file
    #[cfg(feature = "bincode")]
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let mut reader = BufReader::new(file);

        let config = bincode::config::standard();
        let snapshot: Self = bincode::decode_from_std_read(&mut reader, config)
            .map_err(|e| format!("Failed to decode: {}", e))?;

        if snapshot.version != Self::VERSION {
            return Err(format!(
                "Version mismatch: expected '{}', got '{}'",
                Self::VERSION,
                snapshot.version
            ));
        }

        Ok(snapshot)
    }
}
