//! Oracle lookup table for flop-only solving.
//!
//! Pre-computes MatrixTurnCfv for all (boundary_amount, turn_card) pairs,
//! saves to disk, and loads at inference time to solve only the flop.
//!
//! Pipeline:
//! 1. **Build**: solve turn games → extract MatrixTurnCfv → save `.oracle` file
//! 2. **Inference**: load `.oracle` → solve flop only (no turn/river solving)

use crate::action_tree::*;
use crate::card::*;
use crate::game::PostFlopGame;
use crate::game::PostFlopNode;
use crate::interface::*;
use crate::mutex_like::*;
use crate::range::Range;
use crate::sliceop::*;
use crate::turn_cfv::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::slice;
use std::time::Instant;

#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};

// ============================================================================
// Oracle Lookup Table
// ============================================================================

/// Precomputed oracle: MatrixTurnCfv for all (boundary_amount, turn_card) pairs.
///
/// At inference time, this replaces full turn/river solving. Later, replace
/// with a neural network that has the same interface.
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct OracleLookupTable {
    /// Flop cards this oracle was built for.
    flop: [Card; 3],
    /// Boundary amounts (distinct pot contributions at flop→turn boundary).
    boundary_amounts: Vec<i32>,
    /// Matrices indexed as entries[amount_idx * 52 + card].
    /// None for cards that conflict with the flop.
    entries: Vec<Option<MatrixTurnCfv>>,
    /// Starting pot used during oracle build.
    starting_pot: i32,
    /// Effective stack used during oracle build.
    effective_stack: i32,
}

impl OracleLookupTable {
    /// Builds the oracle by solving turn games and extracting matrices.
    ///
    /// For each (boundary_amount, turn_card): builds ExactTurnCfv (solves the
    /// full turn+river game), extracts the matrix via `from_exact()`, drops the tree.
    pub fn build(
        flop: [Card; 3],
        oop_range: &Range,
        ip_range: &Range,
        starting_pot: i32,
        effective_stack: i32,
        boundary_amounts: &[i32],
        bet_config: &TurnBetConfig,
        max_iterations: u32,
        target_exploitability: f32,
        print_progress: bool,
    ) -> Self {
        let flop_mask: u64 = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);
        let num_turn_cards = 52 - 3;
        let total_games = boundary_amounts.len() * num_turn_cards;
        let mut games_done = 0usize;
        let build_start = Instant::now();

        let mut entries: Vec<Option<MatrixTurnCfv>> =
            (0..boundary_amounts.len() * 52).map(|_| None).collect();

        for (amount_idx, &amount) in boundary_amounts.iter().enumerate() {
            let pot = starting_pot + 2 * amount;
            let stack = effective_stack - amount;
            let actual_stack = if stack > 0 { stack } else { 1 };

            if print_progress {
                eprintln!("  [{}/{}] amount={} (pot={}, stack={})",
                    amount_idx + 1, boundary_amounts.len(), amount, pot, actual_stack);
            }

            let amount_start = Instant::now();

            for card in 0u8..52 {
                if flop_mask & (1u64 << card) != 0 {
                    continue;
                }

                let exact = ExactTurnCfv::new(
                    flop, card, oop_range, ip_range,
                    pot, actual_stack, bet_config,
                    max_iterations, target_exploitability,
                ).unwrap();

                let matrix = MatrixTurnCfv::from_exact(&exact);
                entries[amount_idx * 52 + card as usize] = Some(matrix);

                games_done += 1;
                if print_progress {
                    let elapsed = build_start.elapsed().as_secs_f64();
                    let avg = elapsed / games_done as f64;
                    let remaining = avg * (total_games - games_done) as f64;
                    eprint!("\r    turn cards: {}/{} | total: {}/{} | ETA: {:.0}s   ",
                        games_done - amount_idx * num_turn_cards, num_turn_cards,
                        games_done, total_games, remaining);
                    io::stderr().flush().ok();
                }
            }

            if print_progress {
                let amount_time = amount_start.elapsed().as_secs_f64();
                eprintln!("\r    done: {} cards in {:.1}s                              ", num_turn_cards, amount_time);
            }
        }
        if print_progress {
            eprintln!("  All {} turn games completed in {:.1}s", total_games, build_start.elapsed().as_secs_f64());
        }

        Self {
            flop,
            boundary_amounts: boundary_amounts.to_vec(),
            entries,
            starting_pot,
            effective_stack,
        }
    }

    /// Total memory used by all matrices in bytes.
    pub fn matrix_memory_bytes(&self) -> usize {
        self.entries.iter()
            .filter_map(|e| e.as_ref())
            .map(|m| m.matrix_memory_bytes())
            .sum()
    }

    /// The flop cards this oracle was built for.
    pub fn flop(&self) -> [Card; 3] {
        self.flop
    }

    /// Boundary amounts.
    pub fn boundary_amounts(&self) -> &[i32] {
        &self.boundary_amounts
    }

    /// Number of entries (boundary_amounts × 52 slots).
    pub fn num_entries(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    /// Look up a MatrixTurnCfv for a specific (amount, turn_card).
    fn get(&self, amount: i32, card: Card) -> Option<&MatrixTurnCfv> {
        let amount_idx = self.boundary_amounts.iter().position(|&a| a == amount)?;
        self.entries[amount_idx * 52 + card as usize].as_ref()
    }

    /// Save oracle to file using bincode.
    #[cfg(feature = "bincode")]
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        let file = std::fs::File::create(path)
            .map_err(|e| format!("Failed to create oracle file: {}", e))?;
        let mut writer = std::io::BufWriter::new(file);
        bincode::encode_into_std_write(self, &mut writer, bincode::config::standard())
            .map_err(|e| format!("Failed to encode oracle: {}", e))?;
        Ok(())
    }

    /// Load oracle from file using bincode.
    #[cfg(feature = "bincode")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open oracle file: {}", e))?;
        let mut reader = std::io::BufReader::new(file);
        let oracle: Self = bincode::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| format!("Failed to decode oracle: {}", e))?;
        Ok(oracle)
    }
}

// ============================================================================
// Flop-only node
// ============================================================================

/// A node in the flop-only game tree.
pub struct FlopNode {
    player: u8,
    pub(crate) is_boundary: bool,
    amount: i32,
    children_offset: u32,
    num_children: u16,
    num_hands: u16,
    strategy: Vec<f32>,
    regrets: Vec<f32>,
    cfvalues_ip: Vec<f32>,
}

impl Default for FlopNode {
    fn default() -> Self {
        Self {
            player: PLAYER_OOP,
            is_boundary: false,
            amount: 0,
            children_offset: 0,
            num_children: 0,
            num_hands: 0,
            strategy: Vec::new(),
            regrets: Vec::new(),
            cfvalues_ip: Vec::new(),
        }
    }
}

unsafe impl Send for FlopNode {}
unsafe impl Sync for FlopNode {}

impl FlopNode {
    #[inline]
    fn children(&self) -> &[MutexLike<Self>] {
        if self.num_children == 0 {
            return &[];
        }
        let self_ptr = self as *const _ as *const MutexLike<FlopNode>;
        unsafe {
            slice::from_raw_parts(
                self_ptr.add(self.children_offset as usize),
                self.num_children as usize,
            )
        }
    }

    #[inline]
    pub fn amount(&self) -> i32 {
        self.amount
    }
}

impl GameNode for FlopNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.is_boundary || (self.player & PLAYER_TERMINAL_FLAG != 0)
    }

    #[inline]
    fn is_chance(&self) -> bool {
        false
    }

    #[inline]
    fn player(&self) -> usize {
        self.player as usize
    }

    #[inline]
    fn num_actions(&self) -> usize {
        self.num_children as usize
    }

    #[inline]
    fn play(&self, action: usize) -> MutexGuardLike<'_, Self> {
        self.children()[action].lock()
    }

    #[inline]
    fn strategy(&self) -> &[f32] {
        &self.strategy
    }

    #[inline]
    fn strategy_mut(&mut self) -> &mut [f32] {
        &mut self.strategy
    }

    #[inline]
    fn regrets(&self) -> &[f32] {
        &self.regrets
    }

    #[inline]
    fn regrets_mut(&mut self) -> &mut [f32] {
        &mut self.regrets
    }

    #[inline]
    fn cfvalues(&self) -> &[f32] {
        &self.regrets
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        &mut self.regrets
    }

    #[inline]
    fn has_cfvalues_ip(&self) -> bool {
        !self.cfvalues_ip.is_empty()
    }

    #[inline]
    fn cfvalues_ip(&self) -> &[f32] {
        &self.cfvalues_ip
    }

    #[inline]
    fn cfvalues_ip_mut(&mut self) -> &mut [f32] {
        &mut self.cfvalues_ip
    }
}

// ============================================================================
// Flop-only game
// ============================================================================

/// Flop-only game that uses an OracleLookupTable at boundary nodes.
pub struct FlopSolver {
    node_arena: Vec<MutexLike<FlopNode>>,
    card_config: CardConfig,
    tree_config: TreeConfig,
    num_private_hands: [usize; 2],
    initial_weights: [Vec<f32>; 2],
    private_cards: [Vec<(Card, Card)>; 2],
    same_hand_index: [Vec<u16>; 2],
    valid_indices_flop: [Vec<u16>; 2],
    num_combinations: f64,
    is_solved: bool,
    /// Oracle lookup table (loaded from file or built in-memory).
    oracle: Option<OracleLookupTable>,
    /// Flop hand index → turn hand index mapping per (card, player).
    flop_to_turn: Vec<Option<[Vec<usize>; 2]>>,
    /// Number of non-flop cards (typically 49).
    num_turn_cards: usize,
}

impl Game for FlopSolver {
    type Node = FlopNode;

    #[inline]
    fn root(&self) -> MutexGuardLike<'_, FlopNode> {
        self.node_arena[0].lock()
    }

    #[inline]
    fn num_private_hands(&self, player: usize) -> usize {
        self.num_private_hands[player]
    }

    #[inline]
    fn initial_weights(&self, player: usize) -> &[f32] {
        &self.initial_weights[player]
    }

    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        if node.is_boundary {
            self.evaluate_boundary(result, node, player, cfreach);
        } else {
            self.evaluate_fold(result, node, player, cfreach);
        }
    }

    #[inline]
    fn chance_factor(&self, _node: &FlopNode) -> usize {
        unreachable!("No chance nodes in flop-only tree")
    }

    #[inline]
    fn is_solved(&self) -> bool {
        self.is_solved
    }

    #[inline]
    fn set_solved(&mut self) {
        self.is_solved = true;
    }

    #[inline]
    fn is_ready(&self) -> bool {
        !self.node_arena.is_empty()
    }

    #[inline]
    fn starting_pot(&self) -> i32 {
        self.tree_config.starting_pot
    }
}

impl FlopSolver {
    /// Creates a new FlopSolver from config.
    pub fn new(
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<Self, String> {
        let (tree_config, _added, _removed, action_root) = action_tree.eject();

        if tree_config.initial_state != BoardState::Flop {
            return Err("FlopSolver requires initial_state == Flop".to_string());
        }

        let flop = card_config.flop;
        let board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        let mut private_cards: [Vec<(Card, Card)>; 2] = [Vec::new(), Vec::new()];
        let mut initial_weights: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for player in 0..2 {
            let (hands, weights) = card_config.range[player].get_hands_weights(board_mask);
            initial_weights[player] = weights;
            private_cards[player] = hands;
        }
        let num_private_hands = [private_cards[0].len(), private_cards[1].len()];

        // Same-hand index
        let mut same_hand_index: [Vec<u16>; 2] = [Vec::new(), Vec::new()];
        for player in 0..2 {
            let opp = &private_cards[player ^ 1];
            for hand in &private_cards[player] {
                same_hand_index[player].push(
                    opp.binary_search(hand).map_or(u16::MAX, |i| i as u16),
                );
            }
        }

        let (valid_indices_flop, _, _) = card_config.valid_indices(&private_cards);

        let mut num_combinations = 0.0f64;
        for (&(c1, c2), &w1) in private_cards[0].iter().zip(initial_weights[0].iter()) {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            for (&(c3, c4), &w2) in private_cards[1].iter().zip(initial_weights[1].iter()) {
                if oop_mask & ((1u64 << c3) | (1u64 << c4)) == 0 {
                    num_combinations += w1 as f64 * w2 as f64;
                }
            }
        }
        if num_combinations == 0.0 {
            return Err("No valid card combinations".to_string());
        }

        // Flop→turn hand index mappings
        let mut flop_to_turn: Vec<Option<[Vec<usize>; 2]>> = Vec::with_capacity(52);
        let mut num_turn_cards = 0usize;
        for card in 0u8..52 {
            if board_mask & (1u64 << card) != 0 {
                flop_to_turn.push(None);
                continue;
            }
            num_turn_cards += 1;
            let mut mappings: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            for player in 0..2 {
                let mut turn_idx = 0usize;
                for &(c1, c2) in &private_cards[player] {
                    if c1 == card || c2 == card {
                        mappings[player].push(usize::MAX);
                    } else {
                        mappings[player].push(turn_idx);
                        turn_idx += 1;
                    }
                }
            }
            flop_to_turn.push(Some(mappings));
        }

        // Build flop-only tree
        let num_nodes = count_flop_nodes(&action_root.lock());
        let node_arena: Vec<MutexLike<FlopNode>> = (0..num_nodes)
            .map(|_| MutexLike::new(FlopNode::default()))
            .collect();

        let mut solver = Self {
            node_arena,
            card_config,
            tree_config,
            num_private_hands,
            initial_weights,
            private_cards,
            same_hand_index,
            valid_indices_flop,
            num_combinations,
            is_solved: false,
            oracle: None,
            flop_to_turn,
            num_turn_cards,
        };

        let mut next_index = 1;
        solver.build_tree(0, &action_root.lock(), &mut next_index);
        solver.allocate_storage();
        Ok(solver)
    }

    pub fn num_nodes(&self) -> usize {
        self.node_arena.len()
    }

    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    pub fn card_config(&self) -> &CardConfig {
        &self.card_config
    }

    pub fn private_cards(&self, player: usize) -> &[(Card, Card)] {
        &self.private_cards[player]
    }

    /// Sets the oracle lookup table.
    pub fn set_oracle(&mut self, oracle: OracleLookupTable) {
        self.oracle = Some(oracle);
    }

    /// Collects distinct boundary amounts from boundary nodes.
    pub fn boundary_amounts(&self) -> Vec<i32> {
        let mut amounts = Vec::new();
        for i in 0..self.node_arena.len() {
            let node = self.node_arena[i].lock();
            if node.is_boundary && !amounts.contains(&node.amount) {
                amounts.push(node.amount);
            }
        }
        amounts
    }

    /// Evaluate boundary node: uses oracle to compute turn+river CFVs.
    fn evaluate_boundary(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        result.iter_mut().for_each(|r| { r.write(0.0); });
        let result_f32 = unsafe { &mut *(result as *mut [MaybeUninit<f32>] as *mut [f32]) };

        let oracle = match &self.oracle {
            Some(o) => o,
            None => return,
        };

        let opponent = player ^ 1;

        for card in 0u8..52 {
            let mapping = match &self.flop_to_turn[card as usize] {
                Some(m) => m,
                None => continue,
            };

            let matrix = match oracle.get(node.amount, card) {
                Some(m) => m,
                None => continue,
            };

            // Map cfreach from flop indexing to turn indexing
            let num_turn_hands_opp = matrix.num_private_hands(opponent);
            let mut turn_cfreach = vec![0.0f32; num_turn_hands_opp];
            for (flop_idx, &turn_idx) in mapping[opponent].iter().enumerate() {
                if turn_idx != usize::MAX {
                    turn_cfreach[turn_idx] = cfreach[flop_idx];
                }
            }

            // Matrix-vector multiply
            let turn_cfvs = matrix.evaluate(player, &turn_cfreach);

            // Map back to flop indexing
            for (flop_idx, &turn_idx) in mapping[player].iter().enumerate() {
                if turn_idx != usize::MAX {
                    result_f32[flop_idx] += turn_cfvs[turn_idx];
                }
            }
        }

        let scale = 1.0 / self.num_turn_cards as f32;
        for v in result_f32.iter_mut() {
            *v *= scale;
        }
    }

    /// Evaluate fold terminal node.
    fn evaluate_fold(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = f64::min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);

        let folded_player = node.player & PLAYER_MASK;
        let payoff = if folded_player as usize != player {
            (half_pot - rake) / self.num_combinations
        } else {
            -half_pot / self.num_combinations
        };

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let mut cfreach_sum = 0.0f64;
        let mut cfreach_minus = [0.0f64; 52];

        result.iter_mut().for_each(|v| { v.write(0.0); });
        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        for &i in &self.valid_indices_flop[player ^ 1] {
            unsafe {
                let cr = *cfreach.get_unchecked(i as usize);
                if cr != 0.0 {
                    let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                    let cr64 = cr as f64;
                    cfreach_sum += cr64;
                    *cfreach_minus.get_unchecked_mut(c1 as usize) += cr64;
                    *cfreach_minus.get_unchecked_mut(c2 as usize) += cr64;
                }
            }
        }

        if cfreach_sum == 0.0 {
            return;
        }

        for &i in &self.valid_indices_flop[player] {
            unsafe {
                let (c1, c2) = *player_cards.get_unchecked(i as usize);
                let same_i = *self.same_hand_index[player].get_unchecked(i as usize);
                let cfreach_same = if same_i == u16::MAX {
                    0.0
                } else {
                    *cfreach.get_unchecked(same_i as usize) as f64
                };
                let cr = cfreach_sum + cfreach_same
                    - *cfreach_minus.get_unchecked(c1 as usize)
                    - *cfreach_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(i as usize) = (payoff * cr) as f32;
            }
        }
    }

    fn build_tree(&self, node_index: usize, action_node: &ActionTreeNode, next_index: &mut usize) {
        let mut node = self.node_arena[node_index].lock();
        node.player = action_node.player;
        node.amount = action_node.amount;

        if action_node.player & PLAYER_TERMINAL_FLAG != 0 {
            return;
        }
        if action_node.player & PLAYER_CHANCE_FLAG != 0 {
            node.is_boundary = true;
            return;
        }

        let num_children = action_node.children.len();
        node.children_offset = (*next_index - node_index) as u32;
        node.num_children = num_children as u16;
        *next_index += num_children;

        for (i, child) in action_node.children.iter().enumerate() {
            let child_index = node_index + node.children_offset as usize + i;
            self.build_tree(child_index, &child.lock(), next_index);
        }
    }

    fn allocate_storage(&mut self) {
        for i in 0..self.node_arena.len() {
            let (player, num_actions, is_terminal, is_boundary, is_root) = {
                let node = self.node_arena[i].lock();
                (
                    node.player as usize & PLAYER_MASK as usize,
                    node.num_children as usize,
                    node.player & PLAYER_TERMINAL_FLAG != 0,
                    node.is_boundary,
                    i == 0,
                )
            };

            if is_terminal || is_boundary || num_actions == 0 {
                continue;
            }

            let num_hands = self.num_private_hands[player];
            let num_elements = num_actions * num_hands;

            let mut node = self.node_arena[i].lock();
            node.num_hands = num_hands as u16;
            node.strategy = vec![0.0; num_elements];
            node.regrets = vec![0.0; num_elements];

            if is_root {
                node.cfvalues_ip = vec![0.0; self.num_private_hands[PLAYER_IP as usize]];
            }
        }
    }
}

fn count_flop_nodes(node: &ActionTreeNode) -> usize {
    let mut count = 1;
    if node.player & PLAYER_TERMINAL_FLAG != 0 || node.player & PLAYER_CHANCE_FLAG != 0 {
        return count;
    }
    for child in &node.children {
        count += count_flop_nodes(&child.lock());
    }
    count
}

// ============================================================================
// DCFR solver for flop-only tree
// ============================================================================

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    fn new(current_iteration: u32, convergence_mode: bool) -> Self {
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };

        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;

        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);

        if convergence_mode {
            Self {
                alpha_t: ((pow_alpha / (pow_alpha + 1.0)) as f32).max(0.9),
                beta_t: 0.9,
                gamma_t: (pow_gamma as f32).max(0.9),
            }
        } else {
            Self {
                alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
                beta_t: 0.5,
                gamma_t: pow_gamma as f32,
            }
        }
    }
}

fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    unsafe { strategy.set_len(regret.len()) };

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Solves the flop using DCFR with the oracle providing boundary CFVs.
///
/// Returns exploitability of the converged strategy.
pub fn solve_flop(
    game: &mut FlopSolver,
    max_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if game.is_solved() {
        panic!("Already solved");
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);
    let starting_pot = game.starting_pot() as f32;
    let target_pct = if starting_pot > 0.0 {
        target_exploitability / starting_pot * 100.0
    } else {
        0.0
    };
    let mut convergence_mode = false;

    if print_progress {
        print!("iteration: 0 / {max_iterations} ");
        if starting_pot > 0.0 {
            let pct = exploitability / starting_pot * 100.0;
            print!("(exploitability = {pct:.2}% | target = {target_pct:.2}%)");
        }
        io::stdout().flush().unwrap();
    }

    for t in 0..max_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let is_power_of_4 = t > 0 && t == 1u32 << ((t.leading_zeros() ^ 31) & !1);
        if is_power_of_4 {
            exploitability = compute_exploitability(game);
        }

        if starting_pot > 0.0 && exploitability / starting_pot * 100.0 < 1.0 {
            convergence_mode = true;
        }

        let params = DiscountParams::new(t, convergence_mode);

        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive(
                result.spare_capacity_mut(),
                game, &mut root, player,
                game.initial_weights(player ^ 1),
                game.initial_weights(player),
                &params,
            );
        }

        let check = (t + 1) % 10 == 0 || t + 1 == max_iterations;
        if check {
            exploitability = compute_exploitability(game);
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_iterations);
            if starting_pot > 0.0 {
                let pct = exploitability / starting_pot * 100.0;
                print!("(exploitability = {pct:.2}% | target = {target_pct:.2}%)");
            }
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
    }

    finalize(game);
    exploitability
}

fn solve_recursive(
    result: &mut [MaybeUninit<f32>],
    game: &FlopSolver,
    node: &mut FlopNode,
    player: usize,
    cfreach: &[f32],
    player_reach: &[f32],
    params: &DiscountParams,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    if num_actions == 1 {
        let child = &mut node.play(0);
        solve_recursive(result, game, child, player, cfreach, player_reach, params);
        return;
    }

    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    if node.player() == player {
        let strategy = regret_matching(node.regrets(), num_actions);

        let pr_size = player_reach.len();
        let mut pr_actions: Vec<f32> = strategy.iter().copied().collect();
        pr_actions.chunks_exact_mut(pr_size).for_each(|row| {
            mul_slice(row, player_reach);
        });

        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game, &mut node.play(action), player, cfreach,
                row(&pr_actions, action, pr_size), params,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        let gamma = params.gamma_t;
        node.strategy_mut().iter_mut().zip(&strategy).for_each(|(x, y)| {
            *x = *x * gamma + *y;
        });

        let (alpha, beta) = (params.alpha_t, params.beta_t);
        let cum_regret = node.regrets_mut();
        cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
            let coef = if x.is_sign_positive() { alpha } else { beta };
            *x = *x * coef + *y;
        });
        cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
            sub_slice(row, result);
        });
    } else {
        let mut cfreach_actions = regret_matching(node.regrets(), num_actions);
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game, &mut node.play(action), player,
                row(&cfreach_actions, action, row_size),
                player_reach, params,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

// ============================================================================
// Solve PostFlopGame with oracle at turn boundary
// ============================================================================

/// Context for solving a PostFlopGame using an oracle at turn chance nodes.
pub struct OracleContext {
    oracle: OracleLookupTable,
    /// flop_to_turn[card][player][flop_idx] = turn_idx (usize::MAX if blocked)
    flop_to_turn: Vec<Option<[Vec<usize>; 2]>>,
    num_turn_cards: usize,
}

impl OracleContext {
    /// Creates an OracleContext from a PostFlopGame and an OracleLookupTable.
    pub fn new(game: &PostFlopGame, oracle: OracleLookupTable) -> Self {
        let flop = [
            game.card_config().flop[0],
            game.card_config().flop[1],
            game.card_config().flop[2],
        ];
        let board_mask: u64 = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);

        let mut flop_to_turn: Vec<Option<[Vec<usize>; 2]>> = Vec::with_capacity(52);
        let mut num_turn_cards = 0usize;
        for card in 0u8..52 {
            if board_mask & (1u64 << card) != 0 {
                flop_to_turn.push(None);
                continue;
            }
            num_turn_cards += 1;
            let mut mappings: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            for player in 0..2 {
                let private_cards = game.private_cards(player);
                let mut turn_idx = 0usize;
                for &(c1, c2) in private_cards {
                    if c1 == card || c2 == card {
                        mappings[player].push(usize::MAX);
                    } else {
                        mappings[player].push(turn_idx);
                        turn_idx += 1;
                    }
                }
            }
            flop_to_turn.push(Some(mappings));
        }

        Self { oracle, flop_to_turn, num_turn_cards }
    }

    /// Evaluate turn boundary using the oracle. Returns CFVs in flop hand indexing.
    pub fn evaluate_turn_boundary(
        &self,
        result: &mut [MaybeUninit<f32>],
        amount: i32,
        player: usize,
        cfreach: &[f32],
    ) {
        // Debug: log inputs
        // eprintln!(
        //     "[oracle-input] player={} amount={} | cfreach={:?}",
        //     player, amount, cfreach
        // );
        result.iter_mut().for_each(|r| { r.write(0.0); });
        let result_f32 = unsafe { &mut *(result as *mut [MaybeUninit<f32>] as *mut [f32]) };

        let opponent = player ^ 1;

        for card in 0u8..52 {
            let mapping = match &self.flop_to_turn[card as usize] {
                Some(m) => m,
                None => continue,
            };

            let matrix = match self.oracle.get(amount, card) {
                Some(m) => m,
                None => continue,
            };

            // Map cfreach from flop indexing to turn indexing
            let num_turn_hands_opp = matrix.num_private_hands(opponent);
            let mut turn_cfreach = vec![0.0f32; num_turn_hands_opp];
            for (flop_idx, &turn_idx) in mapping[opponent].iter().enumerate() {
                if turn_idx != usize::MAX {
                    turn_cfreach[turn_idx] = cfreach[flop_idx];
                }
            }

            // Matrix-vector multiply
            let turn_cfvs = matrix.evaluate(player, &turn_cfreach);

            // Map back to flop indexing
            for (flop_idx, &turn_idx) in mapping[player].iter().enumerate() {
                if turn_idx != usize::MAX {
                    result_f32[flop_idx] += turn_cfvs[turn_idx];
                }
            }
        }

        let scale = 1.0 / self.num_turn_cards as f32;
        for v in result_f32.iter_mut() {
            *v *= scale;
        }

        // Debug: log full CFV array
        // eprintln!(
        //     "[oracle] player={} amount={} | cfv={:?}",
        //     player, amount, result_f32
        // );
    }
}

/// Solves a PostFlopGame using the oracle at turn chance nodes.
///
/// The solver writes strategies directly into PostFlopGame's nodes.
/// Flop strategies are solved via DCFR; turn/river nodes are left untouched.
///
/// Note: `compute_exploitability` measures the full tree (including uniform
/// turn/river), so it cannot be used as a stopping criterion here. Instead,
/// we run a fixed number of iterations with convergence mode kicking in
/// at a configurable point.
pub fn solve_with_oracle(
    game: &mut PostFlopGame,
    oracle_ctx: &OracleContext,
    max_iterations: u32,
    _target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if game.is_solved() {
        panic!("Already solved");
    }

    let mut root = game.root();

    if print_progress {
        print!("iteration: 0 / {max_iterations}");
        io::stdout().flush().unwrap();
    }

    for t in 0..max_iterations {
        // No convergence mode — we can't compute exploitability without
        // the full tree, so just use standard DCFR discounting throughout.
        let params = DiscountParams::new(t, false);

        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive_oracle(
                result.spare_capacity_mut(),
                game, oracle_ctx, &mut root, player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        if print_progress {
            print!("\riteration: {} / {}", t + 1, max_iterations);
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
    }

    finalize(game);
    0.0
}

fn solve_recursive_oracle(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    oracle_ctx: &OracleContext,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // Terminal node: use PostFlopGame's evaluate
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Turn chance node: use oracle instead of recursing into turn/river
    if node.is_chance() && node.turn() == NOT_DEALT {
        oracle_ctx.evaluate_turn_boundary(result, node.amount(), player, cfreach);
        return;
    }

    // Pass-through for single action (non-chance)
    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive_oracle(result, game, oracle_ctx, child, player, cfreach, params);
        return;
    }

    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // Chance node (river — not turn, since turn was handled above)
    if node.is_chance() {
        let chance_factor = game.chance_factor(node);
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / chance_factor as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        for_each_child(node, |action| {
            solve_recursive_oracle(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game, oracle_ctx, &mut node.play(action), player,
                &cfreach_updated, params,
            );
        });

        let mut result_f64 = Vec::with_capacity(num_hands);
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        let isomorphic_chances = game.isomorphic_chances(node);
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);
            apply_swap(tmp, swap_list);
            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });
            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });
    }
    // Current player's node
    else if node.player() == player {
        let strategy = regret_matching(node.regrets(), num_actions);

        for_each_child(node, |action| {
            solve_recursive_oracle(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game, oracle_ctx, &mut node.play(action), player,
                cfreach, params,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        let gamma = params.gamma_t;
        node.strategy_mut().iter_mut().zip(&strategy).for_each(|(x, y)| {
            *x = *x * gamma + *y;
        });

        let (alpha, beta) = (params.alpha_t, params.beta_t);
        let cum_regret = node.regrets_mut();
        cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
            let coef = if x.is_sign_positive() { alpha } else { beta };
            *x = *x * coef + *y;
        });
        cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
            sub_slice(row, result);
        });
    }
    // Opponent's node
    else {
        let mut cfreach_actions = regret_matching(node.regrets(), num_actions);
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        for_each_child(node, |action| {
            solve_recursive_oracle(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game, oracle_ctx, &mut node.play(action), player,
                row(&cfreach_actions, action, row_size), params,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}
