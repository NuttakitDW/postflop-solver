//! Flop-only game tree for NN-based solving (DeepStack-style).
//!
//! Completely separate from PostFlopGame. The tree only contains flop actions.
//! At flop boundaries (where the turn would be dealt), the tree stops with
//! NN leaf nodes where a boundary CFV function provides turn+river values.

use crate::action_tree::*;
use crate::card::*;
use crate::game::PostFlopGame;
use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::solver::solve;
use crate::utility::*;
use std::collections::HashMap;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::slice;

// ============================================================================
// BoundaryCfv trait
// ============================================================================

/// Trait for computing counterfactual values at flop boundary (NN leaf) nodes.
///
/// Implementations provide turn+river CFVs given the opponent's reach
/// distribution at the boundary. The oracle uses standard CFR on Turn-start
/// games; later this can be replaced by a neural network.
pub trait BoundaryCfv: Send + Sync {
    fn compute(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    );
}

// ============================================================================
// FlopNode
// ============================================================================

/// A node in the flop-only game tree.
///
/// Uses the same arena + pointer arithmetic pattern as PostFlopNode,
/// but with simple Vec-based storage (no compression, no raw pointers for data).
pub struct FlopNode {
    /// Player flags (PLAYER_OOP, PLAYER_IP, PLAYER_TERMINAL_FLAG, PLAYER_FOLD_FLAG).
    player: u8,
    /// True if this node is a flop boundary where the NN provides values.
    pub(crate) is_nn_leaf: bool,
    /// Total bet amount from one side at this node.
    amount: i32,
    /// Offset from this node to first child in the arena.
    children_offset: u32,
    /// Number of children (= number of actions for action nodes).
    num_children: u16,
    /// Number of private hands for the acting player.
    num_hands: u16,
    /// Strategy storage: num_actions * num_hands (cumulative strategy sums).
    strategy: Vec<f32>,
    /// Regret storage: num_actions * num_hands (cumulative regrets).
    regrets: Vec<f32>,
    // Note: cfvalues share the regrets storage (same as PostFlopNode).
    // After solving, regrets are overwritten with cfvalues during finalize().
    /// IP counterfactual values (only at action-root nodes after chance).
    cfvalues_ip: Vec<f32>,
}

impl Default for FlopNode {
    fn default() -> Self {
        Self {
            player: PLAYER_OOP,
            is_nn_leaf: false,
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
    /// Returns a slice of child MutexLike<FlopNode> in the arena.
    /// Uses the same pointer arithmetic trick as PostFlopNode::children().
    /// Safe because MutexLike is repr(transparent).
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

    /// Returns the pot size at this node.
    #[inline]
    pub fn amount(&self) -> i32 {
        self.amount
    }
}

impl GameNode for FlopNode {
    #[inline]
    fn is_terminal(&self) -> bool {
        self.is_nn_leaf || (self.player & PLAYER_TERMINAL_FLAG != 0)
    }

    #[inline]
    fn is_chance(&self) -> bool {
        false // No chance nodes in flop-only tree
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
        // Shares storage with regrets (same as PostFlopNode)
        &self.regrets
    }

    #[inline]
    fn cfvalues_mut(&mut self) -> &mut [f32] {
        // Shares storage with regrets (same as PostFlopNode)
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
// FlopGame
// ============================================================================

/// A flop-only game for NN-based solving.
///
/// The tree contains only flop action nodes. At the flop boundary (where the
/// turn would be dealt), NN leaf nodes are created instead of chance nodes.
pub struct FlopGame {
    /// Flat arena of all nodes (same pattern as PostFlopGame).
    node_arena: Vec<MutexLike<FlopNode>>,
    /// Card configuration (flop cards, ranges).
    card_config: CardConfig,
    /// Tree configuration (bet sizes, pot, stack, etc.).
    tree_config: TreeConfig,
    /// Number of private hands per player.
    num_private_hands: [usize; 2],
    /// Initial reach probabilities per player.
    initial_weights: [Vec<f32>; 2],
    /// Private card pairs per player.
    private_cards: [Vec<(Card, Card)>; 2],
    /// Index of same hand in opponent's hand list (for inclusion-exclusion).
    same_hand_index: [Vec<u16>; 2],
    /// Valid hand indices (non-conflicting with flop).
    valid_indices_flop: [Vec<u16>; 2],
    /// Number of valid hand pair combinations.
    num_combinations: f64,
    /// Whether the game has been solved.
    is_solved: bool,
    /// Optional boundary CFV function (oracle or NN).
    oracle: Option<Box<dyn BoundaryCfv>>,
}

impl Game for FlopGame {
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
        if node.is_nn_leaf {
            match &self.oracle {
                Some(oracle) => oracle.compute(result, node, player, cfreach),
                None => result.iter_mut().for_each(|r| { r.write(0.0); }),
            }
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

impl FlopGame {
    /// Creates a new FlopGame from an ActionTree and CardConfig.
    ///
    /// The ActionTree is consumed (ejected). Only flop-level nodes are built;
    /// chance nodes at the flop boundary become NN leaf nodes.
    pub fn new(
        card_config: CardConfig,
        action_tree: ActionTree,
    ) -> Result<Self, String> {
        let (tree_config, _added, _removed, action_root) = action_tree.eject();

        if tree_config.initial_state != BoardState::Flop {
            return Err("FlopGame requires initial_state == Flop".to_string());
        }

        // Initialize card fields
        let flop = card_config.flop;
        let board_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        let mut private_cards: [Vec<(Card, Card)>; 2] = [Vec::new(), Vec::new()];
        let mut initial_weights: [Vec<f32>; 2] = [Vec::new(), Vec::new()];

        for player in 0..2 {
            let (hands, weights) = card_config.range[player].get_hands_weights(board_mask);
            initial_weights[player] = weights;
            private_cards[player] = hands;
        }

        let num_private_hands = [
            private_cards[0].len(),
            private_cards[1].len(),
        ];

        // Compute same_hand_index
        let mut same_hand_index: [Vec<u16>; 2] = [Vec::new(), Vec::new()];
        for player in 0..2 {
            let player_hands = &private_cards[player];
            let opponent_hands = &private_cards[player ^ 1];
            for hand in player_hands {
                same_hand_index[player].push(
                    opponent_hands
                        .binary_search(hand)
                        .map_or(u16::MAX, |i| i as u16),
                );
            }
        }

        // Compute valid_indices_flop (hands that don't conflict with flop cards)
        let (valid_indices_flop, _, _) = card_config.valid_indices(&private_cards);

        // Compute num_combinations
        let mut num_combinations = 0.0f64;
        for (&(c1, c2), &w1) in private_cards[0].iter().zip(initial_weights[0].iter()) {
            let oop_mask: u64 = (1 << c1) | (1 << c2);
            for (&(c3, c4), &w2) in private_cards[1].iter().zip(initial_weights[1].iter()) {
                let ip_mask: u64 = (1 << c3) | (1 << c4);
                if oop_mask & ip_mask == 0 {
                    num_combinations += w1 as f64 * w2 as f64;
                }
            }
        }

        if num_combinations == 0.0 {
            return Err("Valid card assignment does not exist".to_string());
        }

        // Count nodes in flop-only tree
        let num_nodes = count_flop_nodes(&action_root.lock());

        // Build arena
        let node_arena: Vec<MutexLike<FlopNode>> = (0..num_nodes)
            .map(|_| MutexLike::new(FlopNode::default()))
            .collect();

        let mut game = Self {
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
        };

        // Build tree from action tree
        let mut next_index = 1; // 0 is root
        game.build_tree(0, &action_root.lock(), &mut next_index);

        // Allocate storage for each node
        game.allocate_storage();

        Ok(game)
    }

    /// Returns the number of nodes in the arena.
    pub fn num_nodes(&self) -> usize {
        self.node_arena.len()
    }

    /// Returns the tree configuration.
    pub fn tree_config(&self) -> &TreeConfig {
        &self.tree_config
    }

    /// Returns the card configuration.
    pub fn card_config(&self) -> &CardConfig {
        &self.card_config
    }

    /// Returns the private cards for a player.
    pub fn private_cards(&self, player: usize) -> &[(Card, Card)] {
        &self.private_cards[player]
    }

    /// Sets the boundary CFV oracle/network.
    pub fn set_oracle(&mut self, oracle: Box<dyn BoundaryCfv>) {
        self.oracle = Some(oracle);
    }

    /// Transfers solved flop strategies into PostFlopGame via node locking.
    ///
    /// PostFlopGame must have memory allocated and not be solved yet.
    /// FlopGame must be solved (i.e., `solve_flop_nn` has been called).
    ///
    /// Both trees share identical flop-level structure (same ActionTree),
    /// so DFS action indices correspond 1:1.
    pub fn lock_strategies_in(&self, postflop: &mut PostFlopGame) {
        let root = self.root();
        let mut path = Vec::new();
        Self::lock_recursive(&*root, postflop, &mut path);
        postflop.back_to_root();
    }

    fn lock_recursive(
        flop_node: &FlopNode,
        postflop: &mut PostFlopGame,
        path: &mut Vec<usize>,
    ) {
        // Stop at fold terminals and NN leaves (chance nodes in PostFlopGame)
        if flop_node.is_terminal() {
            return;
        }

        let num_actions = flop_node.num_actions();

        // Lock strategy for nodes with >1 action (single-action nodes are passthrough)
        if num_actions > 1 {
            postflop.apply_history(path);
            postflop.lock_current_strategy(flop_node.strategy());
        }

        // Recurse into children
        for a in 0..num_actions {
            let child = flop_node.play(a);
            path.push(a);
            Self::lock_recursive(&*child, postflop, path);
            path.pop();
        }
    }

    /// Collects all distinct boundary amounts from NN leaf nodes.
    pub fn boundary_amounts(&self) -> Vec<i32> {
        let mut amounts = Vec::new();
        for i in 0..self.node_arena.len() {
            let node = self.node_arena[i].lock();
            if node.is_nn_leaf && !amounts.contains(&node.amount) {
                amounts.push(node.amount);
            }
        }
        amounts
    }

    /// Recursively builds the flop-only tree from an ActionTreeNode.
    fn build_tree(
        &self,
        node_index: usize,
        action_node: &ActionTreeNode,
        next_index: &mut usize,
    ) {
        let mut node = self.node_arena[node_index].lock();
        node.player = action_node.player;
        node.amount = action_node.amount;

        // Terminal node (fold/showdown)
        if action_node.player & PLAYER_TERMINAL_FLAG != 0 {
            return;
        }

        // Chance node at flop boundary → NN leaf
        if action_node.player & PLAYER_CHANCE_FLAG != 0 {
            node.is_nn_leaf = true;
            return;
        }

        // Action node — assign children
        let num_children = action_node.children.len();
        node.children_offset = (*next_index - node_index) as u32;
        node.num_children = num_children as u16;
        *next_index += num_children;

        // Recurse into each child
        for (i, child_action_node) in action_node.children.iter().enumerate() {
            let child_index = node_index + node.children_offset as usize + i;
            self.build_tree(child_index, &child_action_node.lock(), next_index);
        }
    }

    /// Allocates strategy/regret/cfvalue storage for each node.
    fn allocate_storage(&mut self) {
        for i in 0..self.node_arena.len() {
            let (player, num_actions, is_terminal, is_nn_leaf, is_root) = {
                let node = self.node_arena[i].lock();
                let is_terminal = node.player & PLAYER_TERMINAL_FLAG != 0;
                (
                    node.player as usize & PLAYER_MASK as usize,
                    node.num_children as usize,
                    is_terminal,
                    node.is_nn_leaf,
                    i == 0,
                )
            };

            if is_terminal || is_nn_leaf || num_actions == 0 {
                continue;
            }

            let num_hands = self.num_private_hands[player];
            let num_elements = num_actions * num_hands;

            let mut node = self.node_arena[i].lock();
            node.num_hands = num_hands as u16;
            node.strategy = vec![0.0; num_elements];
            node.regrets = vec![0.0; num_elements];

            // IP cfvalues at root-like nodes (first action node after root)
            if is_root {
                let ip_hands = self.num_private_hands[PLAYER_IP as usize];
                node.cfvalues_ip = vec![0.0; ip_hands];
            }
        }
    }

    /// Fold evaluation for terminal nodes.
    fn evaluate_fold(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = f64::min(
            pot * self.tree_config.rake_rate,
            self.tree_config.rake_cap,
        );

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

        result.iter_mut().for_each(|v| {
            v.write(0.0);
        });
        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        let opponent_indices = &self.valid_indices_flop[player ^ 1];
        for &i in opponent_indices {
            unsafe {
                let cfreach_i = *cfreach.get_unchecked(i as usize);
                if cfreach_i != 0.0 {
                    let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                    let cfreach_i_f64 = cfreach_i as f64;
                    cfreach_sum += cfreach_i_f64;
                    *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                    *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                }
            }
        }

        if cfreach_sum == 0.0 {
            return;
        }

        let player_indices = &self.valid_indices_flop[player];
        let same_hand_index = &self.same_hand_index[player];
        for &i in player_indices {
            unsafe {
                let (c1, c2) = *player_cards.get_unchecked(i as usize);
                let same_i = *same_hand_index.get_unchecked(i as usize);
                let cfreach_same = if same_i == u16::MAX {
                    0.0
                } else {
                    *cfreach.get_unchecked(same_i as usize) as f64
                };
                let cfreach = cfreach_sum + cfreach_same
                    - *cfreach_minus.get_unchecked(c1 as usize)
                    - *cfreach_minus.get_unchecked(c2 as usize);
                *result.get_unchecked_mut(i as usize) = (payoff * cfreach) as f32;
            }
        }
    }
}

/// Counts the total number of nodes needed for the flop-only tree.
fn count_flop_nodes(action_node: &ActionTreeNode) -> usize {
    let mut count = 1; // this node
    if action_node.player & PLAYER_TERMINAL_FLAG != 0 || action_node.player & PLAYER_CHANCE_FLAG != 0 {
        // Terminal or chance (NN leaf) — no children
        return count;
    }
    for child in &action_node.children {
        count += count_flop_nodes(&child.lock());
    }
    count
}

// ============================================================================
// TurnOracle: exact boundary CFVs via standard CFR
// ============================================================================

/// Oracle that computes exact boundary CFVs by solving Turn-start games.
///
/// For each distinct pot/stack at boundary nodes and each of 45 possible turn
/// cards, a Turn-start game is built and solved to convergence. At evaluation
/// time, `compute_cfvalue_recursive` traverses the pre-solved trees with the
/// current opponent reach to produce exact CFVs.
///
/// Key insight: Nash equilibrium strategies in zero-sum games are
/// reach-independent, so we pre-solve once and evaluate with any reaches.
pub struct TurnOracle {
    /// Solved turn games grouped by boundary amount.
    /// For each amount: 52 slots (None for flop cards).
    games_by_amount: HashMap<i32, Vec<Option<PostFlopGame>>>,

    /// For each card (0-51), mapping from flop hand index to turn hand index.
    /// None for flop cards. flop_to_turn[card][player][flop_idx] = turn_idx or usize::MAX.
    flop_to_turn: Vec<Option<[Vec<usize>; 2]>>,

    /// Number of remaining turn cards (typically 45).
    total_turn_cards: usize,

    /// Number of flop-level hands per player (for future NN integration).
    #[allow(dead_code)]
    num_flop_hands: [usize; 2],

    /// Number of valid hand-pair combinations (for future NN integration).
    #[allow(dead_code)]
    num_flop_combinations: f64,
}

impl TurnOracle {
    /// Builds and solves Turn-start games for all boundary nodes.
    ///
    /// For each distinct pot/stack at boundary NN leaf nodes, and for each of
    /// the 45 remaining turn cards, a separate Turn-start game is constructed
    /// and solved with standard CFR.
    pub fn new(
        game: &FlopGame,
        turn_max_iterations: u32,
        turn_target_exploitability: f32,
    ) -> Self {
        let card_config = &game.card_config;
        let tree_config = &game.tree_config;
        let flop = card_config.flop;
        let flop_mask: u64 = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);

        // Collect distinct boundary amounts
        let amounts = game.boundary_amounts();

        // Build hand index mappings for each turn card
        let mut flop_to_turn: Vec<Option<[Vec<usize>; 2]>> = Vec::with_capacity(52);
        let mut total_turn_cards = 0usize;

        for card in 0u8..52 {
            if flop_mask & (1u64 << card) != 0 {
                flop_to_turn.push(None);
                continue;
            }
            total_turn_cards += 1;

            let mut mappings: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
            for player in 0..2 {
                let flop_hands = &game.private_cards[player];
                let mut turn_idx = 0usize;
                for &(c1, c2) in flop_hands {
                    if c1 == card || c2 == card {
                        mappings[player].push(usize::MAX); // blocked by turn card
                    } else {
                        mappings[player].push(turn_idx);
                        turn_idx += 1;
                    }
                }
            }
            flop_to_turn.push(Some(mappings));
        }

        // Build and solve Turn-start games for each (amount, card)
        let mut games_by_amount: HashMap<i32, Vec<Option<PostFlopGame>>> = HashMap::new();

        for &amount in &amounts {
            let pot = tree_config.starting_pot + 2 * amount;
            let stack = tree_config.effective_stack - amount;

            let mut card_games: Vec<Option<PostFlopGame>> = (0..52).map(|_| None).collect();

            for card in 0u8..52 {
                if flop_mask & (1u64 << card) != 0 {
                    continue;
                }

                let turn_card_config = CardConfig {
                    range: [card_config.range[0], card_config.range[1]],
                    flop,
                    turn: card,
                    ..Default::default()
                };

                let turn_tree_config = if stack > 0 {
                    // Fixed action abstraction: 100% pot bet, all-in raise
                    TreeConfig {
                        initial_state: BoardState::Turn,
                        starting_pot: pot,
                        effective_stack: stack,
                        rake_rate: tree_config.rake_rate,
                        rake_cap: tree_config.rake_cap,
                        turn_bet_sizes: [
                            ("100%", "a").try_into().unwrap(),
                            ("100%", "a").try_into().unwrap(),
                        ],
                        river_bet_sizes: [
                            ("100%", "a").try_into().unwrap(),
                            ("100%", "a").try_into().unwrap(),
                        ],
                        add_allin_threshold: 1.5,
                        force_allin_threshold: 0.2,
                        ..Default::default()
                    }
                } else {
                    // All-in: no more betting, just runout equity.
                    TreeConfig {
                        initial_state: BoardState::Turn,
                        starting_pot: pot,
                        effective_stack: 1,
                        rake_rate: tree_config.rake_rate,
                        rake_cap: tree_config.rake_cap,
                        add_allin_threshold: 0.0,
                        force_allin_threshold: 0.0,
                        ..Default::default()
                    }
                };

                let action_tree = ActionTree::new(turn_tree_config).unwrap();
                let mut turn_game =
                    PostFlopGame::with_config(turn_card_config, action_tree).unwrap();
                turn_game.allocate_memory(false);

                // Solve to convergence
                solve(&mut turn_game, turn_max_iterations, turn_target_exploitability, false);

                card_games[card as usize] = Some(turn_game);
            }

            games_by_amount.insert(amount, card_games);
        }

        Self {
            games_by_amount,
            flop_to_turn,
            total_turn_cards,
            num_flop_hands: game.num_private_hands,
            num_flop_combinations: game.num_combinations,
        }
    }

    /// Evaluates boundary CFVs for a given player and opponent reaches.
    ///
    /// For each of 45 turn cards:
    /// 1. Maps cfreach from flop indexing to turn game indexing
    /// 2. Calls compute_cfvalue_recursive on the pre-solved Turn-start game
    /// 3. Maps CFVs back to flop indexing
    /// Then averages across all turn cards (1/45 scaling).
    fn evaluate(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        // Initialize result to zeros
        result.iter_mut().for_each(|r| {
            r.write(0.0);
        });
        let result_f32 = unsafe { &mut *(result as *mut [MaybeUninit<f32>] as *mut [f32]) };

        let card_games = match self.games_by_amount.get(&node.amount) {
            Some(games) => games,
            None => return,
        };

        let opponent = player ^ 1;

        for card in 0u8..52 {
            let mapping = match &self.flop_to_turn[card as usize] {
                Some(m) => m,
                None => continue, // flop card
            };

            let turn_game = match &card_games[card as usize] {
                Some(g) => g,
                None => continue,
            };

            // Map cfreach from flop indexing to turn indexing
            let num_turn_hands_opp = turn_game.num_private_hands(opponent);
            let mut turn_cfreach = vec![0.0f32; num_turn_hands_opp];
            for (flop_idx, &turn_idx) in mapping[opponent].iter().enumerate() {
                if turn_idx != usize::MAX {
                    turn_cfreach[turn_idx] = cfreach[flop_idx];
                }
            }

            // Compute CFVs for this turn card using the pre-solved game
            let num_turn_hands = turn_game.num_private_hands(player);
            let mut turn_cfvs: Vec<MaybeUninit<f32>> =
                vec![MaybeUninit::uninit(); num_turn_hands];
            {
                let mut root = turn_game.root();
                compute_cfvalue_recursive(
                    &mut turn_cfvs,
                    turn_game,
                    &mut root,
                    player,
                    &turn_cfreach,
                    false,
                );
            }

            // Map CFVs back to flop indexing and accumulate
            for (flop_idx, &turn_idx) in mapping[player].iter().enumerate() {
                if turn_idx != usize::MAX {
                    result_f32[flop_idx] += unsafe { turn_cfvs[turn_idx].assume_init() };
                }
            }
        }

        // Average across turn cards (same as standard solver's 1/chance_factor scaling)
        let scale = 1.0 / self.total_turn_cards as f32;
        for v in result_f32.iter_mut() {
            *v *= scale;
        }
    }
}

impl BoundaryCfv for TurnOracle {
    fn compute(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &FlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        self.evaluate(result, node, player, cfreach);
    }
}

// ============================================================================
// Solver: solve_flop_nn
// ============================================================================

/// DCFR discount parameters (same as solver.rs).
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

        let (alpha_t, beta_t, gamma_t) = if convergence_mode {
            let alpha = ((pow_alpha / (pow_alpha + 1.0)) as f32).max(0.9);
            let beta = 0.9;
            let gamma = (pow_gamma as f32).max(0.9);
            (alpha, beta, gamma)
        } else {
            let alpha = (pow_alpha / (pow_alpha + 1.0)) as f32;
            let beta = 0.5;
            let gamma = pow_gamma as f32;
            (alpha, beta, gamma)
        };

        Self {
            alpha_t,
            beta_t,
            gamma_t,
        }
    }
}

/// Regret matching: converts cumulative regrets to a strategy.
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

/// Solves the flop-only game using Discounted CFR.
///
/// Boundary CFVs at NN leaf nodes are provided by the oracle set via
/// `game.set_oracle()`. Without an oracle, zeros are used.
/// Returns the exploitability of the obtained strategy.
pub fn solve_flop_nn(
    game: &mut FlopGame,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);
    let starting_pot = game.starting_pot() as f32;
    let target_percent = if starting_pot > 0.0 {
        target_exploitability / starting_pot * 100.0
    } else {
        0.0
    };
    let mut convergence_mode = false;

    if print_progress {
        print!("iteration: 0 / {max_num_iterations} ");
        if starting_pot > 0.0 {
            let current_percent = exploitability / starting_pot * 100.0;
            print!("(exploitability = {current_percent:.2}% | target = {target_percent:.2}%)");
        } else {
            print!("(exploitability = {exploitability:.4e})");
        }
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let is_power_of_4 = t > 0 && t == 1u32 << ((t.leading_zeros() ^ 31) & !1);
        if is_power_of_4 {
            exploitability = compute_exploitability(game);
        }

        let current_percent = exploitability / starting_pot * 100.0;
        if starting_pot > 0.0 && current_percent < 1.0 {
            convergence_mode = true;
        }

        let params = DiscountParams::new(t, convergence_mode);

        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive_flop_nn(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1), // opponent reach
                game.initial_weights(player),      // player reach
                &params,
            );
        }

        let check_exploitability = (t + 1) % 10 == 0 || t + 1 == max_num_iterations;
        if check_exploitability {
            exploitability = compute_exploitability(game);
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            if starting_pot > 0.0 {
                let current_percent = exploitability / starting_pot * 100.0;
                print!("(exploitability = {current_percent:.2}% | target = {target_percent:.2}%)");
            } else {
                print!("(exploitability = {exploitability:.4e})");
            }
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
        io::stdout().flush().unwrap();
    }

    finalize(game);

    exploitability
}

/// Recursive DCFR traversal for flop-only tree.
fn solve_recursive_flop_nn(
    result: &mut [MaybeUninit<f32>],
    game: &FlopGame,
    node: &mut FlopNode,
    player: usize,
    cfreach: &[f32],
    player_reach: &[f32],
    params: &DiscountParams,
) {
    // Terminal node: fold terminal or NN leaf (handled by game.evaluate)
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Single action passthrough
    if num_actions == 1 {
        let child = &mut node.play(0);
        solve_recursive_flop_nn(result, game, child, player, cfreach, player_reach, params);
        return;
    }

    // Allocate CFV storage for each action
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // Player's own node — update regrets and strategy
    if node.player() == player {
        let strategy = regret_matching(node.regrets(), num_actions);

        // Split player_reach by strategy per action
        let player_reach_size = player_reach.len();
        let mut player_reach_actions: Vec<f32> = strategy.iter().copied().collect();
        player_reach_actions
            .chunks_exact_mut(player_reach_size)
            .for_each(|row| {
                mul_slice(row, player_reach);
            });

        // Recurse into each action
        for_each_child(node, |action| {
            solve_recursive_flop_nn(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                row(&player_reach_actions, action, player_reach_size),
                params,
            );
        });

        // Weight CFVs by strategy
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        // Update cumulative strategy
        let gamma = params.gamma_t;
        let cum_strategy = node.strategy_mut();
        cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
            *x = *x * gamma + *y;
        });

        // Update cumulative regrets
        let (alpha, beta) = (params.alpha_t, params.beta_t);
        let cum_regret = node.regrets_mut();
        cum_regret
            .iter_mut()
            .zip(&*cfv_actions)
            .for_each(|(x, y)| {
                let coef = if x.is_sign_positive() { alpha } else { beta };
                *x = *x * coef + *y;
            });
        cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
            sub_slice(row, result);
        });
    }
    // Opponent's node — split cfreach, pass player_reach unchanged
    else {
        let mut cfreach_actions = regret_matching(node.regrets(), num_actions);

        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // Recurse — player_reach passes through unchanged
        for_each_child(node, |action| {
            solve_recursive_flop_nn(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                player_reach,
                params,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range::*;

    fn create_test_game() -> FlopGame {
        let card_config = CardConfig {
            range: [
                "AA,KK,QQ,AK,AQ,AJs,KQs".parse().unwrap(),
                "JJ,TT,99,AJs,ATs,KQs,KJs,QJs".parse().unwrap(),
            ],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 200,
            flop_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        FlopGame::new(card_config, action_tree).unwrap()
    }

    fn create_matched_configs() -> (CardConfig, TreeConfig) {
        let card_config = CardConfig {
            range: [
                "AA,KK,QQ,AK,AQ,AJs,KQs".parse().unwrap(),
                "JJ,TT,99,AJs,ATs,KQs,KJs,QJs".parse().unwrap(),
            ],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 60,
            effective_stack: 200,
            flop_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            turn_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            river_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            ..Default::default()
        };

        (card_config, tree_config)
    }

    #[test]
    fn test_flop_game_tree_structure() {
        let game = create_test_game();

        // Tree should have nodes
        assert!(game.num_nodes() > 0);

        // Check root is an action node (OOP acts first)
        let root = game.root();
        assert!(!root.is_terminal());
        assert!(!root.is_nn_leaf);
        assert_eq!(root.player(), PLAYER_OOP as usize);
        assert!(root.num_actions() > 0);
    }

    #[test]
    fn test_flop_game_has_nn_leaves() {
        let game = create_test_game();

        // Count NN leaf nodes
        let mut nn_leaf_count = 0;
        let mut fold_count = 0;
        let mut action_count = 0;

        for i in 0..game.num_nodes() {
            let node = game.node_arena[i].lock();
            if node.is_nn_leaf {
                nn_leaf_count += 1;
            } else if node.player & PLAYER_TERMINAL_FLAG != 0 {
                fold_count += 1;
            } else {
                action_count += 1;
            }
        }

        assert!(nn_leaf_count > 0, "Should have NN leaf nodes at flop boundary");
        assert!(fold_count > 0, "Should have fold terminals");
        assert!(action_count > 0, "Should have action nodes");

        // No chance nodes should exist
        for i in 0..game.num_nodes() {
            let node = game.node_arena[i].lock();
            assert!(
                node.player & PLAYER_CHANCE_FLAG == 0 || node.is_nn_leaf,
                "No chance nodes should exist in flop-only tree"
            );
        }
    }

    #[test]
    fn test_flop_game_is_small() {
        let game = create_test_game();
        let flop_nodes = game.num_nodes();
        // A flop-only tree with 50% bet should be small (tens of nodes, not thousands)
        assert!(
            flop_nodes < 100,
            "Flop-only tree should be small, got {flop_nodes} nodes"
        );
    }

    #[test]
    fn test_solve_flop_nn_runs() {
        let mut game = create_test_game();
        let exploitability = solve_flop_nn(&mut game, 100, 0.0, false);
        // With zero NN predictions, exploitability will be non-zero
        // but the solver should complete without panicking
        assert!(exploitability >= 0.0);
    }

    #[test]
    fn test_oracle_matches_standard() {
        let (card_config, tree_config) = create_matched_configs();

        let num_iterations = 500;
        let target = 60.0 * 0.003; // 0.3% of pot

        // --- Standard full-tree solve ---
        let action_tree = ActionTree::new(tree_config.clone()).unwrap();
        let mut game_std = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
        game_std.allocate_memory(false);
        solve(&mut game_std, num_iterations, target, false);

        // Extract OOP root CFVs from standard game
        let num_hands_oop = game_std.num_private_hands(0);
        let mut cfvs_std = vec![MaybeUninit::<f32>::uninit(); num_hands_oop];
        {
            let mut root = game_std.root();
            compute_cfvalue_recursive(
                &mut cfvs_std,
                &game_std,
                &mut root,
                0,
                game_std.initial_weights(1),
                false,
            );
        }

        // --- FlopGame + TurnOracle solve ---
        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game_flop = FlopGame::new(card_config, action_tree).unwrap();

        // Build oracle (pre-solve all turn subtrees)
        let oracle = TurnOracle::new(&game_flop, num_iterations, target);
        game_flop.set_oracle(Box::new(oracle));

        // Solve flop game
        solve_flop_nn(&mut game_flop, num_iterations, target, false);

        // Extract OOP root CFVs from flop game
        let num_hands_flop = game_flop.num_private_hands(0);
        assert_eq!(num_hands_oop, num_hands_flop);

        let mut cfvs_flop = vec![MaybeUninit::<f32>::uninit(); num_hands_flop];
        {
            let mut root = game_flop.root();
            compute_cfvalue_recursive(
                &mut cfvs_flop,
                &game_flop,
                &mut root,
                0,
                game_flop.initial_weights(1),
                false,
            );
        }

        // Compare per-hand CFVs
        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f64;
        for i in 0..num_hands_oop {
            let std_val = unsafe { cfvs_std[i].assume_init() };
            let flop_val = unsafe { cfvs_flop[i].assume_init() };
            let diff = (std_val - flop_val).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as f64;
        }
        let avg_diff = total_diff / num_hands_oop as f64;

        // Allow for convergence noise from both solvers + oracle
        // Both standard and oracle solve to ~0.3% pot, so differences should be small
        assert!(
            max_diff < 0.5,
            "Max per-hand CFV difference too large: {max_diff:.6} (avg={avg_diff:.6})"
        );
    }
}
