//! Outcome sampling traversal for Deep PDCFR+.

use super::buffer::{AdvantageBuffer, Sample, StrategyBuffer};
use super::encoder::{ActionType, InfoSetEncoder};
use super::networks::Networks;
use crate::{Action, Game, GameNode, PostFlopGame, PostFlopNode};
use candle_core::Result as CandleResult;
use rand::Rng;

/// Context for a single traversal.
pub struct TraversalContext<'a> {
    /// The game being traversed
    pub game: &'a PostFlopGame,
    /// Neural networks for strategy lookup
    pub networks: &'a Networks,
    /// Information set encoder
    pub encoder: &'a InfoSetEncoder,
    /// Current iteration number
    pub iteration: usize,
    /// Sampled hole cards for OOP player
    pub oop_hand: (u8, u8),
    /// Sampled hole cards for IP player
    pub ip_hand: (u8, u8),
    /// Action history for encoding
    pub action_history: Vec<(ActionType, i32)>,
}

/// Result of a traversal at a node.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Value of the traversal for the traversing player
    pub value: f32,
    /// Samples collected during traversal
    pub advantage_samples: Vec<Sample>,
    /// Strategy samples for average strategy network
    pub strategy_samples: Vec<Sample>,
}

impl TraversalResult {
    pub fn new(value: f32) -> Self {
        Self {
            value,
            advantage_samples: Vec::new(),
            strategy_samples: Vec::new(),
        }
    }

    pub fn merge(&mut self, other: TraversalResult) {
        self.advantage_samples.extend(other.advantage_samples);
        self.strategy_samples.extend(other.strategy_samples);
    }
}

/// Perform outcome sampling traversal.
///
/// Returns the counterfactual value for the traversing player and collected samples.
pub fn outcome_sampling_traverse(
    ctx: &mut TraversalContext,
    node: &PostFlopNode,
    traverser: usize,
    reach_probs: [f32; 2],
    sample_prob: f32,
) -> CandleResult<TraversalResult> {
    // Terminal node
    if node.is_terminal() {
        let value = evaluate_terminal(ctx, node, traverser);
        return Ok(TraversalResult::new(value));
    }

    // Chance node (turn or river card)
    if node.is_chance() {
        return traverse_chance(ctx, node, traverser, reach_probs, sample_prob);
    }

    // Action node
    let current_player = node.player();
    let _num_actions = node.num_actions();

    // Get current strategy from network
    let features = encode_node(ctx, node, current_player);
    let features_tensor = ctx.networks.encode_batch(&[features.clone()])?;
    let strategy = ctx.networks.get_strategy(&features_tensor)?;
    let strategy_vec: Vec<f32> = strategy.flatten_all()?.to_vec1()?;

    // Sample an action proportional to strategy
    let sampled_action = sample_action(&strategy_vec);
    let action_prob = strategy_vec[sampled_action];

    // Get action info for encoding
    let action_info = get_action_info(ctx.game, node, sampled_action);

    // Save action to history
    ctx.action_history.push(action_info);

    // Update reach probabilities
    let mut new_reach = reach_probs;
    new_reach[current_player] *= action_prob;

    // Recurse
    let child = node.play(sampled_action);
    let new_sample_prob = sample_prob * action_prob;
    let mut result = outcome_sampling_traverse(ctx, &child, traverser, new_reach, new_sample_prob)?;

    // Pop action from history
    ctx.action_history.pop();

    // If this is the traverser's node, compute advantages
    if current_player == traverser {
        // Compute counterfactual values for all actions
        let cfvalues = compute_action_cfvalues(ctx, node, traverser, reach_probs, &strategy_vec)?;

        // Compute expected value
        let ev: f32 = cfvalues
            .iter()
            .zip(strategy_vec.iter())
            .map(|(v, p)| v * p)
            .sum();

        // Compute instantaneous advantages: r(I,a) = v(I,a) - v(I)
        let advantages: Vec<f32> = cfvalues.iter().map(|v| v - ev).collect();

        // Weight for importance sampling
        let weight = reach_probs[1 - traverser] / sample_prob;

        // Create advantage sample
        let sample = Sample::new(
            features.clone(),
            advantages,
            ctx.iteration,
            traverser,
            weight,
        );
        result.advantage_samples.push(sample);

        // Create strategy sample
        let strategy_sample = Sample::new(
            features,
            strategy_vec.clone(),
            ctx.iteration,
            traverser,
            ctx.iteration as f32, // Linear weighting
        );
        result.strategy_samples.push(strategy_sample);

        result.value = cfvalues[sampled_action] / action_prob; // Importance sampling
    } else {
        // Opponent node - value already updated from child traversal
    }

    Ok(result)
}

/// Traverse a chance node by sampling one outcome.
fn traverse_chance(
    ctx: &mut TraversalContext,
    node: &PostFlopNode,
    traverser: usize,
    reach_probs: [f32; 2],
    sample_prob: f32,
) -> CandleResult<TraversalResult> {
    let num_chances = node.num_actions();

    // Sample uniformly
    let mut rng = rand::thread_rng();
    let sampled_card = rng.gen_range(0..num_chances);

    let child = node.play(sampled_card);
    let new_sample_prob = sample_prob / num_chances as f32;

    let mut result =
        outcome_sampling_traverse(ctx, &child, traverser, reach_probs, new_sample_prob)?;

    // Importance sampling correction
    result.value *= num_chances as f32;

    Ok(result)
}

/// Evaluate terminal node payoff.
fn evaluate_terminal(ctx: &TraversalContext, node: &PostFlopNode, _player: usize) -> f32 {
    // Use the game's evaluation method
    // For simplicity, we use a basic hand comparison here
    // In practice, this should use ctx.game.evaluate()

    // The pot size is stored in node.amount
    // We need to determine the winner based on hand strength

    // Get hand indices
    let _oop_cards = ctx.oop_hand;
    let _ip_cards = ctx.ip_hand;

    // Use game's private_cards to find indices
    // Then use hand_strength to determine winner

    // For now, return a simplified value based on the pot
    // This will be replaced with proper evaluation
    let _pot = node.bet_amount() as f32;

    // Placeholder: would need to compare hands properly
    // Return pot/2 as a neutral value for now
    0.0
}

/// Compute counterfactual values for all actions at a node.
fn compute_action_cfvalues(
    ctx: &mut TraversalContext,
    node: &PostFlopNode,
    traverser: usize,
    reach_probs: [f32; 2],
    strategy: &[f32],
) -> CandleResult<Vec<f32>> {
    let num_actions = node.num_actions();
    let mut cfvalues = Vec::with_capacity(num_actions);

    for action in 0..num_actions {
        // Get action info
        let action_info = get_action_info(ctx.game, node, action);
        ctx.action_history.push(action_info);

        // Update reach
        let action_prob = strategy[action];
        let mut new_reach = reach_probs;
        new_reach[traverser] *= action_prob;

        let child = node.play(action);

        // Recursive evaluation
        let result = outcome_sampling_traverse(ctx, &child, traverser, new_reach, action_prob)?;

        ctx.action_history.pop();

        cfvalues.push(result.value);
    }

    Ok(cfvalues)
}

/// Encode current node state to feature vector.
fn encode_node(ctx: &TraversalContext, node: &PostFlopNode, player: usize) -> Vec<f32> {
    let flop = ctx.game.card_config().flop;
    let turn = if node.turn_card() == crate::NOT_DEALT {
        255
    } else {
        node.turn_card()
    };
    let river = if node.river_card() == crate::NOT_DEALT {
        255
    } else {
        node.river_card()
    };

    let hole_cards = if player == 0 {
        ctx.oop_hand
    } else {
        ctx.ip_hand
    };

    // Determine street
    let street = if node.turn_card() == crate::NOT_DEALT {
        0 // flop
    } else if node.river_card() == crate::NOT_DEALT {
        1 // turn
    } else {
        2 // river
    };

    // Get pot and stack info from game config
    let starting_pot = ctx.game.tree_config().starting_pot;
    let effective_stack = ctx.game.tree_config().effective_stack;
    let current_pot = starting_pot + 2 * (effective_stack - node.bet_amount());

    ctx.encoder.encode(
        flop,
        turn,
        river,
        hole_cards,
        current_pot,
        node.bet_amount(),
        street,
        &ctx.action_history,
    )
}

/// Sample an action proportional to strategy.
fn sample_action(strategy: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &prob) in strategy.iter().enumerate() {
        cumsum += prob;
        if r < cumsum {
            return i;
        }
    }

    // Fallback to last action (handles floating point issues)
    strategy.len() - 1
}

/// Get action info (type and amount) for encoding.
fn get_action_info(_game: &PostFlopGame, node: &PostFlopNode, action_idx: usize) -> (ActionType, i32) {
    // Get the child node to determine the action
    let child = node.play(action_idx);
    let prev_action = child.previous_action();

    match prev_action {
        Action::Fold => (ActionType::Fold, 0),
        Action::Check => (ActionType::Check, 0),
        Action::Call => (ActionType::Call, child.bet_amount()),
        Action::Bet(amount) => (ActionType::Bet, amount),
        Action::Raise(amount) => (ActionType::Raise, amount),
        Action::AllIn(amount) => (ActionType::Raise, amount),
        Action::Chance(_) | Action::None => (ActionType::Check, 0), // Shouldn't happen
    }
}

/// Run multiple traversals to collect samples.
pub fn collect_samples(
    game: &PostFlopGame,
    networks: &Networks,
    encoder: &InfoSetEncoder,
    iteration: usize,
    num_traversals: usize,
) -> CandleResult<(AdvantageBuffer, StrategyBuffer)> {
    let mut advantage_buffer = AdvantageBuffer::new(num_traversals * 100);
    let mut strategy_buffer = StrategyBuffer::new(num_traversals * 100);

    let private_cards = [game.private_cards(0), game.private_cards(1)];
    let initial_weights = [game.initial_weights(0), game.initial_weights(1)];

    let mut rng = rand::thread_rng();

    for _ in 0..num_traversals {
        // Sample hands for both players
        let oop_idx = sample_weighted(&initial_weights[0], &mut rng);
        let ip_idx = sample_weighted(&initial_weights[1], &mut rng);

        let oop_hand = private_cards[0][oop_idx];
        let ip_hand = private_cards[1][ip_idx];

        // Skip if hands conflict (same card in both)
        if hands_conflict(oop_hand, ip_hand) {
            continue;
        }

        // Traverse for both players
        for traverser in 0..2 {
            let mut ctx = TraversalContext {
                game,
                networks,
                encoder,
                iteration,
                oop_hand: (oop_hand.0, oop_hand.1),
                ip_hand: (ip_hand.0, ip_hand.1),
                action_history: Vec::new(),
            };

            let root = game.root();
            let result = outcome_sampling_traverse(
                &mut ctx,
                &root,
                traverser,
                [1.0, 1.0],
                1.0,
            )?;

            // Add samples to buffers
            for sample in result.advantage_samples {
                advantage_buffer.add(
                    sample.features,
                    sample.targets,
                    sample.iteration,
                    sample.player,
                    sample.weight,
                );
            }

            for sample in result.strategy_samples {
                strategy_buffer.add(
                    sample.features,
                    sample.targets,
                    sample.iteration,
                    sample.player,
                );
            }
        }
    }

    Ok((advantage_buffer, strategy_buffer))
}

/// Sample index weighted by probabilities.
fn sample_weighted(weights: &[f32], rng: &mut impl Rng) -> usize {
    let sum: f32 = weights.iter().sum();
    let r = rng.gen::<f32>() * sum;
    let mut cumsum = 0.0;

    for (i, &w) in weights.iter().enumerate() {
        cumsum += w;
        if r < cumsum {
            return i;
        }
    }

    weights.len() - 1
}

/// Check if two hands share a card.
fn hands_conflict(hand1: (u8, u8), hand2: (u8, u8)) -> bool {
    hand1.0 == hand2.0 || hand1.0 == hand2.1 || hand1.1 == hand2.0 || hand1.1 == hand2.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_action() {
        // Deterministic strategy
        let strategy = vec![0.0, 1.0, 0.0];
        for _ in 0..10 {
            assert_eq!(sample_action(&strategy), 1);
        }
    }

    #[test]
    fn test_hands_conflict() {
        assert!(hands_conflict((0, 1), (1, 2))); // Share card 1
        assert!(hands_conflict((0, 1), (0, 2))); // Share card 0
        assert!(!hands_conflict((0, 1), (2, 3))); // No conflict
    }

    #[test]
    fn test_sample_weighted() {
        let weights = vec![0.0, 0.0, 1.0];
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            assert_eq!(sample_weighted(&weights, &mut rng), 2);
        }
    }
}
