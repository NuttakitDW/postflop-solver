//! Supervised Learning POC for Deep PDCFR+
//!
//! This POC verifies that neural networks can learn from tabular CFR solutions.
//!
//! Steps:
//! 1. Run tabular DCFR on a single board (Td9d6h from medium.json)
//! 2. Extract (features, strategy) pairs from solved game
//! 3. Train strategy network to predict those strategies
//! 4. Compare predictions vs ground truth
//!
//! Run with: cargo run --example supervised_poc --release --features "deep bincode zstd"

use postflop_solver::*;
use std::time::Instant;

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

/// Training sample: (features, target_strategy, num_actions)
#[cfg(feature = "deep")]
struct TrainingSample {
    features: Vec<f32>,
    strategy: Vec<f32>,
    player: usize,
}

/// Simple strategy network for POC
#[cfg(feature = "deep")]
struct SimpleStrategyNet {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
    output: Linear,
}

#[cfg(feature = "deep")]
impl SimpleStrategyNet {
    fn new(vs: VarBuilder, input_dim: usize, num_actions: usize) -> candle_core::Result<Self> {
        let layer1 = linear(input_dim, 256, vs.pp("l1"))?;
        let layer2 = linear(256, 128, vs.pp("l2"))?;
        let layer3 = linear(128, 64, vs.pp("l3"))?;
        let output = linear(64, num_actions, vs.pp("out"))?;
        Ok(Self { layer1, layer2, layer3, output })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.layer1.forward(x)?.relu()?;
        let x = self.layer2.forward(&x)?.relu()?;
        let x = self.layer3.forward(&x)?.relu()?;
        self.output.forward(&x)
    }

    fn forward_softmax(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(x)?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }
}

/// Encode a game node to feature vector (simplified version)
#[cfg(feature = "deep")]
fn encode_node_simple(
    game: &PostFlopGame,
    flop: [u8; 3],
    turn: u8,
    river: u8,
    hole_cards: (u8, u8),
    street: usize,
    pot: i32,
    stack: i32,
) -> Vec<f32> {
    // Simplified encoding: 369 dims to match InfoSetEncoder
    // Board: 5 * 52 = 260
    // Hole: 2 * 52 = 104
    // Pot ratio: 1
    // Stack ratio: 1
    // Street: 3
    // Actions: 0 (simplified for POC)
    let mut features = vec![0.0f32; 369];

    let tree_config = game.tree_config();
    let starting_pot = tree_config.starting_pot as f32;
    let effective_stack = tree_config.effective_stack as f32;

    // Board cards (one-hot)
    let mut offset = 0;
    for (i, &card) in flop.iter().enumerate() {
        if card < 52 {
            features[offset + i * 52 + card as usize] = 1.0;
        }
    }
    offset += 3 * 52;

    // Turn
    if turn < 52 {
        features[offset + turn as usize] = 1.0;
    }
    offset += 52;

    // River
    if river < 52 {
        features[offset + river as usize] = 1.0;
    }
    offset += 52;

    // Hole cards (one-hot)
    let (c1, c2) = if hole_cards.0 <= hole_cards.1 { hole_cards } else { (hole_cards.1, hole_cards.0) };
    if c1 < 52 {
        features[offset + c1 as usize] = 1.0;
    }
    offset += 52;
    if c2 < 52 {
        features[offset + c2 as usize] = 1.0;
    }
    offset += 52;

    // Pot and stack ratios
    features[offset] = pot as f32 / (starting_pot + 2.0 * effective_stack);
    features[offset + 1] = stack as f32 / effective_stack;
    offset += 2;

    // Street (one-hot)
    if street < 3 {
        features[offset + street] = 1.0;
    }

    features
}

/// Collect training samples from solved game tree
#[cfg(feature = "deep")]
fn collect_training_samples(game: &PostFlopGame) -> Vec<TrainingSample> {
    let mut samples = Vec::new();
    let flop = game.card_config().flop;
    let tree_config = game.tree_config();
    let starting_pot = tree_config.starting_pot;
    let effective_stack = tree_config.effective_stack;

    // Get private hands for both players
    let private_cards = [game.private_cards(0), game.private_cards(1)];

    // Recursive traversal
    fn traverse(
        game: &PostFlopGame,
        node: &PostFlopNode,
        samples: &mut Vec<TrainingSample>,
        flop: [u8; 3],
        private_cards: &[&[(u8, u8)]; 2],
        starting_pot: i32,
        effective_stack: i32,
        depth: usize,
    ) {
        if depth > 50 || node.is_terminal() {
            return;
        }

        if node.is_chance() {
            // Traverse all chance outcomes (limit for POC)
            let num_chances = node.num_actions().min(10); // Limit for speed
            for i in 0..num_chances {
                let child = node.play(i);
                traverse(game, &child, samples, flop, private_cards, starting_pot, effective_stack, depth + 1);
            }
            return;
        }

        let player = node.player();
        let num_actions = node.num_actions();

        if num_actions == 0 || player > 1 {
            return;
        }

        // Get turn/river from node
        let turn = node.turn_card();
        let river = node.river_card();

        // Determine street
        let street = if turn == NOT_DEALT { 0 } else if river == NOT_DEALT { 1 } else { 2 };

        // Get strategy from solved game (averaged over all hands)
        // Strategy is stored per-hand, so we need to get the aggregate
        let strategy = node.strategy();

        // The strategy array is: [action0_hand0, action0_hand1, ..., action1_hand0, ...]
        // Size = num_actions * num_private_hands
        let num_hands = private_cards[player].len();

        if strategy.len() == num_actions * num_hands && num_hands > 0 {
            // Create samples for each hand (limit to 50 hands per node for POC speed)
            for (hand_idx, &(c1, c2)) in private_cards[player].iter().enumerate().take(50) {
                // Skip if hole cards conflict with board
                if [flop[0], flop[1], flop[2], turn, river].contains(&c1) ||
                   [flop[0], flop[1], flop[2], turn, river].contains(&c2) {
                    continue;
                }

                // Extract strategy for this hand
                let mut hand_strategy = Vec::with_capacity(num_actions);
                for a in 0..num_actions {
                    hand_strategy.push(strategy[a * num_hands + hand_idx]);
                }

                // Normalize (in case of floating point issues)
                let sum: f32 = hand_strategy.iter().sum();
                if sum > 0.01 {
                    for s in &mut hand_strategy {
                        *s /= sum;
                    }

                    // Compute pot/stack for this node
                    let current_pot = starting_pot + 2 * (effective_stack - node.bet_amount());

                    // Encode features
                    let features = encode_node_simple(
                        game,
                        flop,
                        if turn == NOT_DEALT { 255 } else { turn },
                        if river == NOT_DEALT { 255 } else { river },
                        (c1, c2),
                        street,
                        current_pot,
                        node.bet_amount(),
                    );

                    samples.push(TrainingSample {
                        features,
                        strategy: hand_strategy,
                        player,
                    });
                }
            }
        }

        // Traverse children (limit for POC speed)
        for i in 0..num_actions.min(4) {
            let child = node.play(i);
            traverse(game, &child, samples, flop, private_cards, starting_pot, effective_stack, depth + 1);
        }
    }

    let root = game.root();
    let private_cards_ref = [private_cards[0], private_cards[1]];
    traverse(game, &root, &mut samples, flop, &private_cards_ref, starting_pot, effective_stack, 0);

    samples
}

#[cfg(feature = "deep")]
fn run_poc() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Supervised Learning POC ===\n");

    // Step 1: Setup game (Td9d6h from medium.json)
    println!("Step 1: Setting up game...");
    let start = Instant::now();

    // Parse ranges (simplified from medium.json)
    let oop_range: Range = "66,55,44,33,22,ATs-A2s,A9o-A2o,KTs-K2s,KQo-K6o,QJs-Q2s,QJo-Q2o,J8s+,J6s-J2s,JTo-J4o,T8s+,T5s-T2s,T9o-T6o,98s,97s,96s,95s-92s,98o-96o,87s,86s,85s-82s,86o+,76s,75s-72s,75o+,62s+,65o,64o,63o,53o,52s+,42s+,32s".parse()?;
    let ip_range: Range = "22+,A2s+,A2o+,K2s+,K6o+,Q2s+,Q8o+,J2s+,J8o+,T3s+,T8o+,95s+,98o,85s+,87o,74s+,64s+,53s+".parse()?;

    // Parse flop: Td9d6h
    // T=8 (rank), d=1 (suit) => 4*8+1 = 33
    // 9=7, d=1 => 4*7+1 = 29
    // 6=4, h=2 => 4*4+2 = 18
    let flop = flop_from_str("Td9d6h")?;

    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    // Simplified bet sizes for faster POC
    let bet_sizes = BetSizeOptions::try_from(("33%", "60%"))?;

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 61,
        effective_stack: 477,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.2,
        merging_threshold: 0.1,
        max_raises_per_street: 3,
    };

    let action_tree = ActionTree::new(tree_config.clone())?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;

    let (mem_usage, _) = game.memory_usage();
    println!("  Memory usage: {:.2} MB", mem_usage as f64 / 1024.0 / 1024.0);
    println!("  OOP hands: {}, IP hands: {}", game.private_cards(0).len(), game.private_cards(1).len());
    println!("  Setup time: {:.2}s", start.elapsed().as_secs_f64());

    // Step 2: Solve with tabular DCFR
    println!("\nStep 2: Solving with tabular DCFR...");
    let solve_start = Instant::now();

    game.allocate_memory(false); // No compression for POC

    // Solve to high accuracy (0.3% exploitability for quality ground truth)
    let target_exploitability = 61.0 * 0.003; // 0.3% of pot
    let exploitability = solve(&mut game, 1000, target_exploitability, true);

    println!("  Final exploitability: {:.4} ({:.2}% of pot)",
             exploitability, exploitability / 61.0 * 100.0);
    println!("  Solve time: {:.2}s", solve_start.elapsed().as_secs_f64());

    // Step 3: Collect training samples
    println!("\nStep 3: Collecting training samples...");
    let collect_start = Instant::now();

    let samples = collect_training_samples(&game);
    println!("  Collected {} samples", samples.len());
    println!("  Collection time: {:.2}s", collect_start.elapsed().as_secs_f64());

    if samples.is_empty() {
        println!("  ERROR: No samples collected!");
        return Ok(());
    }

    // Analyze sample statistics
    let num_actions_dist: std::collections::HashMap<usize, usize> = samples.iter()
        .map(|s| s.strategy.len())
        .fold(std::collections::HashMap::new(), |mut acc, n| {
            *acc.entry(n).or_insert(0) += 1;
            acc
        });
    println!("  Action count distribution: {:?}", num_actions_dist);

    // Step 4: Train neural network
    println!("\nStep 4: Training neural network...");
    let train_start = Instant::now();

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);

    // Find max actions in samples
    let max_actions = samples.iter().map(|s| s.strategy.len()).max().unwrap_or(6);
    println!("  Max actions: {}", max_actions);

    let net = SimpleStrategyNet::new(vs, 369, max_actions)?;

    let params = candle_nn::ParamsAdamW {
        lr: 0.001,
        ..Default::default()
    };
    let mut optimizer = candle_nn::AdamW::new(var_map.all_vars(), params)?;

    // Training loop (larger batch for speed, more epochs for convergence)
    let batch_size = 2048;
    let num_epochs = 100;
    let num_batches = (samples.len() + batch_size - 1) / batch_size;

    println!("  Training with {} samples, {} epochs, batch_size={}",
             samples.len(), num_epochs, batch_size);

    let mut losses = Vec::new();

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut num_samples_processed = 0;

        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size;
            let end_idx = (start_idx + batch_size).min(samples.len());
            let batch = &samples[start_idx..end_idx];

            if batch.is_empty() {
                continue;
            }

            // Filter samples with same action count (for batching)
            let first_num_actions = batch[0].strategy.len();
            let valid_batch: Vec<_> = batch.iter()
                .filter(|s| s.strategy.len() == first_num_actions)
                .collect();

            if valid_batch.is_empty() {
                continue;
            }

            let actual_batch_size = valid_batch.len();

            // Prepare batch tensors
            let features_flat: Vec<f32> = valid_batch.iter()
                .flat_map(|s| s.features.iter().copied())
                .collect();

            // Pad strategies to max_actions
            let targets_flat: Vec<f32> = valid_batch.iter()
                .flat_map(|s| {
                    let mut padded = s.strategy.clone();
                    padded.resize(max_actions, 0.0);
                    padded
                })
                .collect();

            let features_tensor = Tensor::from_vec(
                features_flat, (actual_batch_size, 369), &device
            )?;
            let targets_tensor = Tensor::from_vec(
                targets_flat, (actual_batch_size, max_actions), &device
            )?;

            // Forward pass
            let logits = net.forward(&features_tensor)?;

            // Cross-entropy loss with target probabilities
            let log_probs = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let ce = (&targets_tensor * &log_probs)?.neg()?;
            let loss_sum = ce.sum_all()?;
            let loss = (&loss_sum / actual_batch_size as f64)?;

            // Backward pass
            optimizer.backward_step(&loss)?;

            let loss_val: f32 = loss.to_scalar()?;
            epoch_loss += loss_val * actual_batch_size as f32;
            num_samples_processed += actual_batch_size;
        }

        let avg_loss = epoch_loss / num_samples_processed.max(1) as f32;
        losses.push(avg_loss);

        if epoch % 20 == 0 || epoch == num_epochs - 1 {
            println!("  Epoch {}: loss = {:.6}", epoch, avg_loss);
        }
    }

    println!("  Training time: {:.2}s", train_start.elapsed().as_secs_f64());

    // Step 5: Evaluate on training data
    println!("\nStep 5: Evaluating predictions...");

    let mut total_kl_div = 0.0;
    let mut total_max_diff = 0.0;
    let mut num_evaluated = 0;

    // Evaluate on random samples
    for (i, sample) in samples.iter().enumerate().step_by(samples.len() / 20 + 1) {
        let features_tensor = Tensor::from_vec(
            sample.features.clone(), (1, 369), &device
        )?;

        let predicted = net.forward_softmax(&features_tensor)?;
        let predicted_vec: Vec<f32> = predicted.flatten_all()?.to_vec1()?;

        // Compute KL divergence and max absolute difference
        let num_actions = sample.strategy.len();
        let mut kl_div: f32 = 0.0;
        let mut max_diff: f32 = 0.0;

        for a in 0..num_actions {
            let p = sample.strategy[a].max(1e-8);
            let q = predicted_vec[a].max(1e-8);
            kl_div += p * (p / q).ln();
            max_diff = max_diff.max((p - q).abs());
        }

        total_kl_div += kl_div;
        total_max_diff += max_diff;
        num_evaluated += 1;

        if i < 5 {
            println!("  Sample {}: target={:?}", i, &sample.strategy[..sample.strategy.len().min(4)]);
            println!("            pred  ={:?}", &predicted_vec[..sample.strategy.len().min(4)]);
            println!("            KL={:.4}, MaxDiff={:.4}", kl_div, max_diff);
        }
    }

    let avg_kl = total_kl_div / num_evaluated as f32;
    let avg_max_diff = total_max_diff / num_evaluated as f32;

    println!("\n=== Results ===");
    println!("  Average KL divergence: {:.6}", avg_kl);
    println!("  Average max difference: {:.4}", avg_max_diff);
    println!("  Final training loss: {:.6}", losses.last().unwrap_or(&0.0));
    println!("  Total time: {:.2}s", start.elapsed().as_secs_f64());

    // Success criteria
    if avg_kl < 0.5 && avg_max_diff < 0.2 {
        println!("\n✓ POC SUCCESS: Neural network learned the tabular strategies!");
    } else {
        println!("\n✗ POC needs more training or architecture tuning");
    }

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn run_poc() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("This example requires the 'deep' feature.");
    eprintln!("Run with: cargo run --example supervised_poc --release --features \"deep bincode zstd\"");
    Ok(())
}

fn main() {
    if let Err(e) = run_poc() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
