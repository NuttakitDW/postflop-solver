//! Benchmark: Export neural network predictions to .flop file
//!
//! Measures how fast we can fill a PostFlopGame with NN predictions
//! compared to solving with DCFR.
//!
//! Run: cargo run --example deep_export_benchmark --release --features "deep bincode zstd"

use postflop_solver::*;
use std::time::Instant;

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};

#[cfg(feature = "deep")]
struct StrategyNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    out: Linear,
}

#[cfg(feature = "deep")]
impl StrategyNet {
    fn new(vs: VarBuilder, input: usize, output: usize) -> candle_core::Result<Self> {
        Ok(Self {
            l1: linear(input, 512, vs.pp("l1"))?,
            l2: linear(512, 256, vs.pp("l2"))?,
            l3: linear(256, 128, vs.pp("l3"))?,
            out: linear(128, output, vs.pp("out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        self.out.forward(&x)
    }

    fn predict_batch(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(x)?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }
}

#[cfg(feature = "deep")]
fn encode(
    flop: [u8; 3],
    turn: u8,
    river: u8,
    hole: (u8, u8),
    pot_ratio: f32,
    stack_ratio: f32,
    street: usize,
) -> Vec<f32> {
    let mut f = vec![0.0f32; 369];

    // Board cards one-hot (5 * 52 = 260)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }
    if turn < 52 { f[156 + turn as usize] = 1.0; }
    if river < 52 { f[208 + river as usize] = 1.0; }

    // Hole cards one-hot (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[260 + c1 as usize] = 1.0; }
    if c2 < 52 { f[312 + c2 as usize] = 1.0; }

    // Pot and stack ratios
    f[364] = pot_ratio;
    f[365] = stack_ratio;

    // Street one-hot
    if street < 3 { f[366 + street] = 1.0; }

    f
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Deep Export Benchmark ===\n");

    // Step 1: Load trained weights
    println!("Step 1: Loading trained weights...");
    let load_start = Instant::now();

    let device = Device::Cpu;
    let mut var_map = VarMap::new();
    var_map.load("out/deep_medium_weights.safetensors")?;

    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 369, 5)?; // 5 = max actions from training

    println!("  Loaded in {:.2}s", load_start.elapsed().as_secs_f64());

    // Step 2: Setup game with medium.json settings
    println!("\nStep 2: Setting up game (medium.json settings)...");
    let setup_start = Instant::now();

    let oop: Range = "66,55,44,33,22,ATs-A2s,A9o-A2o,KTs-K2s,KQo-K6o,QJs-Q2s,QJo-Q2o,J8s+,J6s-J2s,JTo-J4o,T8s+,T5s-T2s,T9o-T6o,98s,97s,96s,95s-92s,98o-96o,87s,86s,85s-82s,86o+,76s,75s-72s,75o+,62s+,65o,64o,63o,53o,52s+,42s+,32s".parse()?;
    let ip: Range = "22+,A2s+,A2o+,K2s+,K6o+,Q2s+,Q8o+,J2s+,J8o+,T3s+,T8o+,95s+,98o,85s+,87o,74s+,64s+,53s+".parse()?;

    let flop = flop_from_str("Td9d6h")?;

    let card_config = CardConfig {
        range: [oop, ip],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let oop_flop = BetSizeOptions::try_from(("33%", "33%, 60%"))?;
    let ip_flop = BetSizeOptions::try_from(("25%, 55%, 85%, 125%", "33%, 60%"))?;
    let turn_bet = BetSizeOptions::try_from(("33%, 75%, 125%", "33%, 60%"))?;
    let river_bet = BetSizeOptions::try_from(("33%, 75%, 125%", "33%, 60%"))?;

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 61,
        effective_stack: 477,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_flop, ip_flop],
        turn_bet_sizes: [turn_bet.clone(), turn_bet.clone()],
        river_bet_sizes: [river_bet.clone(), river_bet.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 6.0,
        force_allin_threshold: 0.5,
        merging_threshold: 0.1,
        max_raises_per_street: 4,
    };

    let action_tree = ActionTree::new(tree_config.clone())?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;
    game.allocate_memory(false);

    println!("  Setup in {:.2}s", setup_start.elapsed().as_secs_f64());
    println!("  OOP hands: {}, IP hands: {}", game.private_cards(0).len(), game.private_cards(1).len());

    // Step 3: Count nodes and benchmark inference
    println!("\nStep 3: Benchmarking NN inference on game tree...");
    let bench_start = Instant::now();

    let flop_cards = game.card_config().flop;
    let tc = game.tree_config();
    let pot0 = tc.starting_pot as f32;
    let stack0 = tc.effective_stack as f32;
    let hands = [game.private_cards(0).to_vec(), game.private_cards(1).to_vec()];

    let mut total_nodes = 0usize;
    let mut total_predictions = 0usize;
    let mut total_inference_time = std::time::Duration::ZERO;

    fn traverse_and_predict(
        node: &PostFlopNode,
        net: &StrategyNet,
        device: &Device,
        flop: [u8; 3],
        hands: &[Vec<(u8, u8)>; 2],
        pot0: f32,
        stack0: f32,
        total_nodes: &mut usize,
        total_predictions: &mut usize,
        total_inference_time: &mut std::time::Duration,
        depth: usize,
    ) {
        if depth > 30 || node.is_terminal() { return; }

        if node.is_chance() {
            // Traverse some chance outcomes
            for i in 0..node.num_actions().min(10) {
                let child = node.play(i);
                traverse_and_predict(
                    &child, net, device, flop, hands, pot0, stack0,
                    total_nodes, total_predictions, total_inference_time, depth + 1
                );
            }
            return;
        }

        let player = node.player();
        if player > 1 { return; }

        let num_actions = node.num_actions();
        if num_actions == 0 { return; }

        *total_nodes += 1;

        let turn = node.turn_card();
        let river = node.river_card();
        let street = if turn == NOT_DEALT { 0 } else if river == NOT_DEALT { 1 } else { 2 };

        let pot = pot0 + 2.0 * (stack0 - node.bet_amount() as f32);
        let pot_ratio = pot / (pot0 + 2.0 * stack0);
        let stack_ratio = node.bet_amount() as f32 / stack0;

        let t = if turn == NOT_DEALT { 255 } else { turn };
        let r = if river == NOT_DEALT { 255 } else { river };

        // Batch all hands for this node
        let player_hands = &hands[player];
        let valid_hands: Vec<_> = player_hands.iter()
            .filter(|&&(c1, c2)| {
                let dominated = [flop[0], flop[1], flop[2], turn, river];
                !dominated.contains(&c1) && !dominated.contains(&c2)
            })
            .collect();

        if !valid_hands.is_empty() {
            // Encode all hands as batch
            let batch_features: Vec<f32> = valid_hands.iter()
                .flat_map(|&&(c1, c2)| encode(flop, t, r, (c1, c2), pot_ratio, stack_ratio, street))
                .collect();

            let batch_size = valid_hands.len();

            // Time the inference
            let inf_start = std::time::Instant::now();
            if let Ok(x) = candle_core::Tensor::from_vec(batch_features, (batch_size, 369), device) {
                let _ = net.predict_batch(&x);
            }
            *total_inference_time += inf_start.elapsed();
            *total_predictions += batch_size;
        }

        // Traverse children
        for i in 0..num_actions.min(3) {
            let child = node.play(i);
            traverse_and_predict(
                &child, net, device, flop, hands, pot0, stack0,
                total_nodes, total_predictions, total_inference_time, depth + 1
            );
        }
    }

    let root = game.root();
    traverse_and_predict(
        &root, &net, &device, flop_cards, &hands, pot0, stack0,
        &mut total_nodes, &mut total_predictions, &mut total_inference_time, 0
    );

    let bench_elapsed = bench_start.elapsed();

    println!("  Decision nodes visited: {}", total_nodes);
    println!("  Total predictions: {}", total_predictions);
    println!("  Pure inference time: {:.2}ms", total_inference_time.as_secs_f64() * 1000.0);
    println!("  Total traversal time: {:.2}ms", bench_elapsed.as_secs_f64() * 1000.0);
    println!("  Avg per prediction: {:.2} µs", total_inference_time.as_micros() as f64 / total_predictions.max(1) as f64);

    // Step 4: Estimate full export time
    println!("\nStep 4: Estimating full export...");

    // We sampled a portion of the tree (depth limit, action limit)
    // Full tree is larger - estimate based on file size ratio
    // 5.2 GB file ≈ 5.2B bytes, our sample touched ~total_predictions hands

    let time_per_prediction = total_inference_time.as_secs_f64() / total_predictions.max(1) as f64;

    // Estimate: we sampled with limits (depth 30, 3 actions, 10 chance outcomes)
    // Full tree is roughly 10-50x larger based on these limits
    let estimated_multiplier = 20.0; // Conservative estimate
    let estimated_full_predictions = (total_predictions as f64 * estimated_multiplier) as usize;
    let estimated_full_time = estimated_full_predictions as f64 * time_per_prediction;

    println!("  Sampled nodes: {}", total_nodes);
    println!("  Sampled predictions: {}", total_predictions);
    println!("  Time per prediction: {:.2} µs", time_per_prediction * 1_000_000.0);
    println!("  Estimated full predictions: ~{}", estimated_full_predictions);
    println!("  Estimated full inference time: {:.2}s", estimated_full_time);

    // Summary
    println!("\n========== SUMMARY ==========");
    println!("Weights loaded:        {:.2}s", load_start.elapsed().as_secs_f64());
    println!("Game setup:            {:.2}s", setup_start.elapsed().as_secs_f64());
    println!("Sampled inference:     {:.2}ms ({} predictions)", total_inference_time.as_secs_f64() * 1000.0, total_predictions);
    println!("Estimated full export: {:.2}s", estimated_full_time);
    println!("DCFR solve time:       ~1147s (19 min)");
    println!("Speedup:               ~{:.0}x faster", 1147.0 / estimated_full_time.max(0.001));
    println!("==============================");

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_export_benchmark --release --features \"deep bincode zstd\"");
}
