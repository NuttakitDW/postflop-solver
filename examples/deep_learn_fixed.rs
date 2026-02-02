//! Deep PDCFR+ POC - FIXED version using correct API
//!
//! Run: cargo run --example deep_learn_fixed --release --features "deep bincode zstd"

use postflop_solver::*;
use std::time::Instant;

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

#[cfg(feature = "deep")]
struct Sample {
    features: Vec<f32>,
    strategy: Vec<f32>,
}

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

    fn predict(&self, x: &Tensor) -> candle_core::Result<Tensor> {
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

    // Board (5 * 52 = 260)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }
    if turn < 52 { f[156 + turn as usize] = 1.0; }
    if river < 52 { f[208 + river as usize] = 1.0; }

    // Hole cards (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[260 + c1 as usize] = 1.0; }
    if c2 < 52 { f[312 + c2 as usize] = 1.0; }

    // Ratios
    f[364] = pot_ratio;
    f[365] = stack_ratio;

    // Street
    if street < 3 { f[366 + street] = 1.0; }

    f
}

/// FIXED: Collect samples using game.strategy() instead of node.strategy()
#[cfg(feature = "deep")]
fn collect_samples_fixed(game: &mut PostFlopGame) -> Vec<Sample> {
    let mut samples = Vec::new();
    let flop = game.card_config().flop;
    let tc = game.tree_config();
    let pot0 = tc.starting_pot as f32;
    let stack0 = tc.effective_stack as f32;

    // Collect from root node (OOP flop)
    game.cache_normalized_weights();

    let player = game.current_player();
    let hands = game.private_cards(player).to_vec();
    let num_hands = hands.len();
    let actions = game.available_actions();
    let num_actions = actions.len();

    // Use game.strategy() - the CORRECT API!
    let strategy = game.strategy();

    let turn = NOT_DEALT;
    let river = NOT_DEALT;
    let street = 0; // flop

    // Calculate pot_ratio and stack_ratio for root
    let pot = pot0 + 2.0 * (stack0 - 0.0); // bet_amount = 0 at root
    let pot_ratio = pot / (pot0 + 2.0 * stack0);
    let stack_ratio = 0.0;

    println!("  Collecting from root: player={}, hands={}, actions={}", player, num_hands, num_actions);
    println!("  pot_ratio={:.3}, stack_ratio={:.3}", pot_ratio, stack_ratio);

    for (hi, &(c1, c2)) in hands.iter().enumerate() {
        // Skip if conflicts with board
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        // Extract strategy for this hand
        let mut strat = Vec::with_capacity(num_actions);
        for a in 0..num_actions {
            strat.push(strategy[a * num_hands + hi]);
        }

        // Check if valid (should already be normalized by game.strategy())
        let sum: f32 = strat.iter().sum();
        if sum > 0.5 && sum < 1.5 { // Should be ~1.0 after normalization
            let t = if turn == NOT_DEALT { 255 } else { turn };
            let r = if river == NOT_DEALT { 255 } else { river };

            samples.push(Sample {
                features: encode(flop, t, r, (c1, c2), pot_ratio, stack_ratio, street),
                strategy: strat,
            });
        }
    }

    println!("  Collected {} samples from root", samples.len());

    // Show sample strategies
    if !samples.is_empty() {
        let avg_check: f32 = samples.iter().map(|s| s.strategy[0]).sum::<f32>() / samples.len() as f32;
        let avg_bet: f32 = samples.iter().map(|s| s.strategy.get(1).unwrap_or(&0.0)).sum::<f32>() / samples.len() as f32;
        println!("  Average strategy: Check={:.1}%, Bet={:.1}%", avg_check * 100.0, avg_bet * 100.0);
    }

    samples
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Deep PDCFR+ POC - FIXED VERSION ===\n");
    let total_start = Instant::now();

    // Step 1: Load solved game
    println!("Step 1: Loading DCFR solution...");
    let load_start = Instant::now();

    let (mut game, _memo): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;

    println!("  Loaded in {:.1}s", load_start.elapsed().as_secs_f64());
    println!("  OOP hands: {}, IP hands: {}", game.private_cards(0).len(), game.private_cards(1).len());

    // Step 2: Collect samples using FIXED method
    println!("\nStep 2: Collecting training samples (FIXED)...");
    let collect_start = Instant::now();

    let samples = collect_samples_fixed(&mut game);
    let max_actions = samples.iter().map(|s| s.strategy.len()).max().unwrap_or(2);

    println!("  Total samples: {}", samples.len());
    println!("  Max actions: {}", max_actions);
    println!("  Time: {:.1}s", collect_start.elapsed().as_secs_f64());

    if samples.is_empty() {
        println!("ERROR: No samples collected!");
        return Ok(());
    }

    // Step 3: Train
    println!("\nStep 3: Training neural network...");
    let train_start = Instant::now();

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 369, max_actions)?;

    let mut opt = candle_nn::AdamW::new(
        var_map.all_vars(),
        candle_nn::ParamsAdamW { lr: 0.001, ..Default::default() },
    )?;

    let batch_size = 512;
    let epochs = 100;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batches = 0;

        for batch in samples.chunks(batch_size) {
            let bs = batch.len();
            let feats: Vec<f32> = batch.iter().flat_map(|s| s.features.iter().copied()).collect();
            let targs: Vec<f32> = batch.iter().flat_map(|s| {
                let mut t = s.strategy.clone();
                t.resize(max_actions, 0.0);
                t
            }).collect();

            let x = Tensor::from_vec(feats, (bs, 369), &device)?;
            let y = Tensor::from_vec(targs, (bs, max_actions), &device)?;

            let logits = net.forward(&x)?;
            let log_p = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let loss = (&y * &log_p)?.neg()?.sum_all()?;
            let loss = (&loss / bs as f64)?;

            opt.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;
            batches += 1;
        }

        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: loss = {:.4}", epoch, total_loss / batches.max(1) as f32);
        }
    }
    println!("  Training time: {:.1}s", train_start.elapsed().as_secs_f64());

    // Step 4: Test
    println!("\nStep 4: Testing...");

    let mut pred_check_total = 0.0f32;
    let mut pred_bet_total = 0.0f32;
    let mut target_check_total = 0.0f32;
    let mut target_bet_total = 0.0f32;

    for sample in &samples {
        let x = Tensor::from_vec(sample.features.clone(), (1, 369), &device)?;
        let pred = net.predict(&x)?;
        let pred_vec: Vec<f32> = pred.flatten_all()?.to_vec1()?;

        pred_check_total += pred_vec[0];
        pred_bet_total += pred_vec.get(1).unwrap_or(&0.0);
        target_check_total += sample.strategy[0];
        target_bet_total += sample.strategy.get(1).unwrap_or(&0.0);
    }

    let n = samples.len() as f32;
    println!("\n  Target:     Check {:.1}%, Bet {:.1}%", target_check_total / n * 100.0, target_bet_total / n * 100.0);
    println!("  Predicted:  Check {:.1}%, Bet {:.1}%", pred_check_total / n * 100.0, pred_bet_total / n * 100.0);

    let check_error = ((target_check_total - pred_check_total) / n).abs() * 100.0;
    let bet_error = ((target_bet_total - pred_bet_total) / n).abs() * 100.0;
    let total_error = check_error + bet_error;

    println!("\n  Check error: {:.1}%", check_error);
    println!("  Bet error: {:.1}%", bet_error);
    println!("  Total error: {:.1}%", total_error);

    // Summary
    println!("\n========== SUMMARY ==========");
    println!("Samples:       {}", samples.len());
    println!("Training time: {:.1}s", train_start.elapsed().as_secs_f64());
    println!("Total error:   {:.1}%", total_error);
    println!("Total time:    {:.1}s", total_start.elapsed().as_secs_f64());
    println!("==============================");

    if total_error < 5.0 {
        println!("\n✓ SUCCESS: Pipeline FIXED! Network learned Check 100%");
    } else {
        println!("\n✗ Still has error > 5%");
    }

    // Save weights
    var_map.save("out/deep_fixed_weights.safetensors")?;
    println!("\nWeights saved to: out/deep_fixed_weights.safetensors");

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_learn_fixed --release --features \"deep bincode zstd\"");
}
