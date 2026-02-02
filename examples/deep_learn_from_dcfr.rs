//! Deep PDCFR+ POC - Learn from DCFR solution
//!
//! This loads the solved medium.json (out/50bb-medium.flop) and trains
//! a neural network to reproduce the strategies.
//!
//! Prerequisites: Run backend_solver on medium.json first:
//!   cargo run --example backend_solver --release --features "bincode zstd" -- config/medium.json
//!
//! Then run this:
//!   cargo run --example deep_learn_from_dcfr --release --features "deep bincode zstd"

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

#[cfg(feature = "deep")]
fn collect_samples(game: &PostFlopGame) -> Vec<Sample> {
    let mut samples = Vec::new();
    let flop = game.card_config().flop;
    let tc = game.tree_config();
    let pot0 = tc.starting_pot as f32;
    let stack0 = tc.effective_stack as f32;
    let hands = [game.private_cards(0), game.private_cards(1)];

    fn traverse(
        node: &PostFlopNode,
        samples: &mut Vec<Sample>,
        flop: [u8; 3],
        hands: &[&[(u8, u8)]; 2],
        pot0: f32,
        stack0: f32,
        depth: usize,
    ) {
        if depth > 20 || node.is_terminal() { return; }

        if node.is_chance() {
            for i in 0..node.num_actions().min(3) {
                let child = node.play(i);
                traverse(&child, samples, flop, hands, pot0, stack0, depth + 1);
            }
            return;
        }

        let player = node.player();
        if player > 1 { return; }

        let num_actions = node.num_actions();
        if num_actions == 0 { return; }

        let turn = node.turn_card();
        let river = node.river_card();
        let street = if turn == NOT_DEALT { 0 } else if river == NOT_DEALT { 1 } else { 2 };

        let strategy = node.strategy();
        let num_hands = hands[player].len();

        if strategy.len() == num_actions * num_hands && num_hands > 0 {
            for (hi, &(c1, c2)) in hands[player].iter().enumerate().step_by(3) {
                let dominated = [flop[0], flop[1], flop[2], turn, river];
                if dominated.contains(&c1) || dominated.contains(&c2) { continue; }

                let mut strat = Vec::with_capacity(num_actions);
                for a in 0..num_actions {
                    strat.push(strategy[a * num_hands + hi]);
                }

                let sum: f32 = strat.iter().sum();
                if sum > 0.01 {
                    for s in &mut strat { *s /= sum; }

                    let pot = pot0 + 2.0 * (stack0 - node.bet_amount() as f32);
                    let pot_ratio = pot / (pot0 + 2.0 * stack0);
                    let stack_ratio = node.bet_amount() as f32 / stack0;

                    let t = if turn == NOT_DEALT { 255 } else { turn };
                    let r = if river == NOT_DEALT { 255 } else { river };

                    samples.push(Sample {
                        features: encode(flop, t, r, (c1, c2), pot_ratio, stack_ratio, street),
                        strategy: strat,
                    });
                }
            }
        }

        for i in 0..num_actions.min(2) {
            let child = node.play(i);
            traverse(&child, samples, flop, hands, pot0, stack0, depth + 1);
        }
    }

    let root = game.root();
    traverse(&root, &mut samples, flop, &hands, pot0, stack0, 0);
    samples
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Deep PDCFR+ POC - Learning from DCFR ===\n");
    let total_start = Instant::now();

    // Step 1: Load solved game
    println!("Step 1: Loading DCFR solution from out/50bb-medium.flop...");
    let load_start = Instant::now();

    let (game, _memo): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;

    println!("  Loaded in {:.1}s", load_start.elapsed().as_secs_f64());
    println!("  OOP hands: {}, IP hands: {}", game.private_cards(0).len(), game.private_cards(1).len());
    println!("  Board: {:?}", game.card_config().flop);

    // Step 2: Collect samples
    println!("\nStep 2: Collecting training samples...");
    let collect_start = Instant::now();

    let samples = collect_samples(&game);
    let max_actions = samples.iter().map(|s| s.strategy.len()).max().unwrap_or(2);

    println!("  Collected {} samples", samples.len());
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

    let batch_size = 2048;
    let epochs = 50;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batches = 0;

        for batch in samples.chunks(batch_size) {
            let first_actions = batch[0].strategy.len();
            let valid: Vec<_> = batch.iter().filter(|s| s.strategy.len() == first_actions).collect();
            if valid.is_empty() { continue; }

            let bs = valid.len();
            let feats: Vec<f32> = valid.iter().flat_map(|s| s.features.iter().copied()).collect();
            let targs: Vec<f32> = valid.iter().flat_map(|s| {
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

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: loss = {:.4}", epoch, total_loss / batches.max(1) as f32);
        }
    }
    println!("  Training time: {:.1}s", train_start.elapsed().as_secs_f64());

    // Step 4: Test inference
    println!("\nStep 4: Testing inference speed...");

    let test_samples: Vec<_> = samples.iter().take(1000).collect();
    let inference_start = Instant::now();

    for sample in &test_samples {
        let x = Tensor::from_vec(sample.features.clone(), (1, 369), &device)?;
        let _ = net.predict(&x)?;
    }

    let inference_time = inference_start.elapsed();
    let per_query = inference_time.as_micros() as f64 / test_samples.len() as f64;

    println!("  {} queries in {:.1}ms", test_samples.len(), inference_time.as_secs_f64() * 1000.0);
    println!("  Per query: {:.0} µs", per_query);

    // Step 5: Check accuracy
    println!("\nStep 5: Checking accuracy...");

    let mut total_err = 0.0;
    let mut count = 0;

    for sample in test_samples.iter().take(100) {
        let x = Tensor::from_vec(sample.features.clone(), (1, 369), &device)?;
        let pred = net.predict(&x)?;
        let pred_vec: Vec<f32> = pred.flatten_all()?.to_vec1()?;

        let mut max_diff: f32 = 0.0;
        for (i, &t) in sample.strategy.iter().enumerate() {
            let p = if i < pred_vec.len() { pred_vec[i] } else { 0.0 };
            max_diff = max_diff.max((t - p).abs());
        }
        total_err += max_diff;
        count += 1;
    }

    let avg_err = total_err / count as f32;
    println!("  Average max error: {:.1}%", avg_err * 100.0);

    // Summary
    println!("\n========== SUMMARY ==========");
    println!("DCFR file size:      5.2 GB");
    println!("Samples collected:   {}", samples.len());
    println!("Training time:       {:.1}s", train_start.elapsed().as_secs_f64());
    println!("Inference speed:     {:.0} µs/query", per_query);
    println!("Average error:       {:.1}%", avg_err * 100.0);
    println!("Total time:          {:.1}s", total_start.elapsed().as_secs_f64());
    println!("=============================");

    if avg_err < 0.20 {
        println!("\n✓ SUCCESS: Deep network learned from DCFR!");
    } else {
        println!("\n⚠ Needs more training (error > 20%)");
    }

    // Save network weights
    var_map.save("out/deep_medium_weights.safetensors")?;
    println!("\nNetwork weights saved to: out/deep_medium_weights.safetensors");

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_learn_from_dcfr --release --features \"deep bincode zstd\"");
}
