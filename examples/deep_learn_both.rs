//! Deep PDCFR+ POC - Train on BOTH OOP and IP
//!
//! Run: cargo run --example deep_learn_both --release --features "deep bincode zstd"

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
    hole: (u8, u8),
    player: usize,  // 0=OOP, 1=IP
    pot_ratio: f32,
    stack_ratio: f32,
) -> Vec<f32> {
    // 264 features: board(156) + hole(104) + pot(1) + stack(1) + street(1) + player(1) = 264
    let mut f = vec![0.0f32; 264];

    // Board (3 * 52 = 156 for flop)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }

    // Hole cards (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[156 + c1 as usize] = 1.0; }
    if c2 < 52 { f[208 + c2 as usize] = 1.0; }

    // Pot and stack ratios
    f[260] = pot_ratio;
    f[261] = stack_ratio;

    // Street (flop = 1)
    f[262] = 1.0;

    // Player indicator (0=OOP, 1=IP)
    f[263] = player as f32;

    f
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Deep PDCFR+ POC - Train OOP + IP ===\n");
    let total_start = Instant::now();

    // Load DCFR solution
    println!("Step 1: Loading DCFR solution...");
    let (mut game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;
    println!("  Loaded!");

    let flop = game.card_config().flop;
    game.cache_normalized_weights();

    // Collect OOP samples (root node)
    println!("\nStep 2: Collecting OOP samples (root node)...");
    let mut oop_samples: Vec<Sample> = Vec::new();

    let oop_hands = game.private_cards(0).to_vec();
    let oop_strategy = game.strategy();
    let oop_num_hands = oop_hands.len();
    let oop_actions = game.available_actions();
    let oop_num_actions = oop_actions.len();

    println!("  OOP actions: {:?}", oop_actions);

    for (hi, &(c1, c2)) in oop_hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let mut strat = Vec::new();
        for a in 0..oop_num_actions {
            strat.push(oop_strategy[a * oop_num_hands + hi]);
        }

        let sum: f32 = strat.iter().sum();
        if sum > 0.5 {
            oop_samples.push(Sample {
                features: encode(flop, (c1, c2), 0, 1.0, 0.0),
                strategy: strat,
            });
        }
    }

    let oop_avg_check: f32 = oop_samples.iter().map(|s| s.strategy[0]).sum::<f32>() / oop_samples.len() as f32;
    println!("  Collected {} OOP samples", oop_samples.len());
    println!("  OOP average: Check={:.1}%", oop_avg_check * 100.0);

    // Navigate to IP node (after OOP checks)
    println!("\nStep 3: Collecting IP samples (after OOP check)...");
    game.play(0); // OOP checks (action 0)

    let mut ip_samples: Vec<Sample> = Vec::new();

    let ip_hands = game.private_cards(1).to_vec();
    let ip_strategy = game.strategy();
    let ip_num_hands = ip_hands.len();
    let ip_actions = game.available_actions();
    let ip_num_actions = ip_actions.len();

    println!("  IP actions: {:?}", ip_actions);

    for (hi, &(c1, c2)) in ip_hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let mut strat = Vec::new();
        for a in 0..ip_num_actions {
            strat.push(ip_strategy[a * ip_num_hands + hi]);
        }

        let sum: f32 = strat.iter().sum();
        if sum > 0.5 {
            // Pad to max actions (5 for IP)
            while strat.len() < 5 {
                strat.push(0.0);
            }
            ip_samples.push(Sample {
                features: encode(flop, (c1, c2), 1, 1.0, 0.0),
                strategy: strat,
            });
        }
    }

    println!("  Collected {} IP samples", ip_samples.len());
    if !ip_samples.is_empty() {
        let ip_avg: Vec<f32> = (0..ip_num_actions.min(5))
            .map(|a| ip_samples.iter().map(|s| s.strategy[a]).sum::<f32>() / ip_samples.len() as f32)
            .collect();
        println!("  IP average strategy:");
        for (i, action) in ip_actions.iter().enumerate() {
            println!("    {:?}: {:.1}%", action, ip_avg.get(i).unwrap_or(&0.0) * 100.0);
        }
    }

    // Combine samples - pad OOP to 5 actions
    let mut all_samples: Vec<Sample> = Vec::new();
    for mut s in oop_samples {
        while s.strategy.len() < 5 {
            s.strategy.push(0.0);
        }
        all_samples.push(s);
    }
    all_samples.extend(ip_samples);

    println!("\n  Total samples: {}", all_samples.len());

    // Train network with 5 outputs (max actions)
    println!("\nStep 4: Training neural network...");
    let train_start = Instant::now();

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 264, 5)?; // 264 features

    let mut opt = candle_nn::AdamW::new(
        var_map.all_vars(),
        candle_nn::ParamsAdamW { lr: 0.001, ..Default::default() },
    )?;

    let epochs = 100;

    for epoch in 0..epochs {
        let bs = all_samples.len();
        let feats: Vec<f32> = all_samples.iter().flat_map(|s| s.features.iter().copied()).collect();
        let targs: Vec<f32> = all_samples.iter().flat_map(|s| s.strategy.iter().copied()).collect();

        let feature_dim = all_samples[0].features.len();
        let x = Tensor::from_vec(feats, (bs, feature_dim), &device)?;
        let y = Tensor::from_vec(targs, (bs, 5), &device)?;

        let logits = net.forward(&x)?;
        let log_p = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
        let loss = (&y * &log_p)?.neg()?.sum_all()?;
        let loss = (&loss / bs as f64)?;

        opt.backward_step(&loss)?;

        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: loss = {:.4}", epoch, loss.to_scalar::<f32>()?);
        }
    }
    println!("  Training time: {:.1}s", train_start.elapsed().as_secs_f64());

    // Test
    println!("\nStep 5: Testing...");

    // Test OOP
    let oop_test: Vec<_> = all_samples.iter().filter(|s| s.features[263] == 0.0).collect();
    let mut oop_pred_check = 0.0f32;
    for s in &oop_test {
        let x = Tensor::from_vec(s.features.clone(), (1, s.features.len()), &device)?;
        let pred = net.predict(&x)?;
        let probs: Vec<f32> = pred.flatten_all()?.to_vec1()?;
        oop_pred_check += probs[0];
    }
    println!("  OOP predicted Check: {:.1}%", oop_pred_check / oop_test.len() as f32 * 100.0);

    // Test IP
    let ip_test: Vec<_> = all_samples.iter().filter(|s| s.features[263] == 1.0).collect();
    if !ip_test.is_empty() {
        let mut ip_preds = vec![0.0f32; 5];
        for s in &ip_test {
            let x = Tensor::from_vec(s.features.clone(), (1, s.features.len()), &device)?;
            let pred = net.predict(&x)?;
            let probs: Vec<f32> = pred.flatten_all()?.to_vec1()?;
            for i in 0..5 {
                ip_preds[i] += probs[i];
            }
        }
        println!("  IP predicted:");
        for (i, action) in ip_actions.iter().enumerate() {
            println!("    {:?}: {:.1}%", action, ip_preds[i] / ip_test.len() as f32 * 100.0);
        }
    }

    // Save
    var_map.save("out/deep_both_weights.safetensors")?;
    println!("\nWeights saved to: out/deep_both_weights.safetensors");
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_learn_both --release --features \"deep bincode zstd\"");
}
