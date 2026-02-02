//! Deep Value-Based Training - Train on EVs instead of strategies
//!
//! This allows changing bet sizes at runtime without retraining!
//!
//! Run: cargo run --example deep_value_train --release --features "deep bincode zstd"

use postflop_solver::*;
use std::time::Instant;

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

#[cfg(feature = "deep")]
struct Sample {
    features: Vec<f32>,
    ev: f32,  // Expected value for this hand
}

#[cfg(feature = "deep")]
struct ValueNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    out: Linear,
}

#[cfg(feature = "deep")]
impl ValueNet {
    fn new(vs: VarBuilder, input: usize) -> candle_core::Result<Self> {
        Ok(Self {
            l1: linear(input, 256, vs.pp("l1"))?,
            l2: linear(256, 128, vs.pp("l2"))?,
            l3: linear(128, 64, vs.pp("l3"))?,
            out: linear(64, 1, vs.pp("out"))?,  // Single value output
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        self.out.forward(&x)
    }
}

#[cfg(feature = "deep")]
fn encode(flop: [u8; 3], hole: (u8, u8), player: usize, action_taken: usize, num_actions: usize) -> Vec<f32> {
    // 264 + action encoding
    let mut f = vec![0.0f32; 270];

    // Board (3 * 52 = 156)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }

    // Hole cards (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[156 + c1 as usize] = 1.0; }
    if c2 < 52 { f[208 + c2 as usize] = 1.0; }

    // Player (0=OOP, 1=IP)
    f[260] = player as f32;

    // Street (flop)
    f[261] = 1.0;

    // Action taken (one-hot, up to 6 actions)
    if action_taken < 6 {
        f[262 + action_taken] = 1.0;
    }

    // Number of actions available (normalized)
    f[268] = num_actions as f32 / 6.0;

    f
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Deep Value-Based Training ===\n");
    let total_start = Instant::now();

    // Load DCFR solution
    println!("Step 1: Loading DCFR solution...");
    let (mut game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;
    game.cache_normalized_weights();

    let flop = game.card_config().flop;
    println!("  Board: {:?}", flop);

    // Collect OOP samples with EVs
    println!("\nStep 2: Collecting OOP value samples...");
    let mut samples: Vec<Sample> = Vec::new();

    let oop_hands = game.private_cards(0).to_vec();
    let oop_evs = game.expected_values(0);
    let oop_strategy = game.strategy();
    let oop_actions = game.available_actions();
    let num_oop_hands = oop_hands.len();
    let num_oop_actions = oop_actions.len();

    println!("  OOP actions: {:?}", oop_actions);
    println!("  OOP EVs length: {}, hands: {}", oop_evs.len(), num_oop_hands);

    // For each hand, we have an EV
    for (hi, &(c1, c2)) in oop_hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let ev = oop_evs[hi];

        // Sample for the current state (before action)
        samples.push(Sample {
            features: encode(flop, (c1, c2), 0, 0, num_oop_actions), // action 0 = check
            ev,
        });
    }

    println!("  Collected {} OOP samples", samples.len());
    let oop_avg_ev: f32 = samples.iter().map(|s| s.ev).sum::<f32>() / samples.len() as f32;
    println!("  OOP average EV: {:.2}", oop_avg_ev);

    // Navigate to IP node (after OOP checks)
    println!("\nStep 3: Collecting IP value samples...");
    game.play(0); // OOP checks
    game.cache_normalized_weights();

    let ip_hands = game.private_cards(1).to_vec();
    let ip_evs = game.expected_values(1);
    let ip_actions = game.available_actions();
    let num_ip_hands = ip_hands.len();
    let num_ip_actions = ip_actions.len();

    println!("  IP actions: {:?}", ip_actions);
    println!("  IP EVs length: {}, hands: {}", ip_evs.len(), num_ip_hands);

    let ip_start = samples.len();
    for (hi, &(c1, c2)) in ip_hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let ev = ip_evs[hi];

        samples.push(Sample {
            features: encode(flop, (c1, c2), 1, 0, num_ip_actions),
            ev,
        });
    }

    println!("  Collected {} IP samples", samples.len() - ip_start);
    let ip_avg_ev: f32 = samples[ip_start..].iter().map(|s| s.ev).sum::<f32>() / (samples.len() - ip_start) as f32;
    println!("  IP average EV: {:.2}", ip_avg_ev);

    println!("\n  Total samples: {}", samples.len());

    // Train
    println!("\nStep 4: Training value network...");
    let train_start = Instant::now();

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = ValueNet::new(vs, 270)?;

    let mut opt = candle_nn::AdamW::new(
        var_map.all_vars(),
        candle_nn::ParamsAdamW { lr: 0.001, ..Default::default() },
    )?;

    let epochs = 200;

    for epoch in 0..epochs {
        let bs = samples.len();
        let feats: Vec<f32> = samples.iter().flat_map(|s| s.features.iter().copied()).collect();
        let targets: Vec<f32> = samples.iter().map(|s| s.ev).collect();

        let x = Tensor::from_vec(feats, (bs, 270), &device)?;
        let y = Tensor::from_vec(targets, (bs, 1), &device)?;

        let pred = net.forward(&x)?;
        let diff = (&pred - &y)?;
        let loss = (&diff * &diff)?.mean_all()?;  // MSE loss

        opt.backward_step(&loss)?;

        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: MSE = {:.6}", epoch, loss.to_scalar::<f32>()?);
        }
    }
    println!("  Training time: {:.1}s", train_start.elapsed().as_secs_f64());

    // Test
    println!("\nStep 5: Testing...");

    // Test OOP
    let oop_samples: Vec<_> = samples.iter().filter(|s| s.features[260] == 0.0).collect();
    let mut oop_pred_sum = 0.0f32;
    let mut oop_target_sum = 0.0f32;
    for s in &oop_samples {
        let x = Tensor::from_vec(s.features.clone(), (1, 270), &device)?;
        let pred = net.forward(&x)?;
        oop_pred_sum += pred.flatten_all()?.to_vec1::<f32>()?[0];
        oop_target_sum += s.ev;
    }
    println!("  OOP - Target avg EV: {:.2}, Predicted avg EV: {:.2}",
        oop_target_sum / oop_samples.len() as f32,
        oop_pred_sum / oop_samples.len() as f32);

    // Test IP
    let ip_samples: Vec<_> = samples.iter().filter(|s| s.features[260] == 1.0).collect();
    let mut ip_pred_sum = 0.0f32;
    let mut ip_target_sum = 0.0f32;
    for s in &ip_samples {
        let x = Tensor::from_vec(s.features.clone(), (1, 270), &device)?;
        let pred = net.forward(&x)?;
        ip_pred_sum += pred.flatten_all()?.to_vec1::<f32>()?[0];
        ip_target_sum += s.ev;
    }
    println!("  IP - Target avg EV: {:.2}, Predicted avg EV: {:.2}",
        ip_target_sum / ip_samples.len() as f32,
        ip_pred_sum / ip_samples.len() as f32);

    // Save
    var_map.save("out/deep_value_weights.safetensors")?;
    println!("\nWeights saved to: out/deep_value_weights.safetensors");
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());

    println!("\n=== Summary ===");
    println!("Trained value network on {} samples", samples.len());
    println!("This network predicts EV, allowing runtime strategy computation");

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_value_train --release --features \"deep bincode zstd\"");
}
