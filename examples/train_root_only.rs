//! Train ONLY on root node samples to verify pipeline works
//!
//! Run: cargo run --example train_root_only --release --features "deep bincode zstd"

use postflop_solver::*;

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

#[cfg(feature = "deep")]
struct StrategyNet {
    l1: Linear,
    l2: Linear,
    out: Linear,
}

#[cfg(feature = "deep")]
impl StrategyNet {
    fn new(vs: VarBuilder, input: usize, output: usize) -> candle_core::Result<Self> {
        Ok(Self {
            l1: linear(input, 64, vs.pp("l1"))?,
            l2: linear(64, 32, vs.pp("l2"))?,
            out: linear(32, output, vs.pp("out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        self.out.forward(&x)
    }

    fn predict(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(x)?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }
}

#[cfg(feature = "deep")]
fn encode(flop: [u8; 3], hole: (u8, u8)) -> Vec<f32> {
    let mut f = vec![0.0f32; 260]; // Just board + hole cards

    // Board one-hot (3 * 52 = 156)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }

    // Hole cards one-hot (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[156 + c1 as usize] = 1.0; }
    if c2 < 52 { f[208 + c2 as usize] = 1.0; }

    f
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Train ONLY on Root Node ===\n");

    // Load DCFR
    println!("Loading DCFR solution...");
    let (game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;

    let flop = game.card_config().flop;
    let hands = game.private_cards(0);
    let root = game.root();
    let strategy = root.strategy();
    let num_hands = hands.len();

    // Collect ROOT ONLY samples
    println!("Collecting root node samples...");
    let mut samples: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    for (hi, &(c1, c2)) in hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let check = strategy[0 * num_hands + hi];
        let bet = strategy[1 * num_hands + hi];
        let sum = check + bet;

        if sum > 0.01 {
            let features = encode(flop, (c1, c2));
            let strat = vec![check / sum, bet / sum];
            samples.push((features, strat));
        }
    }

    println!("Collected {} root node samples", samples.len());

    // Check what strategy we're training on
    let avg_check: f32 = samples.iter().map(|(_, s)| s[0]).sum::<f32>() / samples.len() as f32;
    let avg_bet: f32 = samples.iter().map(|(_, s)| s[1]).sum::<f32>() / samples.len() as f32;
    println!("Target strategy: Check {:.1}%, Bet {:.1}%\n", avg_check * 100.0, avg_bet * 100.0);

    // Train
    println!("Training neural network...");
    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 260, 2)?;

    let mut opt = candle_nn::AdamW::new(
        var_map.all_vars(),
        candle_nn::ParamsAdamW { lr: 0.01, ..Default::default() },
    )?;

    for epoch in 0..200 {
        let bs = samples.len();
        let feats: Vec<f32> = samples.iter().flat_map(|(f, _)| f.iter().copied()).collect();
        let targs: Vec<f32> = samples.iter().flat_map(|(_, s)| s.iter().copied()).collect();

        let x = Tensor::from_vec(feats, (bs, 260), &device)?;
        let y = Tensor::from_vec(targs, (bs, 2), &device)?;

        let logits = net.forward(&x)?;
        let log_p = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
        let loss = (&y * &log_p)?.neg()?.sum_all()?;
        let loss = (&loss / bs as f64)?;

        opt.backward_step(&loss)?;

        if epoch % 50 == 0 || epoch == 199 {
            let loss_val = loss.to_scalar::<f32>()?;
            println!("  Epoch {}: loss = {:.4}", epoch, loss_val);
        }
    }

    // Test
    println!("\n=== Testing ===");
    let mut pred_check_total = 0.0f32;
    let mut pred_bet_total = 0.0f32;

    for (features, _) in &samples {
        let x = Tensor::from_vec(features.clone(), (1, 260), &device)?;
        let pred = net.predict(&x)?;
        let probs: Vec<f32> = pred.flatten_all()?.to_vec1()?;
        pred_check_total += probs[0];
        pred_bet_total += probs[1];
    }

    let n = samples.len() as f32;
    println!("Target:     Check {:.1}%, Bet {:.1}%", avg_check * 100.0, avg_bet * 100.0);
    println!("Predicted:  Check {:.1}%, Bet {:.1}%", pred_check_total / n * 100.0, pred_bet_total / n * 100.0);

    let error = ((avg_check - pred_check_total / n).abs() + (avg_bet - pred_bet_total / n).abs()) * 100.0;
    println!("\nTotal error: {:.1}%", error);

    if error < 5.0 {
        println!("\n✓ PIPELINE WORKS! Network learned root node strategy.");
    } else {
        println!("\n✗ Still has error - pipeline issue?");
    }

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with --features \"deep bincode zstd\"");
}
