//! Test the fixed model on the same data it was trained on
//!
//! Run: cargo run --example test_fixed_model --release --features "deep bincode zstd"

use postflop_solver::*;

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

    fn predict(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(x)?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }
}

#[cfg(feature = "deep")]
fn encode(flop: [u8; 3], hole: (u8, u8), pot_ratio: f32, stack_ratio: f32) -> Vec<f32> {
    let mut f = vec![0.0f32; 369];

    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }
    // turn/river = 255 (not dealt)

    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[260 + c1 as usize] = 1.0; }
    if c2 < 52 { f[312 + c2 as usize] = 1.0; }

    f[364] = pot_ratio;
    f[365] = stack_ratio;
    f[366] = 1.0; // street = flop

    f
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Test Fixed Model ===\n");

    // Load DCFR to get the same hands
    let (game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;
    let flop = game.card_config().flop;
    let hands = game.private_cards(0); // OOP hands

    println!("Flop: {:?}", flop);
    println!("OOP hands in range: {}\n", hands.len());

    // Load model
    let device = Device::Cpu;
    let mut var_map = VarMap::new();
    var_map.load("out/deep_fixed_weights.safetensors")?;
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 369, 2)?;
    println!("Model loaded!\n");

    // Test on hands from the DCFR solution (same as training)
    let pot_ratio = 1.0;
    let stack_ratio = 0.0;

    let mut check_total = 0.0f32;
    let mut bet_total = 0.0f32;
    let mut count = 0;

    for &(c1, c2) in hands.iter() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let features = encode(flop, (c1, c2), pot_ratio, stack_ratio);
        let x = Tensor::from_vec(features, (1, 369), &device)?;
        let pred = net.predict(&x)?;
        let probs: Vec<f32> = pred.flatten_all()?.to_vec1()?;

        check_total += probs[0];
        bet_total += probs[1];
        count += 1;
    }

    println!("Tested {} hands (same as training)", count);
    println!("Average prediction:");
    println!("  Check: {:.1}%", check_total / count as f32 * 100.0);
    println!("  Bet:   {:.1}%", bet_total / count as f32 * 100.0);

    if check_total / count as f32 > 0.95 {
        println!("\n✓ Model correctly predicts Check ~100%");
    } else {
        println!("\n✗ Model not predicting Check 100%");
    }

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with --features \"deep bincode zstd\"");
}
