//! Test the fixed model - try loading weights AFTER creating network
//!
//! Run: cargo run --example test_fixed_model2 --release --features "deep bincode zstd"

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

    fn predict(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        let logits = self.out.forward(&x)?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }
}

#[cfg(feature = "deep")]
fn encode(flop: [u8; 3], hole: (u8, u8)) -> Vec<f32> {
    let mut f = vec![0.0f32; 369];

    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }

    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[260 + c1 as usize] = 1.0; }
    if c2 < 52 { f[312 + c2 as usize] = 1.0; }

    f[364] = 1.0; // pot_ratio
    f[365] = 0.0; // stack_ratio
    f[366] = 1.0; // street = flop

    f
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Test Fixed Model v2 ===\n");

    let (game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;
    let flop = game.card_config().flop;
    let hands = game.private_cards(0);

    println!("Flop: {:?}", flop);
    println!("OOP hands: {}\n", hands.len());

    let device = Device::Cpu;

    // Method 1: Create network first, then load weights
    println!("Method 1: Create network, then load weights...");
    let mut var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 369, 2)?;

    // Try to load weights after network creation
    match var_map.load("out/deep_fixed_weights.safetensors") {
        Ok(_) => println!("  Loaded weights into existing VarMap"),
        Err(e) => println!("  Failed to load: {}", e),
    }

    // Test one hand
    let test_hand = hands[0];
    let features = encode(flop, test_hand);
    let x = Tensor::from_vec(features, (1, 369), &device)?;
    let pred = net.predict(&x)?;
    let probs: Vec<f32> = pred.flatten_all()?.to_vec1()?;
    println!("  Test hand {:?}: Check={:.1}%, Bet={:.1}%\n", test_hand, probs[0]*100.0, probs[1]*100.0);

    // Method 2: Load weights into new VarMap, then create network
    println!("Method 2: Load weights first, then create network...");
    let mut var_map2 = VarMap::new();
    var_map2.load("out/deep_fixed_weights.safetensors")?;

    // Check what's in the VarMap
    let data = var_map2.data().lock().unwrap();
    println!("  VarMap has {} tensors", data.len());
    for (name, _) in data.iter().take(4) {
        println!("    - {}", name);
    }
    drop(data);

    let vs2 = VarBuilder::from_varmap(&var_map2, DType::F32, &device);
    let net2 = StrategyNet::new(vs2, 369, 2)?;

    let pred2 = net2.predict(&x)?;
    let probs2: Vec<f32> = pred2.flatten_all()?.to_vec1()?;
    println!("  Test hand {:?}: Check={:.1}%, Bet={:.1}%\n", test_hand, probs2[0]*100.0, probs2[1]*100.0);

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with --features \"deep bincode zstd\"");
}
