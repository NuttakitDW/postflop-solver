//! Deep Q-Network Training - Predict EV for any (state, action) pair
//!
//! This allows querying strategy with ANY bet size at runtime!
//!
//! Run: cargo run --example deep_q_train --release --features "deep bincode zstd"

use postflop_solver::*;
use std::time::Instant;

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

#[cfg(feature = "deep")]
struct Sample {
    features: Vec<f32>,  // state + action encoding
    q_value: f32,        // EV after taking this action
}

#[cfg(feature = "deep")]
struct QNetwork {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    out: Linear,
}

#[cfg(feature = "deep")]
impl QNetwork {
    fn new(vs: VarBuilder, input: usize) -> candle_core::Result<Self> {
        Ok(Self {
            l1: linear(input, 256, vs.pp("l1"))?,
            l2: linear(256, 128, vs.pp("l2"))?,
            l3: linear(128, 64, vs.pp("l3"))?,
            out: linear(64, 1, vs.pp("out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        self.out.forward(&x)
    }
}

/// Encode state + action into features
/// The key insight: action is encoded as bet_size (0.0 = check, 0.2 = bet 20%, etc.)
#[cfg(feature = "deep")]
fn encode_state_action(
    flop: [u8; 3],
    hole: (u8, u8),
    player: usize,
    bet_size: f32,  // 0.0 = check, 0.2 = bet 20%, etc.
) -> Vec<f32> {
    // 265 features: board(156) + hole(104) + player(1) + street(1) + pot(1) + bet_size(1) + is_check(1)
    let mut f = vec![0.0f32; 265];

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

    // Pot ratio (fixed at root)
    f[262] = 1.0;

    // Bet size (normalized: 0.0 = check, 0.2 = 20%, 1.0 = pot, etc.)
    f[263] = bet_size;

    // Is check (binary flag)
    f[264] = if bet_size == 0.0 { 1.0 } else { 0.0 };

    f
}

/// Convert Action to bet size ratio
#[cfg(feature = "deep")]
fn action_to_bet_size(action: &Action) -> f32 {
    match action {
        Action::Check => 0.0,
        Action::Bet(size) => *size as f32 / 100.0,  // e.g., Bet(20) → 0.20
        Action::Raise(size) => *size as f32 / 100.0,
        Action::AllIn(_) => 2.0,  // All-in represented as 200%
        _ => 0.0,
    }
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Deep Q-Network Training ===");
    println!("Train on (state, action) → EV, then query ANY bet size!\n");
    let total_start = Instant::now();

    // Load DCFR solution
    println!("Step 1: Loading DCFR solution...");
    let (mut game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;
    game.cache_normalized_weights();

    let flop = game.card_config().flop;
    println!("  Board: {:?}", flop);

    // Collect Q-value samples for OOP
    println!("\nStep 2: Collecting OOP Q-value samples...");
    let mut samples: Vec<Sample> = Vec::new();

    let oop_hands = game.private_cards(0).to_vec();
    let oop_actions = game.available_actions();

    println!("  OOP actions: {:?}", oop_actions);

    // For each action, navigate to child, get EV, then go back
    for (action_idx, action) in oop_actions.iter().enumerate() {
        let bet_size = action_to_bet_size(action);

        // Navigate to this action's child
        game.play(action_idx);
        game.cache_normalized_weights();

        // Get EVs at the child state (these are the true Q-values)
        let child_evs = game.expected_values(0);

        for (hi, &(c1, c2)) in oop_hands.iter().enumerate() {
            if flop.contains(&c1) || flop.contains(&c2) { continue; }

            samples.push(Sample {
                features: encode_state_action(flop, (c1, c2), 0, bet_size),
                q_value: child_evs[hi],
            });
        }

        // Go back to root for next action
        game.back_to_root();
        game.cache_normalized_weights();
    }
    println!("  Collected {} OOP samples", samples.len());

    // Collect Q-value samples for IP (after OOP checks)
    println!("\nStep 3: Collecting IP Q-value samples...");
    game.play(0); // OOP checks
    game.cache_normalized_weights();

    let ip_hands = game.private_cards(1).to_vec();
    let ip_actions = game.available_actions();

    println!("  IP actions: {:?}", ip_actions);

    let ip_start = samples.len();
    for (action_idx, action) in ip_actions.iter().enumerate() {
        let bet_size = action_to_bet_size(action);

        // Navigate to this action's child
        game.play(action_idx);
        game.cache_normalized_weights();

        // Get EVs at child (true Q-values for IP)
        let child_evs = game.expected_values(1);

        for (hi, &(c1, c2)) in ip_hands.iter().enumerate() {
            if flop.contains(&c1) || flop.contains(&c2) { continue; }

            samples.push(Sample {
                features: encode_state_action(flop, (c1, c2), 1, bet_size),
                q_value: child_evs[hi],
            });
        }

        // Go back to OOP-check state for next IP action
        game.back_to_root();
        game.play(0); // OOP checks again
        game.cache_normalized_weights();
    }
    println!("  Collected {} IP samples", samples.len() - ip_start);
    println!("\n  Total samples: {}", samples.len());

    // Train Q-network
    println!("\nStep 4: Training Q-network...");
    let train_start = Instant::now();

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = QNetwork::new(vs, 265)?;

    let mut opt = candle_nn::AdamW::new(
        var_map.all_vars(),
        candle_nn::ParamsAdamW { lr: 0.001, ..Default::default() },
    )?;

    let epochs = 300;

    for epoch in 0..epochs {
        let bs = samples.len();
        let feats: Vec<f32> = samples.iter().flat_map(|s| s.features.iter().copied()).collect();
        let targets: Vec<f32> = samples.iter().map(|s| s.q_value).collect();

        let x = Tensor::from_vec(feats, (bs, 265), &device)?;
        let y = Tensor::from_vec(targets, (bs, 1), &device)?;

        let pred = net.forward(&x)?;
        let diff = (&pred - &y)?;
        let loss = (&diff * &diff)?.mean_all()?;

        opt.backward_step(&loss)?;

        if epoch % 50 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: MSE = {:.6}", epoch, loss.to_scalar::<f32>()?);
        }
    }
    println!("  Training time: {:.1}s", train_start.elapsed().as_secs_f64());

    // Test: predict Q-values for different bet sizes
    println!("\nStep 5: Testing Q-value predictions...");

    let test_hand = oop_hands.iter()
        .find(|&&(c1, c2)| !flop.contains(&c1) && !flop.contains(&c2))
        .unwrap();

    println!("  Test hand: ({}, {})", test_hand.0, test_hand.1);
    println!("\n  OOP Q-values for different bet sizes:");

    for bet_pct in [0, 10, 20, 33, 50, 75, 100] {
        let bet_size = bet_pct as f32 / 100.0;
        let features = encode_state_action(flop, *test_hand, 0, bet_size);
        let x = Tensor::from_vec(features, (1, 265), &device)?;
        let pred = net.forward(&x)?;
        let q = pred.flatten_all()?.to_vec1::<f32>()?[0];

        let action_name = if bet_pct == 0 { "Check".to_string() } else { format!("Bet {}%", bet_pct) };
        println!("    {:12} → Q = {:.2}", action_name, q);
    }

    // Save
    var_map.save("out/deep_q_weights.safetensors")?;
    println!("\nWeights saved to: out/deep_q_weights.safetensors");
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());

    println!("\n=== Next: Use deep_q_cli to query ANY bet size! ===");

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_q_train --release --features \"deep bincode zstd\"");
}
