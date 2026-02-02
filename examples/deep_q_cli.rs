//! Deep Q-Network CLI - Query strategy with ANY bet sizes!
//!
//! Run: cargo run --example deep_q_cli --release --features "deep"
//!
//! Examples:
//!   # Use default bet sizes (check, 33%, 67%, 100%)
//!   ./target/release/examples/deep_q_cli --position oop
//!
//!   # Specify custom bet sizes
//!   ./target/release/examples/deep_q_cli --position ip --bets 0,25,50,100,150
//!
//!   # Try a bet size that wasn't in training!
//!   ./target/release/examples/deep_q_cli --position oop --bets 0,10,20,30,40,50

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};

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

#[cfg(feature = "deep")]
fn encode_state_action(
    flop: [u8; 3],
    hole: (u8, u8),
    player: usize,
    bet_size: f32,
) -> Vec<f32> {
    let mut f = vec![0.0f32; 265];

    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }

    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[156 + c1 as usize] = 1.0; }
    if c2 < 52 { f[208 + c2 as usize] = 1.0; }

    f[260] = player as f32;
    f[261] = 1.0;
    f[262] = 1.0;
    f[263] = bet_size;
    f[264] = if bet_size == 0.0 { 1.0 } else { 0.0 };

    f
}

#[cfg(feature = "deep")]
fn parse_card(s: &str) -> Option<u8> {
    let s = s.trim();
    if s.len() < 2 { return None; }
    let chars: Vec<char> = s.chars().collect();
    let rank = match chars[0].to_ascii_uppercase() {
        '2' => 0, '3' => 1, '4' => 2, '5' => 3, '6' => 4,
        '7' => 5, '8' => 6, '9' => 7, 'T' => 8, 'J' => 9,
        'Q' => 10, 'K' => 11, 'A' => 12, _ => return None,
    };
    let suit = match chars[1].to_ascii_lowercase() {
        'c' => 0, 'd' => 1, 'h' => 2, 's' => 3, _ => return None,
    };
    Some(rank * 4 + suit)
}

#[cfg(feature = "deep")]
fn card_to_string(card: u8) -> String {
    if card >= 52 { return "?".to_string(); }
    let rank = card / 4;
    let suit = card % 4;
    let r = match rank { 0=>'2', 1=>'3', 2=>'4', 3=>'5', 4=>'6', 5=>'7', 6=>'8', 7=>'9', 8=>'T', 9=>'J', 10=>'Q', 11=>'K', 12=>'A', _=>'?' };
    let s = match suit { 0=>'c', 1=>'d', 2=>'h', 3=>'s', _=>'?' };
    format!("{}{}", r, s)
}

/// Convert Q-values to strategy using regret matching
#[cfg(feature = "deep")]
fn q_to_strategy(q_values: &[f32]) -> Vec<f32> {
    // Find max Q
    let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute advantages (regrets)
    let advantages: Vec<f32> = q_values.iter().map(|q| (q - max_q).max(0.0)).collect();

    // Regret matching: strategy proportional to positive regrets
    let sum: f32 = advantages.iter().sum();
    if sum > 0.0 {
        advantages.iter().map(|a| a / sum).collect()
    } else {
        // All equal, uniform
        vec![1.0 / q_values.len() as f32; q_values.len()]
    }
}

/// Alternative: softmax strategy (smoother)
#[cfg(feature = "deep")]
fn q_to_strategy_softmax(q_values: &[f32], temperature: f32) -> Vec<f32> {
    let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = q_values.iter().map(|q| ((q - max_q) / temperature).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.iter().map(|e| e / sum).collect()
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut position = "oop".to_string();
    let mut bet_sizes: Vec<u32> = vec![0, 33, 67, 100];  // Default: check, 33%, 67%, pot

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--position" | "-p" => {
                if i + 1 < args.len() {
                    position = args[i + 1].clone();
                    i += 1;
                }
            }
            "--bets" | "-b" => {
                if i + 1 < args.len() {
                    bet_sizes = args[i + 1]
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    i += 1;
                }
            }
            "--help" => {
                println!("Deep Q-Network CLI - Query ANY bet sizes!\n");
                println!("Usage:");
                println!("  deep_q_cli --position oop|ip --bets 0,25,50,100\n");
                println!("Examples:");
                println!("  deep_q_cli --position oop --bets 0,20");
                println!("  deep_q_cli --position ip --bets 0,15,34,52,76");
                println!("  deep_q_cli --position ip --bets 0,10,20,30,40,50,60,70,80,90,100");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("Loading Q-network...");
    let device = Device::Cpu;
    let mut var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = QNetwork::new(vs, 265)?;
    var_map.load("out/deep_q_weights.safetensors")?;

    // Flop: Td9d6h
    let flop = [
        parse_card("Td").unwrap(),
        parse_card("9d").unwrap(),
        parse_card("6h").unwrap(),
    ];

    let player = if position == "oop" { 0 } else { 1 };

    println!("Board: Td 9d 6h");
    println!("Position: {}", position.to_uppercase());
    println!("Bet sizes: {:?}\n", bet_sizes);

    // Action names
    let action_names: Vec<String> = bet_sizes.iter()
        .map(|&b| if b == 0 { "Check".to_string() } else { format!("Bet {}%", b) })
        .collect();

    // Get all valid hands
    let mut hands = Vec::new();
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            if !flop.contains(&c1) && !flop.contains(&c2) {
                hands.push((c1, c2));
            }
        }
    }

    println!("Analyzing {} hands...\n", hands.len());

    // Aggregate strategy across all hands
    let mut totals = vec![0.0f32; bet_sizes.len()];

    for &hand in &hands {
        // Query Q-value for each bet size
        let mut q_values = Vec::new();
        for &bet_pct in &bet_sizes {
            let bet_size = bet_pct as f32 / 100.0;
            let features = encode_state_action(flop, hand, player, bet_size);
            let x = Tensor::from_vec(features, (1, 265), &device)?;
            let pred = net.forward(&x)?;
            let q = pred.flatten_all()?.to_vec1::<f32>()?[0];
            q_values.push(q);
        }

        // Convert Q-values to strategy
        let strategy = q_to_strategy_softmax(&q_values, 5.0);

        for (i, prob) in strategy.iter().enumerate() {
            totals[i] += prob;
        }
    }

    let n = hands.len() as f32;

    println!("=== {} Range Summary ===\n", position.to_uppercase());
    println!("Average strategy (with custom bet sizes):");
    for (i, name) in action_names.iter().enumerate() {
        let pct = totals[i] / n * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len.min(50));
        println!("  {:12} {:5.1}% {}", name, pct, bar);
    }

    println!("\n=== Key Insight ===");
    println!("These bet sizes were computed at RUNTIME!");
    println!("The model was trained on: Check, Bet 20% (OOP) / Check, Bet 15%, 34%, 52%, 76% (IP)");
    println!("But you can query ANY bet size without retraining!");

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_q_cli --release --features \"deep\"");
}
