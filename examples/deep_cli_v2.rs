//! Deep Strategy CLI v2 - Supports both OOP and IP
//!
//! Run: cargo run --example deep_cli_v2 --release --features "deep"
//!
//! Examples:
//!   ./target/release/examples/deep_cli_v2 --range --position oop
//!   ./target/release/examples/deep_cli_v2 --range --position ip
//!   ./target/release/examples/deep_cli_v2 --range --position oop --hands "QQ+,AKs"
//!   ./target/release/examples/deep_cli_v2 --range --position ip --hands "22+,A2s+,KQs"

use postflop_solver::Range;

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
fn encode(flop: [u8; 3], hole: (u8, u8), player: usize) -> Vec<f32> {
    let mut f = vec![0.0f32; 264];

    // Board (3 * 52 = 156)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }

    // Hole cards (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[156 + c1 as usize] = 1.0; }
    if c2 < 52 { f[208 + c2 as usize] = 1.0; }

    // Pot/stack ratios (fixed for root)
    f[260] = 1.0;
    f[261] = 0.0;

    // Street (flop)
    f[262] = 1.0;

    // Player (0=OOP, 1=IP)
    f[263] = player as f32;

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

/// Get hands from a Range, filtered by flop blockers
#[cfg(feature = "deep")]
fn range_to_hands(range: &Range, flop: &[u8; 3]) -> Vec<(u8, u8)> {
    let mut hands = Vec::new();

    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            if flop.contains(&c1) || flop.contains(&c2) { continue; }

            let weight = range.get_weight_by_cards(c1, c2);
            if weight > 0.0 {
                hands.push((c1, c2));
            }
        }
    }

    hands
}

#[cfg(feature = "deep")]
fn show_range(net: &StrategyNet, device: &Device, flop: [u8; 3], position: &str, hands_filter: Option<&str>) {
    let player = if position == "oop" { 0 } else { 1 };

    let action_names: &[&str] = if position == "oop" {
        &["Check", "Bet 20"]
    } else {
        &["Check", "Bet 15", "Bet 34", "Bet 52", "Bet 76"]
    };

    // Get hands based on filter
    let hands: Vec<(u8, u8)> = if let Some(range_str) = hands_filter {
        match range_str.parse::<Range>() {
            Ok(range) => {
                println!("Using range: {}", range_str);
                range_to_hands(&range, &flop)
            }
            Err(e) => {
                println!("Invalid range '{}': {}", range_str, e);
                return;
            }
        }
    } else {
        // All valid hands
        let mut h = Vec::new();
        for c1 in 0..52u8 {
            for c2 in (c1 + 1)..52u8 {
                if !flop.contains(&c1) && !flop.contains(&c2) {
                    h.push((c1, c2));
                }
            }
        }
        h
    };

    println!("Analyzing {} {} hands on {} {} {}\n",
        hands.len(), position.to_uppercase(),
        card_to_string(flop[0]), card_to_string(flop[1]), card_to_string(flop[2]));

    let mut totals = vec![0.0f32; 5];

    for &hand in &hands {
        let features = encode(flop, hand, player);
        if let Ok(x) = Tensor::from_vec(features, (1, 264), device) {
            if let Ok(pred) = net.predict(&x) {
                if let Ok(probs) = pred.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                    for (i, p) in probs.iter().enumerate() {
                        totals[i] += p;
                    }
                }
            }
        }
    }

    let n = hands.len() as f32;

    println!("=== {} Range Summary ===\n", position.to_uppercase());
    println!("Average strategy:");
    for (i, name) in action_names.iter().enumerate() {
        let pct = totals[i] / n * 100.0;
        let bar_len = (pct / 2.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:12} {:5.1}% {}", name, pct, bar);
    }
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut position = "oop".to_string();
    let mut show_range_flag = false;
    let mut hands_filter: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--position" | "-p" => {
                if i + 1 < args.len() {
                    position = args[i + 1].clone();
                    i += 1;
                }
            }
            "--range" => {
                show_range_flag = true;
            }
            "--hands" | "-h" => {
                if i + 1 < args.len() {
                    hands_filter = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--help" => {
                println!("Deep Strategy CLI v2\n");
                println!("Usage:");
                println!("  deep_cli_v2 --range --position oop");
                println!("  deep_cli_v2 --range --position ip");
                println!("  deep_cli_v2 --range --position oop --hands \"AA,KK,QQ+\"");
                println!("  deep_cli_v2 --range --position ip --hands \"22+,AKs,AKo\"");
                println!("\nRange syntax:");
                println!("  AA      - pocket aces");
                println!("  QQ+     - QQ, KK, AA");
                println!("  AKs     - ace-king suited");
                println!("  AKo     - ace-king offsuit");
                println!("  ATs+    - ATs, AJs, AQs, AKs");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("Loading neural network...");
    let device = Device::Cpu;
    let mut var_map = VarMap::new();

    // Create network first, then load weights
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 264, 5)?;
    var_map.load("out/deep_both_weights.safetensors")?;

    // Flop: Td9d6h
    let flop = [
        parse_card("Td").unwrap(),
        parse_card("9d").unwrap(),
        parse_card("6h").unwrap(),
    ];

    println!("Loaded! Board: Td 9d 6h\n");

    if show_range_flag {
        show_range(&net, &device, flop, &position, hands_filter.as_deref());
    } else {
        println!("Use --range --position oop/ip to see strategy");
    }

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_cli_v2 --release --features \"deep\"");
}
