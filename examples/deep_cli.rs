//! Deep Strategy CLI - Query neural network for poker strategies
//!
//! Uses trained weights to provide instant strategy lookups for medium.json settings.
//!
//! Run: cargo run --example deep_cli --release --features "deep"
//!
//! Examples:
//!   cargo run --example deep_cli --release --features "deep" -- --hand AhKs --position oop
//!   cargo run --example deep_cli --release --features "deep" -- --hand 9d9c --position ip --turn 2h
//!   cargo run --example deep_cli --release --features "deep" -- --interactive

use postflop_solver::*;
use std::io::{self, Write};

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

    // Board cards one-hot (5 * 52 = 260)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }
    if turn < 52 { f[156 + turn as usize] = 1.0; }
    if river < 52 { f[208 + river as usize] = 1.0; }

    // Hole cards one-hot (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[260 + c1 as usize] = 1.0; }
    if c2 < 52 { f[312 + c2 as usize] = 1.0; }

    // Pot and stack ratios
    f[364] = pot_ratio;
    f[365] = stack_ratio;

    // Street one-hot
    if street < 3 { f[366 + street] = 1.0; }

    f
}

#[cfg(feature = "deep")]
fn parse_card(s: &str) -> Option<u8> {
    let s = s.trim();
    if s.len() < 2 { return None; }

    let chars: Vec<char> = s.chars().collect();
    let rank_char = chars[0].to_ascii_uppercase();
    let suit_char = chars[1].to_ascii_lowercase();

    let rank = match rank_char {
        '2' => 0, '3' => 1, '4' => 2, '5' => 3, '6' => 4,
        '7' => 5, '8' => 6, '9' => 7, 'T' => 8, 'J' => 9,
        'Q' => 10, 'K' => 11, 'A' => 12,
        _ => return None,
    };

    let suit = match suit_char {
        'c' => 0, 'd' => 1, 'h' => 2, 's' => 3,
        _ => return None,
    };

    Some(rank * 4 + suit)
}

#[cfg(feature = "deep")]
fn card_to_string(card: u8) -> String {
    if card >= 52 { return "?".to_string(); }
    let rank = card / 4;
    let suit = card % 4;
    let rank_char = match rank {
        0 => '2', 1 => '3', 2 => '4', 3 => '5', 4 => '6',
        5 => '7', 6 => '8', 7 => '9', 8 => 'T', 9 => 'J',
        10 => 'Q', 11 => 'K', 12 => 'A',
        _ => '?',
    };
    let suit_char = match suit {
        0 => 'c', 1 => 'd', 2 => 'h', 3 => 's',
        _ => '?',
    };
    format!("{}{}", rank_char, suit_char)
}

#[cfg(feature = "deep")]
fn parse_hand(s: &str) -> Option<(u8, u8)> {
    let s = s.trim();
    if s.len() < 4 { return None; }

    let c1 = parse_card(&s[0..2])?;
    let c2 = parse_card(&s[2..4])?;
    Some((c1, c2))
}

#[cfg(feature = "deep")]
struct DeepSolver {
    net: StrategyNet,
    device: Device,
    flop: [u8; 3],
    actions: Vec<String>,
}

#[cfg(feature = "deep")]
impl DeepSolver {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut var_map = VarMap::new();

        // IMPORTANT: Create network FIRST (creates variables in VarMap)
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let net = StrategyNet::new(vs, 369, 2)?; // 2 actions for OOP root

        // THEN load weights (replaces values in existing variables)
        var_map.load("out/deep_fixed_weights.safetensors")?;

        // Flop from medium.json: Td9d6h
        let flop = [
            parse_card("Td").unwrap(),
            parse_card("9d").unwrap(),
            parse_card("6h").unwrap(),
        ];

        // Actions for medium.json flop (OOP first to act)
        // OOP: Check, Bet 33%
        // IP: Check, Bet 25%, 55%, 85%, 125%
        let actions = vec![
            "Check".to_string(),
            "Bet 33%".to_string(),
            "Bet 25%".to_string(),
            "Bet 55%".to_string(),
            "Bet 85%".to_string(),
        ];

        Ok(Self { net, device, flop, actions })
    }

    fn get_strategy(
        &self,
        hand: (u8, u8),
        position: &str,
        turn: Option<u8>,
        river: Option<u8>,
        pot_ratio: f32,
        stack_ratio: f32,
    ) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
        let t = turn.unwrap_or(255);
        let r = river.unwrap_or(255);

        let street = if t == 255 { 0 } else if r == 255 { 1 } else { 2 };

        let features = encode(self.flop, t, r, hand, pot_ratio, stack_ratio, street);
        let x = Tensor::from_vec(features, (1, 369), &self.device)?;
        let pred = self.net.predict(&x)?;
        let probs: Vec<f32> = pred.flatten_all()?.to_vec1()?;

        // Get actions based on position and street
        let action_names = if position.to_lowercase() == "oop" {
            vec!["Check", "Bet 33%"]
        } else {
            vec!["Check", "Bet 25%", "Bet 55%", "Bet 85%", "Bet 125%"]
        };

        let mut result = Vec::new();
        for (i, name) in action_names.iter().enumerate() {
            if i < probs.len() {
                result.push((name.to_string(), probs[i]));
            }
        }

        // Normalize to action count
        let sum: f32 = result.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in &mut result {
                *p /= sum;
            }
        }

        Ok(result)
    }
}

#[cfg(feature = "deep")]
fn run_interactive(solver: &DeepSolver) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Deep Strategy CLI (Interactive Mode) ===");
    println!("Board: Td 9d 6h (medium.json settings)");
    println!("Pot: 61, Stack: 477\n");
    println!("Commands:");
    println!("  <hand> <position>           - e.g., 'AhKs oop' or '9c9s ip'");
    println!("  <hand> <position> <turn>    - e.g., 'AhKs oop 2c'");
    println!("  quit                        - exit\n");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() { continue; }
        if input == "quit" || input == "exit" || input == "q" { break; }

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.len() < 2 {
            println!("Usage: <hand> <position> [turn] [river]");
            println!("Example: AhKs oop");
            continue;
        }

        let hand = match parse_hand(parts[0]) {
            Some(h) => h,
            None => {
                println!("Invalid hand: {}", parts[0]);
                continue;
            }
        };

        let position = parts[1];
        if position != "oop" && position != "ip" {
            println!("Position must be 'oop' or 'ip'");
            continue;
        }

        let turn = if parts.len() > 2 {
            parse_card(parts[2])
        } else {
            None
        };

        let river = if parts.len() > 3 {
            parse_card(parts[3])
        } else {
            None
        };

        // Check for card conflicts
        let board_cards = [solver.flop[0], solver.flop[1], solver.flop[2]];
        if board_cards.contains(&hand.0) || board_cards.contains(&hand.1) {
            println!("Hand conflicts with board cards");
            continue;
        }
        if let Some(t) = turn {
            if board_cards.contains(&t) || t == hand.0 || t == hand.1 {
                println!("Turn conflicts with board or hand");
                continue;
            }
        }

        // Default pot/stack ratios for flop open (must match training!)
        // Training uses: pot = pot0 + 2*(stack0 - bet_amount) = 61 + 2*477 = 1015
        // pot_ratio = 1015 / 1015 = 1.0
        let pot_ratio = 1.0;
        let stack_ratio = 0.0; // no bet yet

        match solver.get_strategy(hand, position, turn, river, pot_ratio, stack_ratio) {
            Ok(strategy) => {
                let hand_str = format!("{}{}", card_to_string(hand.0), card_to_string(hand.1));
                let street = if turn.is_some() { "turn" } else { "flop" };
                println!("\nHand: {} | Position: {} | Street: {}", hand_str, position.to_uppercase(), street);
                println!("Strategy:");
                for (action, prob) in &strategy {
                    let bar_len = (prob * 40.0) as usize;
                    let bar: String = "█".repeat(bar_len);
                    println!("  {:12} {:5.1}% {}", action, prob * 100.0, bar);
                }
                println!();
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    Ok(())
}

#[cfg(feature = "deep")]
fn show_range_summary(solver: &DeepSolver, position: &str) -> Result<(), Box<dyn std::error::Error>> {
    let board_cards = [solver.flop[0], solver.flop[1], solver.flop[2]];

    // Get all hands that don't conflict with board
    // For simplicity, use all valid hands (the model was trained on full ranges)
    let mut hands = Vec::new();
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            // Skip if conflicts with board
            if !board_cards.contains(&c1) && !board_cards.contains(&c2) {
                hands.push((c1, c2));
            }
        }
    }

    println!("Analyzing {} {} hands on Td 9d 6h...\n", hands.len(), position.to_uppercase());

    let action_names: Vec<&str> = if position == "oop" {
        vec!["Check", "Bet 33%"]
    } else {
        vec!["Check", "Bet 25%", "Bet 55%", "Bet 85%", "Bet 125%"]
    };

    let mut total_probs = vec![0.0f32; action_names.len()];
    // Must match training: pot_ratio = 1.0 for root node
    let pot_ratio = 1.0;
    let stack_ratio = 0.0;

    for &hand in &hands {
        if let Ok(strategy) = solver.get_strategy(hand, position, None, None, pot_ratio, stack_ratio) {
            for (i, (_, prob)) in strategy.iter().enumerate() {
                if i < total_probs.len() {
                    total_probs[i] += prob;
                }
            }
        }
    }

    // Average
    let n = hands.len() as f32;
    for p in &mut total_probs {
        *p /= n;
    }

    println!("=== {} Range Summary (Flop: Td 9d 6h) ===\n", position.to_uppercase());
    println!("Total hands in range: {}\n", hands.len());
    println!("Average strategy across all hands:");

    for (i, name) in action_names.iter().enumerate() {
        let prob = total_probs[i];
        let bar_len = (prob * 50.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:12} {:5.1}% {}", name, prob * 100.0, bar);
    }

    // Show some example hands
    println!("\n--- Sample hands ---");
    let sample_hands = [
        ("AhAs", "Aces"),
        ("KhKs", "Kings"),
        ("TcTs", "Top set"),
        ("9c9s", "Middle set"),
        ("AdKd", "AK flush draw"),
        ("7h8h", "OESD"),
        ("2c3c", "Low suited"),
    ];

    for (hand_str, label) in sample_hands {
        if let Some(hand) = parse_hand(hand_str) {
            if !board_cards.contains(&hand.0) && !board_cards.contains(&hand.1) {
                if let Ok(strategy) = solver.get_strategy(hand, position, None, None, pot_ratio, stack_ratio) {
                    let probs: Vec<String> = strategy.iter()
                        .map(|(a, p)| format!("{}: {:.0}%", a.chars().take(5).collect::<String>(), p * 100.0))
                        .collect();
                    println!("  {:8} {:12} -> {}", hand_str, label, probs.join(", "));
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut hand: Option<(u8, u8)> = None;
    let mut position: Option<String> = None;
    let mut turn: Option<u8> = None;
    let mut river: Option<u8> = None;
    let mut interactive = false;
    let mut range_summary = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hand" | "-h" => {
                if i + 1 < args.len() {
                    hand = parse_hand(&args[i + 1]);
                    i += 1;
                }
            }
            "--position" | "-p" => {
                if i + 1 < args.len() {
                    position = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--turn" | "-t" => {
                if i + 1 < args.len() {
                    turn = parse_card(&args[i + 1]);
                    i += 1;
                }
            }
            "--river" | "-r" => {
                if i + 1 < args.len() {
                    river = parse_card(&args[i + 1]);
                    i += 1;
                }
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            "--range" | "--summary" => {
                range_summary = true;
            }
            "--help" => {
                println!("Deep Strategy CLI - Query neural network for poker strategies\n");
                println!("Usage:");
                println!("  deep_cli --hand AhKs --position oop");
                println!("  deep_cli --hand 9d9c --position ip --turn 2h");
                println!("  deep_cli --range --position oop    # Show full range summary");
                println!("  deep_cli --interactive\n");
                println!("Options:");
                println!("  --hand, -h       Hand (e.g., AhKs, 9d9c)");
                println!("  --position, -p   Position (oop or ip)");
                println!("  --turn, -t       Turn card (optional)");
                println!("  --river, -r      River card (optional)");
                println!("  --range          Show total % across entire range");
                println!("  --interactive    Interactive mode");
                println!("  --help           Show this help\n");
                println!("Board: Td 9d 6h | Pot: 61 | Stack: 477");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    println!("Loading neural network weights...");
    let solver = DeepSolver::new()?;
    println!("Loaded! Board: Td 9d 6h\n");

    if interactive {
        return run_interactive(&solver);
    }

    if range_summary {
        let pos = position.as_deref().unwrap_or("oop");
        return show_range_summary(&solver, pos);
    }

    // Single query mode
    let hand = match hand {
        Some(h) => h,
        None => {
            println!("No hand specified. Use --interactive for interactive mode.");
            println!("Example: deep_cli --hand AhKs --position oop");
            return Ok(());
        }
    };

    let position = position.unwrap_or_else(|| "oop".to_string());

    // Check for card conflicts
    let board_cards = [solver.flop[0], solver.flop[1], solver.flop[2]];
    if board_cards.contains(&hand.0) || board_cards.contains(&hand.1) {
        println!("Error: Hand conflicts with board cards");
        return Ok(());
    }

    // Must match training: pot_ratio = 1.0 for root node
    let pot_ratio = 1.0;
    let stack_ratio = 0.0;

    let strategy = solver.get_strategy(hand, &position, turn, river, pot_ratio, stack_ratio)?;

    let hand_str = format!("{}{}", card_to_string(hand.0), card_to_string(hand.1));
    let street = if turn.is_some() {
        if river.is_some() { "river" } else { "turn" }
    } else {
        "flop"
    };

    println!("Board: Td 9d 6h");
    if let Some(t) = turn {
        print!(" {}", card_to_string(t));
    }
    if let Some(r) = river {
        print!(" {}", card_to_string(r));
    }
    println!("\nHand: {} | Position: {} | Street: {}\n", hand_str, position.to_uppercase(), street);

    println!("Strategy:");
    for (action, prob) in &strategy {
        let bar_len = (prob * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  {:12} {:5.1}% {}", action, prob * 100.0, bar);
    }

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_cli --release --features \"deep\"");
}
