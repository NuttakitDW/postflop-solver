//! Compare Deep network predictions vs actual DCFR solution
//!
//! Run: cargo run --example deep_vs_dcfr --release --features "deep bincode zstd"

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

    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }
    if turn < 52 { f[156 + turn as usize] = 1.0; }
    if river < 52 { f[208 + river as usize] = 1.0; }

    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[260 + c1 as usize] = 1.0; }
    if c2 < 52 { f[312 + c2 as usize] = 1.0; }

    f[364] = pot_ratio;
    f[365] = stack_ratio;

    if street < 3 { f[366 + street] = 1.0; }

    f
}

#[cfg(feature = "deep")]
fn card_to_string(card: u8) -> String {
    if card >= 52 { return "?".to_string(); }
    let rank = card / 4;
    let suit = card % 4;
    let rank_char = match rank {
        0 => '2', 1 => '3', 2 => '4', 3 => '5', 4 => '6',
        5 => '7', 6 => '8', 7 => '9', 8 => 'T', 9 => 'J',
        10 => 'Q', 11 => 'K', 12 => 'A', _ => '?',
    };
    let suit_char = match suit { 0 => 'c', 1 => 'd', 2 => 'h', 3 => 's', _ => '?' };
    format!("{}{}", rank_char, suit_char)
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comparing Deep Network vs DCFR Solution ===\n");

    // Load DCFR solution
    println!("Loading DCFR solution from out/50bb-medium.flop...");
    let (game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;
    println!("Loaded!\n");

    // Load neural network
    println!("Loading neural network weights...");
    let device = Device::Cpu;
    let mut var_map = VarMap::new();
    var_map.load("out/deep_medium_weights.safetensors")?;
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 369, 5)?;
    println!("Loaded!\n");

    // Get root node from DCFR
    let root = game.root();
    let flop = game.card_config().flop;
    let hands = game.private_cards(0); // OOP hands
    let num_actions = root.num_actions();
    let dcfr_strategy = root.strategy();

    println!("=== DCFR Root Node (OOP First to Act) ===");
    println!("Board: {} {} {}", card_to_string(flop[0]), card_to_string(flop[1]), card_to_string(flop[2]));
    println!("OOP hands: {}", hands.len());
    println!("Actions: {}", num_actions);
    println!();

    // Calculate DCFR aggregate strategy
    let mut dcfr_totals = vec![0.0f32; num_actions];
    let mut valid_count = 0;

    for (hi, &(c1, c2)) in hands.iter().enumerate() {
        // Skip blocked hands
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let mut hand_strat = Vec::new();
        for a in 0..num_actions {
            hand_strat.push(dcfr_strategy[a * hands.len() + hi]);
        }

        let sum: f32 = hand_strat.iter().sum();
        if sum > 0.01 {
            for (a, &s) in hand_strat.iter().enumerate() {
                dcfr_totals[a] += s / sum;
            }
            valid_count += 1;
        }
    }

    // Average
    for t in &mut dcfr_totals {
        *t /= valid_count as f32;
    }

    println!("=== DCFR Average Strategy (OOP Flop) ===");
    let action_names = ["Check", "Bet 33%"];
    for (i, name) in action_names.iter().enumerate() {
        if i < dcfr_totals.len() {
            println!("  {:12} {:5.1}%", name, dcfr_totals[i] * 100.0);
        }
    }
    println!();

    // Calculate Neural Network aggregate strategy
    // IMPORTANT: Must match training feature encoding exactly!
    let tc = game.tree_config();
    let pot0 = tc.starting_pot as f32;
    let stack0 = tc.effective_stack as f32;
    // Training uses: pot = pot0 + 2.0 * (stack0 - bet_amount)
    // For root node: pot = 61 + 2*(477-0) = 1015
    // pot_ratio = pot / (pot0 + 2*stack0) = 1015/1015 = 1.0
    let bet_amount = 0.0; // root node
    let pot = pot0 + 2.0 * (stack0 - bet_amount);
    let pot_ratio = pot / (pot0 + 2.0 * stack0);
    let stack_ratio = bet_amount / stack0;

    let mut nn_totals = vec![0.0f32; num_actions];
    let mut nn_count = 0;

    for &(c1, c2) in hands.iter() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let features = encode(flop, 255, 255, (c1, c2), pot_ratio, stack_ratio, 0);
        let x = Tensor::from_vec(features, (1, 369), &device)?;
        let pred = net.predict(&x)?;
        let probs: Vec<f32> = pred.flatten_all()?.to_vec1()?;

        // Only take first num_actions
        let sum: f32 = probs.iter().take(num_actions).sum();
        if sum > 0.01 {
            for a in 0..num_actions {
                nn_totals[a] += probs[a] / sum;
            }
            nn_count += 1;
        }
    }

    for t in &mut nn_totals {
        *t /= nn_count as f32;
    }

    println!("=== Neural Network Average Strategy (OOP Flop) ===");
    for (i, name) in action_names.iter().enumerate() {
        if i < nn_totals.len() {
            println!("  {:12} {:5.1}%", name, nn_totals[i] * 100.0);
        }
    }
    println!();

    // Compare
    println!("=== Comparison ===");
    println!("{:12} {:>10} {:>10} {:>10}", "Action", "DCFR", "NN", "Error");
    println!("{:-<46}", "");

    let mut total_error = 0.0f32;
    for (i, name) in action_names.iter().enumerate() {
        if i < dcfr_totals.len() && i < nn_totals.len() {
            let error = (dcfr_totals[i] - nn_totals[i]).abs() * 100.0;
            total_error += error;
            println!("{:12} {:>9.1}% {:>9.1}% {:>9.1}%",
                name, dcfr_totals[i] * 100.0, nn_totals[i] * 100.0, error);
        }
    }
    println!("{:-<46}", "");
    println!("Total absolute error: {:.1}%", total_error);

    // Show some individual hand comparisons
    println!("\n=== Sample Hand Comparisons ===");
    println!("{:8} {:>12} {:>12} {:>12} {:>12}", "Hand", "DCFR Chk", "NN Chk", "DCFR Bet", "NN Bet");
    println!("{:-<60}", "");

    for (hi, &(c1, c2)) in hands.iter().enumerate().take(20) {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        // DCFR strategy for this hand
        let mut dcfr_strat = Vec::new();
        for a in 0..num_actions {
            dcfr_strat.push(dcfr_strategy[a * hands.len() + hi]);
        }
        let sum: f32 = dcfr_strat.iter().sum();
        if sum > 0.01 {
            for s in &mut dcfr_strat { *s /= sum; }
        }

        // NN strategy
        let features = encode(flop, 255, 255, (c1, c2), pot_ratio, stack_ratio, 0);
        let x = Tensor::from_vec(features, (1, 369), &device)?;
        let pred = net.predict(&x)?;
        let nn_strat: Vec<f32> = pred.flatten_all()?.to_vec1()?;

        let hand_str = format!("{}{}", card_to_string(c1), card_to_string(c2));
        println!("{:8} {:>11.1}% {:>11.1}% {:>11.1}% {:>11.1}%",
            hand_str,
            dcfr_strat.get(0).unwrap_or(&0.0) * 100.0,
            nn_strat.get(0).unwrap_or(&0.0) * 100.0,
            dcfr_strat.get(1).unwrap_or(&0.0) * 100.0,
            nn_strat.get(1).unwrap_or(&0.0) * 100.0,
        );
    }

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_vs_dcfr --release --features \"deep bincode zstd\"");
}
