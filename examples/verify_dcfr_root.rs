//! Verify what the DCFR solution actually says for OOP flop strategy
//!
//! Run: cargo run --example verify_dcfr_root --release --features "bincode zstd"

use postflop_solver::*;

fn card_str(c: u8) -> String {
    let rank = c / 4;
    let suit = c % 4;
    let r = match rank { 0=>'2', 1=>'3', 2=>'4', 3=>'5', 4=>'6', 5=>'7', 6=>'8', 7=>'9', 8=>'T', 9=>'J', 10=>'Q', 11=>'K', 12=>'A', _=>'?' };
    let s = match suit { 0=>'c', 1=>'d', 2=>'h', 3=>'s', _=>'?' };
    format!("{}{}", r, s)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Verify DCFR Root Node Strategy ===\n");

    let (game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;

    let flop = game.card_config().flop;
    println!("Board: {} {} {}", card_str(flop[0]), card_str(flop[1]), card_str(flop[2]));

    let root = game.root();
    println!("Player to act: {} (0=OOP, 1=IP)", root.player());
    println!("Actions available: {}", root.num_actions());

    // Actions: 0 = Check, 1 = Bet 33%
    println!("(Action 0 = Check, Action 1 = Bet 33%)\n");

    let strategy = root.strategy();
    let hands = game.private_cards(0);
    let num_hands = hands.len();
    let num_actions = root.num_actions();

    println!("Strategy array length: {}", strategy.len());
    println!("Expected: {} actions * {} hands = {}\n", num_actions, num_hands, num_actions * num_hands);

    // Print raw strategy values for first 10 hands
    println!("=== Raw Strategy Values (first 10 valid hands) ===");
    println!("{:8} {:>12} {:>12} {:>12}", "Hand", "Action 0", "Action 1", "Sum");
    println!("{:-<50}", "");

    let mut shown = 0;
    for (hi, &(c1, c2)) in hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }
        if shown >= 10 { break; }

        let v0 = strategy[0 * num_hands + hi];
        let v1 = strategy[1 * num_hands + hi];
        let sum = v0 + v1;

        println!("{:8} {:>12.6} {:>12.6} {:>12.6}",
            format!("{}{}", card_str(c1), card_str(c2)), v0, v1, sum);
        shown += 1;
    }

    // Calculate overall frequencies
    println!("\n=== Overall Strategy (normalized) ===");
    let mut total_action0 = 0.0f64;
    let mut total_action1 = 0.0f64;
    let mut valid_count = 0;

    for (hi, &(c1, c2)) in hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let v0 = strategy[0 * num_hands + hi] as f64;
        let v1 = strategy[1 * num_hands + hi] as f64;
        let sum = v0 + v1;

        if sum > 0.0001 {
            total_action0 += v0 / sum;
            total_action1 += v1 / sum;
            valid_count += 1;
        }
    }

    if valid_count > 0 {
        println!("Valid hands: {}", valid_count);
        println!("Action 0 (Check?): {:.2}%", total_action0 / valid_count as f64 * 100.0);
        println!("Action 1 (Bet?):   {:.2}%", total_action1 / valid_count as f64 * 100.0);
    } else {
        println!("No valid hands found!");
    }

    // Also check using game's built-in method if available
    println!("\n=== Using game.root().strategy() directly ===");
    let strat = root.strategy();
    println!("First 20 values: {:?}", &strat[..20.min(strat.len())]);

    Ok(())
}
