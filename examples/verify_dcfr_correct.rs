//! Verify DCFR strategy using the CORRECT API
//!
//! Run: cargo run --example verify_dcfr_correct --release --features "bincode zstd"

use postflop_solver::*;

fn card_str(c: u8) -> String {
    let rank = c / 4;
    let suit = c % 4;
    let r = match rank { 0=>'2', 1=>'3', 2=>'4', 3=>'5', 4=>'6', 5=>'7', 6=>'8', 7=>'9', 8=>'T', 9=>'J', 10=>'Q', 11=>'K', 12=>'A', _=>'?' };
    let s = match suit { 0=>'c', 1=>'d', 2=>'h', 3=>'s', _=>'?' };
    format!("{}{}", r, s)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Verify DCFR Using Correct API ===\n");

    // Load game - this returns PostFlopGame which has the proper strategy() method
    let (mut game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;

    let flop = game.card_config().flop;
    println!("Board: {} {} {}", card_str(flop[0]), card_str(flop[1]), card_str(flop[2]));

    // The game is already at root after loading
    println!("Current player: {}", game.current_player());
    println!("Is chance: {}", game.is_chance_node());
    println!("Num actions: {}\n", game.available_actions().len());

    let actions = game.available_actions();
    println!("Available actions: {:?}\n", actions);

    // Use game.strategy() - this is the CORRECT method that normalizes properly!
    let strategy = game.strategy();
    let num_hands = game.private_cards(0).len();
    let num_actions = actions.len();

    println!("Strategy length: {} (expected {} * {} = {})",
        strategy.len(), num_actions, num_hands, num_actions * num_hands);

    // Calculate average strategy
    let mut action_totals = vec![0.0f64; num_actions];
    let hands = game.private_cards(0);

    for (hi, &(c1, c2)) in hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        for a in 0..num_actions {
            action_totals[a] += strategy[a * num_hands + hi] as f64;
        }
    }

    let valid_hands = hands.iter().filter(|&&(c1, c2)| !flop.contains(&c1) && !flop.contains(&c2)).count();

    println!("\n=== OOP Root Strategy (using game.strategy()) ===");
    for (i, action) in actions.iter().enumerate() {
        let avg = action_totals[i] / valid_hands as f64 * 100.0;
        println!("  {:?}: {:.1}%", action, avg);
    }

    // Show first 10 hands
    println!("\n=== Sample Hands ===");
    let mut shown = 0;
    for (hi, &(c1, c2)) in hands.iter().enumerate() {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }
        if shown >= 10 { break; }

        print!("  {}{}: ", card_str(c1), card_str(c2));
        for a in 0..num_actions {
            print!("{:?}={:.1}% ", actions[a], strategy[a * num_hands + hi] * 100.0);
        }
        println!();
        shown += 1;
    }

    Ok(())
}
