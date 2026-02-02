//! Check what training samples were collected from the DCFR solution
//!
//! Run: cargo run --example check_training_data --release --features "bincode zstd"

use postflop_solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Checking Training Data from DCFR Solution ===\n");

    // Load DCFR solution
    println!("Loading DCFR solution...");
    let (game, _): (PostFlopGame, String) = load_data_from_file("out/50bb-medium.flop", None)?;
    println!("Loaded!\n");

    let flop = game.card_config().flop;
    let tc = game.tree_config();
    let pot0 = tc.starting_pot as f32;
    let stack0 = tc.effective_stack as f32;
    let hands = [game.private_cards(0), game.private_cards(1)];

    println!("Flop: {:?}", flop);
    println!("Pot: {}, Stack: {}", pot0, stack0);
    println!("OOP hands: {}, IP hands: {}", hands[0].len(), hands[1].len());
    println!();

    // Check root node
    let root = game.root();
    println!("=== ROOT NODE ===");
    println!("Is terminal: {}", root.is_terminal());
    println!("Is chance: {}", root.is_chance());
    println!("Player: {}", root.player());
    println!("Num actions: {}", root.num_actions());
    println!("Bet amount: {}", root.bet_amount());

    let strategy = root.strategy();
    let num_actions = root.num_actions();
    let num_hands = hands[0].len();

    println!("Strategy length: {} (expected: {} * {} = {})",
        strategy.len(), num_actions, num_hands, num_actions * num_hands);
    println!();

    // Calculate what pot_ratio would be for training
    let bet_amount = root.bet_amount() as f32;
    let pot = pot0 + 2.0 * (stack0 - bet_amount);
    let pot_ratio = pot / (pot0 + 2.0 * stack0);
    let stack_ratio = bet_amount / stack0;
    println!("Training features for root:");
    println!("  pot = {} + 2*({} - {}) = {}", pot0, stack0, bet_amount, pot);
    println!("  pot_ratio = {} / {} = {}", pot, pot0 + 2.0 * stack0, pot_ratio);
    println!("  stack_ratio = {} / {} = {}", bet_amount, stack0, stack_ratio);
    println!();

    // Check some hand strategies at root
    println!("=== Sample Root Node Strategies ===");
    println!("{:8} {:>10} {:>10} {:>10}", "Hand", "Check", "Bet 33%", "Sum");
    println!("{:-<45}", "");

    let mut check_total = 0.0f32;
    let mut bet_total = 0.0f32;
    let mut valid_hands = 0;

    for (hi, &(c1, c2)) in hands[0].iter().enumerate() {
        // Skip blocked hands
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let check_prob = strategy[0 * num_hands + hi];
        let bet_prob = strategy[1 * num_hands + hi];
        let sum = check_prob + bet_prob;

        // Normalize
        let (check_norm, bet_norm) = if sum > 0.001 {
            (check_prob / sum, bet_prob / sum)
        } else {
            (0.0, 0.0)
        };

        check_total += check_norm;
        bet_total += bet_norm;
        valid_hands += 1;

        if hi < 20 {
            let card_str = |c: u8| {
                let rank = c / 4;
                let suit = c % 4;
                let r = match rank { 0=>'2', 1=>'3', 2=>'4', 3=>'5', 4=>'6', 5=>'7', 6=>'8', 7=>'9', 8=>'T', 9=>'J', 10=>'Q', 11=>'K', 12=>'A', _=>'?' };
                let s = match suit { 0=>'c', 1=>'d', 2=>'h', 3=>'s', _=>'?' };
                format!("{}{}", r, s)
            };
            println!("{:8} {:>9.1}% {:>9.1}% {:>9.3}",
                format!("{}{}", card_str(c1), card_str(c2)),
                check_norm * 100.0, bet_norm * 100.0, sum);
        }
    }

    println!("{:-<45}", "");
    println!("Average: {:>9.1}% {:>9.1}%",
        check_total / valid_hands as f32 * 100.0,
        bet_total / valid_hands as f32 * 100.0);
    println!("Valid hands: {}", valid_hands);

    // Now simulate what collect_samples would do
    println!("\n=== Simulating collect_samples ===");

    // The collect_samples uses step_by(3) to skip hands
    let mut collected_check = 0.0f32;
    let mut collected_bet = 0.0f32;
    let mut collected_count = 0;

    for (hi, &(c1, c2)) in hands[0].iter().enumerate().step_by(3) {
        if flop.contains(&c1) || flop.contains(&c2) { continue; }

        let check_prob = strategy[0 * num_hands + hi];
        let bet_prob = strategy[1 * num_hands + hi];
        let sum = check_prob + bet_prob;

        if sum > 0.01 {
            collected_check += check_prob / sum;
            collected_bet += bet_prob / sum;
            collected_count += 1;
        }
    }

    if collected_count > 0 {
        println!("Collected {} samples from root with step_by(3)", collected_count);
        println!("Average strategy in training data:");
        println!("  Check: {:.1}%", collected_check / collected_count as f32 * 100.0);
        println!("  Bet 33%: {:.1}%", collected_bet / collected_count as f32 * 100.0);
    } else {
        println!("NO SAMPLES COLLECTED FROM ROOT NODE!");
    }

    Ok(())
}
