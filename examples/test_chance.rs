use postflop_solver::*;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("Loading 50bb.flop...");
    let (mut game, memo): (PostFlopGame, String) =
        load_data_from_file("50bb.flop", None).expect("Failed to load");

    println!("Loaded! Memo: {}", memo);
    println!("Is solved: {}", game.is_solved());
    println!("Is chance node: {}", game.is_chance_node());

    // Play action 0 twice (like the UI does)
    println!("\nPlaying action 0...");
    game.play(0);
    println!("After action 0: is_chance={}", game.is_chance_node());

    println!("\nPlaying action 0 again...");
    game.play(0);
    println!("After action 0: is_chance={}", game.is_chance_node());

    // Now we should be at a chance node
    println!("\nAt chance node. Available actions:");
    for (i, action) in game.available_actions().iter().enumerate() {
        println!("  [{}] {:?}", i, action);
        if i > 10 {
            println!("  ... ({} total)", game.available_actions().len());
            break;
        }
    }

    println!("\nPossible cards: {:064b}", game.possible_cards());

    // Try playing usize::MAX (auto-select)
    println!("\nTrying to play usize::MAX (auto-select)...");
    game.play(usize::MAX);

    println!("Success! Now at: is_chance={}, is_terminal={}",
        game.is_chance_node(), game.is_terminal_node());
}
