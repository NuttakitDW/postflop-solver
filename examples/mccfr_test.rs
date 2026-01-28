use postflop_solver::*;

fn create_config() -> (CardConfig, TreeConfig) {
    // Simple setup for testing MCCFR
    let oop_range = "AA,KK,QQ,AKs,AKo";
    let ip_range = "JJ,TT,99,AQs,AQo";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: card_from_str("Qc").unwrap(),
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("50%, a", "2x")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 100,
        effective_stack: 400,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        max_raises_per_street: 3,
    };

    (card_config, tree_config)
}

fn main() {
    let iterations = 2000;
    let target_exploitability = 1.0; // 1% of pot

    // Test MCCFR with default (random) seed
    println!("=== Testing MCCFR ({} iterations) ===", iterations);
    let (card_config, tree_config) = create_config();
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game.allocate_memory(false);

    let exploitability = solve(&mut game, iterations, target_exploitability, true);
    println!("MCCFR final exploitability: {:.4}", exploitability);

    // Test MCCFR with fixed seed for reproducibility
    println!("\n=== Testing MCCFR with fixed seed (seed=42) ===");
    let (card_config, tree_config) = create_config();
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game_seeded = PostFlopGame::with_config(card_config, action_tree).unwrap();
    game_seeded.allocate_memory(false);

    let exploitability_seeded =
        solve_with_seed(&mut game_seeded, iterations, target_exploitability, true, Some(42));
    println!("MCCFR (seeded) final exploitability: {:.4}", exploitability_seeded);

    // Summary
    println!("\n=== Summary ===");
    println!("MCCFR exploitability (random):  {:.4}", exploitability);
    println!("MCCFR exploitability (seeded):  {:.4}", exploitability_seeded);
    println!("MCCFR uses External Sampling with CFR+ enhancements.");
}
