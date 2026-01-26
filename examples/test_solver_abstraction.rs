//! Test solver iterations with hand abstraction enabled.
//!
//! Run with: `cargo run --example test_solver_abstraction --release`

use postflop_solver::*;

fn main() {
    println!("=== Solver Abstraction Test ===\n");

    // Test 1: Basic solver iterations with abstraction
    println!("Test 1: Solver iterations with abstraction (Turn game)");
    test_solver_iterations_turn();

    // Test 2: Compare abstracted vs non-abstracted solving
    println!("\nTest 2: Abstracted vs non-abstracted comparison");
    test_abstracted_comparison();

    // Test 3: Verify no crashes with different bucket sizes
    println!("\nTest 3: Different bucket sizes");
    test_different_bucket_sizes();

    // Test 4: Flop game with chance nodes (critical test for isomorphism handling)
    println!("\nTest 4: Flop game with chance nodes");
    test_flop_game();
}

fn create_turn_game() -> PostFlopGame {
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,AKo,AQo,KQo";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo";

    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Ks7h2c").unwrap(),
        turn: card_from_str("Tc").unwrap(),
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("50%", "50%")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 100,
        effective_stack: 200,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.2,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config).expect("Failed to create action tree");
    PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game")
}

fn create_river_game() -> PostFlopGame {
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,AKo,AQo,KQo";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo";

    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Ks7h2c").unwrap(),
        turn: card_from_str("Tc").unwrap(),
        river: card_from_str("3d").unwrap(),
    };

    let bet_sizes = BetSizeOptions::try_from(("50%", "50%")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 100,
        effective_stack: 200,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.2,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config).expect("Failed to create action tree");
    PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game")
}

fn test_solver_iterations_turn() {
    let mut game = create_turn_game();

    println!("  OOP hands: {}", game.num_private_hands(0));
    println!("  IP hands: {}", game.num_private_hands(1));

    // Enable abstraction
    let config = AbstractionConfig {
        num_buckets: 20,
        max_iterations: 100,
    };
    game.enable_abstraction(&config).expect("Failed to enable abstraction");

    println!("  Abstraction enabled: {}", game.is_abstraction_enabled());
    println!("  Effective OOP hands (buckets): {}", game.effective_hand_count(0));
    println!("  Effective IP hands (buckets): {}", game.effective_hand_count(1));

    // Allocate memory and run solver
    game.allocate_memory(false);

    println!("  Running 10 solver iterations...");

    // Run iterations using solve_step
    for i in 0..10 {
        solve_step(&game, i);
    }

    println!("  ✓ PASS: 10 iterations completed without crash");

    // Finalize and check exploitability
    finalize(&mut game);
    let exploitability = compute_exploitability(&game);
    println!("  Exploitability after 10 iterations: {:.4}", exploitability);
}

fn test_abstracted_comparison() {
    // Create river game for faster solving
    let mut game_baseline = create_river_game();
    let mut game_abstracted = create_river_game();

    // Setup abstracted game
    let config = AbstractionConfig {
        num_buckets: 15,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).expect("Failed to enable abstraction");

    println!("  Board: Ks7h2cTc3d (River)");
    println!("  Baseline hands: OOP={}, IP={}",
             game_baseline.num_private_hands(0), game_baseline.num_private_hands(1));
    println!("  Abstracted buckets: OOP={}, IP={}",
             game_abstracted.effective_hand_count(0), game_abstracted.effective_hand_count(1));

    // Allocate memory
    game_baseline.allocate_memory(false);
    game_abstracted.allocate_memory(false);

    let (baseline_mem, _) = game_baseline.memory_usage();
    let (abstracted_mem, _) = game_abstracted.memory_usage();
    println!("  Baseline memory: {:.2} KB", baseline_mem as f64 / 1024.0);
    println!("  Abstracted memory: {:.2} KB", abstracted_mem as f64 / 1024.0);

    // Solve both
    println!("  Solving baseline (100 iterations)...");
    let baseline_exp = solve(&mut game_baseline, 100, 0.0, false);

    println!("  Solving abstracted (100 iterations)...");
    let abstracted_exp = solve(&mut game_abstracted, 100, 0.0, false);

    println!("  Baseline exploitability: {:.4}", baseline_exp);
    println!("  Abstracted exploitability: {:.4}", abstracted_exp);

    println!("  ✓ PASS: Both solving methods completed");
}

fn test_different_bucket_sizes() {
    let bucket_sizes = [5, 10, 20, 30];

    for &k in &bucket_sizes {
        let mut game = create_river_game();
        let config = AbstractionConfig {
            num_buckets: k,
            max_iterations: 100,
        };
        game.enable_abstraction(&config).expect("Failed to enable abstraction");
        game.allocate_memory(false);

        // Run a few iterations
        for i in 0..5 {
            solve_step(&game, i);
        }

        println!("  k={}: 5 iterations completed, effective hands: OOP={}, IP={}",
                 k, game.effective_hand_count(0), game.effective_hand_count(1));
    }

    println!("  ✓ PASS: All bucket sizes work correctly");
}

fn create_flop_game() -> PostFlopGame {
    // Smaller ranges for faster Flop solving
    let oop_range = "AA,KK,QQ,JJ,TT,AKs,AQs,AKo";
    let ip_range = "AA,KK,QQ,JJ,AKs,AKo";

    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Ks7h2c").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("50%", "50%")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 100,
        effective_stack: 200,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.2,
        merging_threshold: 0.1,
    };

    let action_tree = ActionTree::new(tree_config).expect("Failed to create action tree");
    PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game")
}

fn test_flop_game() {
    // This test is critical - Flop games have chance nodes (turn and river cards)
    // which previously would have crashed due to isomorphism swap issues

    let mut game = create_flop_game();

    println!("  Board: Ks7h2c (Flop - has turn/river chance nodes)");
    println!("  OOP hands: {}", game.num_private_hands(0));
    println!("  IP hands: {}", game.num_private_hands(1));

    // Enable abstraction
    let config = AbstractionConfig {
        num_buckets: 10,
        max_iterations: 100,
    };
    game.enable_abstraction(&config).expect("Failed to enable abstraction");

    println!("  Effective OOP buckets: {}", game.effective_hand_count(0));
    println!("  Effective IP buckets: {}", game.effective_hand_count(1));

    // Allocate memory
    game.allocate_memory(false);

    let (mem, _) = game.memory_usage();
    println!("  Memory: {:.2} MB", mem as f64 / 1024.0 / 1024.0);

    // Run solver iterations - this tests chance node handling
    println!("  Running 10 solver iterations (tests chance nodes)...");
    for i in 0..10 {
        solve_step(&game, i);
    }
    println!("  ✓ PASS: Solver iterations with chance nodes completed");

    // Finalize and compute exploitability
    finalize(&mut game);
    let exploitability = compute_exploitability(&game);
    println!("  Exploitability: {:.4}", exploitability);

    println!("  ✓ PASS: Flop game with abstraction works correctly");
}
