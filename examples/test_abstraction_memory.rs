//! Test memory reduction with hand abstraction enabled.
//!
//! Run with: `cargo run --example test_abstraction_memory --release`

use postflop_solver::*;

fn main() {
    println!("=== Hand Abstraction Memory Reduction Test ===\n");

    // Test 1: Compare memory usage with and without abstraction
    println!("Test 1: Memory comparison (Flop game)");
    test_memory_comparison_flop();

    // Test 2: Turn game comparison
    println!("\nTest 2: Memory comparison (Turn game)");
    test_memory_comparison_turn();

    // Test 3: effective_hand_count() verification
    println!("\nTest 3: effective_hand_count() verification");
    test_effective_hand_count();

    // Test 4: enable_abstraction() error handling
    println!("\nTest 4: Error handling tests");
    test_error_handling();
}

fn create_flop_game() -> PostFlopGame {
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s,AKo,AQo,KQo";

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

fn create_turn_game() -> PostFlopGame {
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s,AKo,AQo,KQo";

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

fn test_memory_comparison_flop() {
    // Create game without abstraction
    let game_baseline = create_flop_game();
    let (baseline_uncompressed, baseline_compressed) = game_baseline.memory_usage();

    println!("  OOP hands: {}", game_baseline.private_cards(0).len());
    println!("  IP hands: {}", game_baseline.private_cards(1).len());
    println!("  Baseline memory: {:.2} MB (uncompressed), {:.2} MB (compressed)",
             baseline_uncompressed as f64 / 1024.0 / 1024.0,
             baseline_compressed as f64 / 1024.0 / 1024.0);

    // Create game with abstraction
    let mut game_abstracted = create_flop_game();
    let config = AbstractionConfig {
        num_buckets: 50,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).expect("Failed to enable abstraction");

    let (abstracted_uncompressed, abstracted_compressed) = game_abstracted.memory_usage();

    println!("  Abstracted memory (k=50): {:.2} MB (uncompressed), {:.2} MB (compressed)",
             abstracted_uncompressed as f64 / 1024.0 / 1024.0,
             abstracted_compressed as f64 / 1024.0 / 1024.0);

    let reduction_uncompressed = 100.0 * (1.0 - abstracted_uncompressed as f64 / baseline_uncompressed as f64);
    let reduction_compressed = 100.0 * (1.0 - abstracted_compressed as f64 / baseline_compressed as f64);

    println!("  Memory reduction: {:.1}% (uncompressed), {:.1}% (compressed)",
             reduction_uncompressed, reduction_compressed);

    // Verify significant reduction
    if reduction_compressed >= 50.0 {
        println!("  ✓ PASS: Memory reduction >= 50%");
    } else {
        println!("  ✗ FAIL: Memory reduction < 50% (got {:.1}%)", reduction_compressed);
    }

    // Test with smaller k for even more reduction
    let mut game_small_k = create_flop_game();
    let small_config = AbstractionConfig {
        num_buckets: 20,
        max_iterations: 100,
    };
    game_small_k.enable_abstraction(&small_config).expect("Failed to enable abstraction");

    let (small_k_uncompressed, small_k_compressed) = game_small_k.memory_usage();
    let reduction_small = 100.0 * (1.0 - small_k_compressed as f64 / baseline_compressed as f64);

    println!("  Abstracted memory (k=20): {:.2} MB (compressed), {:.1}% reduction",
             small_k_compressed as f64 / 1024.0 / 1024.0, reduction_small);
}

fn test_memory_comparison_turn() {
    // Create game without abstraction
    let game_baseline = create_turn_game();
    let (baseline_uncompressed, baseline_compressed) = game_baseline.memory_usage();

    println!("  Board: Ks7h2c Tc");
    println!("  OOP hands: {}", game_baseline.private_cards(0).len());
    println!("  IP hands: {}", game_baseline.private_cards(1).len());
    println!("  Baseline memory: {:.2} MB (compressed)",
             baseline_compressed as f64 / 1024.0 / 1024.0);

    // Create game with abstraction
    let mut game_abstracted = create_turn_game();
    let config = AbstractionConfig {
        num_buckets: 30,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).expect("Failed to enable abstraction");

    let (abstracted_uncompressed, abstracted_compressed) = game_abstracted.memory_usage();

    println!("  Abstracted memory (k=30): {:.2} MB (compressed)",
             abstracted_compressed as f64 / 1024.0 / 1024.0);

    let reduction = 100.0 * (1.0 - abstracted_compressed as f64 / baseline_compressed as f64);
    println!("  Memory reduction: {:.1}%", reduction);

    if reduction >= 50.0 {
        println!("  ✓ PASS: Memory reduction >= 50%");
    } else {
        println!("  ✗ FAIL: Memory reduction < 50%");
    }
}

fn test_effective_hand_count() {
    // Without abstraction
    let game_normal = create_flop_game();
    let oop_hands = game_normal.num_private_hands(0);
    let ip_hands = game_normal.num_private_hands(1);

    println!("  Without abstraction:");
    println!("    effective_hand_count(OOP) = {} (same as num_private_hands)",
             game_normal.effective_hand_count(0));
    println!("    effective_hand_count(IP) = {} (same as num_private_hands)",
             game_normal.effective_hand_count(1));

    assert_eq!(game_normal.effective_hand_count(0), oop_hands);
    assert_eq!(game_normal.effective_hand_count(1), ip_hands);

    // With abstraction
    let mut game_abstracted = create_flop_game();
    let config = AbstractionConfig {
        num_buckets: 30,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).expect("Failed to enable abstraction");

    println!("  With abstraction (k=30):");
    println!("    effective_hand_count(OOP) = {} (buckets)",
             game_abstracted.effective_hand_count(0));
    println!("    effective_hand_count(IP) = {} (buckets)",
             game_abstracted.effective_hand_count(1));

    // Should return bucket counts
    assert!(game_abstracted.effective_hand_count(0) <= 30);
    assert!(game_abstracted.effective_hand_count(1) <= 30);

    // Verify abstraction is enabled
    assert!(game_abstracted.is_abstraction_enabled());
    assert!(game_abstracted.abstraction_data().is_some());

    println!("  ✓ PASS: effective_hand_count() returns correct values");
}

fn test_error_handling() {
    // Test: Cannot enable abstraction after memory is allocated
    let mut game = create_flop_game();
    game.allocate_memory(false);

    let config = AbstractionConfig::default();
    let result = game.enable_abstraction(&config);

    if result.is_err() {
        println!("  ✓ PASS: Cannot enable abstraction after memory allocation");
        println!("    Error: {}", result.unwrap_err());
    } else {
        println!("  ✗ FAIL: Should have returned error");
    }
}
