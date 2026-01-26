//! Test bucket equity precomputation for the postflop solver.
//!
//! Run with: `cargo run --example test_bucket_equity`

use postflop_solver::*;
use std::time::Instant;

fn main() {
    println!("=== Bucket Equity Precomputation Test ===\n");

    // Test 1: Basic abstraction data computation
    println!("Test 1: Basic AbstractionData computation (flop)");
    test_basic_abstraction();

    // Test 2: Verify bucket weights sum correctly
    println!("\nTest 2: Bucket weight verification");
    test_bucket_weights();

    // Test 3: Verify bucket equity symmetry
    println!("\nTest 3: Bucket equity verification");
    test_bucket_equity();

    // Test 4: Turn case
    println!("\nTest 4: Turn case (fewer runouts)");
    test_turn_abstraction();

    // Test 5: Benchmark precomputation time
    println!("\nTest 5: Benchmark precomputation time");
    benchmark_precomputation();
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
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,KQs";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,KQs";

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

fn test_basic_abstraction() {
    let game = create_flop_game();

    println!("  OOP hands: {}", game.private_cards(0).len());
    println!("  IP hands: {}", game.private_cards(1).len());

    let config = AbstractionConfig {
        num_buckets: 30,
        max_iterations: 100,
    };

    let start = Instant::now();
    let abstraction = AbstractionData::compute(&game, &config);
    let elapsed = start.elapsed();

    println!("  Computation time: {:?}", elapsed);
    println!("  OOP buckets: {}", abstraction.num_buckets(0));
    println!("  IP buckets: {}", abstraction.num_buckets(1));
    println!("  Runouts stored: {}", abstraction.num_runouts);
    println!("  Memory usage: {:.2} KB", abstraction.memory_usage() as f64 / 1024.0);

    // Verify mappings
    let oop_hands = game.private_cards(0).len();
    let ip_hands = game.private_cards(1).len();
    assert_eq!(abstraction.hand_to_bucket[0].len(), oop_hands);
    assert_eq!(abstraction.hand_to_bucket[1].len(), ip_hands);

    // Check all buckets are valid
    for &bucket in &abstraction.hand_to_bucket[0] {
        assert!((bucket as usize) < abstraction.num_buckets(0));
    }
    for &bucket in &abstraction.hand_to_bucket[1] {
        assert!((bucket as usize) < abstraction.num_buckets(1));
    }

    println!("  ✓ PASS: AbstractionData computed correctly");
}

fn test_bucket_weights() {
    let game = create_flop_game();

    let config = AbstractionConfig {
        num_buckets: 20,
        max_iterations: 100,
    };

    let abstraction = AbstractionData::compute(&game, &config);

    // Verify bucket weights sum to total hand weights for each player
    for player in 0..2 {
        let initial_weights = game.initial_weights(player);
        let total_hand_weight: f32 = initial_weights.iter().sum();
        let total_bucket_weight: f32 = abstraction.bucket_weights[player].iter().sum();

        println!(
            "  Player {} - Hand weight sum: {:.4}, Bucket weight sum: {:.4}",
            player, total_hand_weight, total_bucket_weight
        );

        // Allow small floating point tolerance
        let diff = (total_hand_weight - total_bucket_weight).abs();
        assert!(
            diff < 0.01,
            "Bucket weights don't sum to hand weights! Diff: {}",
            diff
        );
    }

    // Verify all hands in a bucket contribute to bucket weight
    for player in 0..2 {
        let initial_weights = game.initial_weights(player);
        for bucket in 0..abstraction.num_buckets(player) {
            let hands = abstraction.get_hands_in_bucket(player, bucket);
            let computed_weight: f32 = hands.iter().map(|&h| initial_weights[h as usize]).sum();
            let stored_weight = abstraction.get_bucket_weight(player, bucket);

            let diff = (computed_weight - stored_weight).abs();
            assert!(
                diff < 0.001,
                "Bucket {} weight mismatch! Computed: {}, Stored: {}",
                bucket,
                computed_weight,
                stored_weight
            );
        }
    }

    println!("  ✓ PASS: Bucket weights verified");
}

fn test_bucket_equity() {
    let game = create_flop_game();

    let config = AbstractionConfig {
        num_buckets: 10,
        max_iterations: 100,
    };

    let abstraction = AbstractionData::compute(&game, &config);

    // Get a specific runout equity and verify properties
    // For flop game, we need to pick valid turn+river
    let test_turn = card_from_str("Ah").unwrap();
    let test_river = card_from_str("3d").unwrap();

    if let Some(equity_matrix) = abstraction.get_bucket_equity(test_turn, test_river) {
        println!("  Testing runout Ah3d:");
        println!("  Matrix size: {} x {}", equity_matrix.len(), equity_matrix[0].len());

        // Verify dimensions
        assert_eq!(equity_matrix.len(), abstraction.num_buckets(0));
        assert_eq!(equity_matrix[0].len(), abstraction.num_buckets(1));

        // Verify all values are in [0, 1]
        let mut min_eq = 1.0f32;
        let mut max_eq = 0.0f32;
        let mut sum_eq = 0.0f64;
        let mut count = 0usize;

        for row in equity_matrix {
            for &eq in row {
                assert!(eq >= 0.0 && eq <= 1.0, "Equity {} out of range", eq);
                min_eq = min_eq.min(eq);
                max_eq = max_eq.max(eq);
                sum_eq += eq as f64;
                count += 1;
            }
        }

        let avg_eq = sum_eq / count as f64;
        println!("  Equity range: [{:.3}, {:.3}], avg: {:.3}", min_eq, max_eq, avg_eq);

        // Average should be somewhere around 0.5 (roughly balanced game)
        assert!(
            avg_eq > 0.2 && avg_eq < 0.8,
            "Average equity {} seems unreasonable",
            avg_eq
        );

        println!("  ✓ PASS: Bucket equity values verified");
    } else {
        println!("  ✗ FAIL: Could not get equity for runout");
    }

    // Print sample equity matrix
    println!("\n  Sample equity matrix (first 5x5):");
    if let Some(eq) = abstraction.get_bucket_equity(test_turn, test_river) {
        print!("        ");
        for j in 0..5.min(abstraction.num_buckets(1)) {
            print!("  IP{:<2}", j);
        }
        println!();
        for i in 0..5.min(abstraction.num_buckets(0)) {
            print!("  OOP{}: ", i);
            for j in 0..5.min(abstraction.num_buckets(1)) {
                print!(" {:.2}", eq[i][j]);
            }
            println!();
        }
    }
}

fn test_turn_abstraction() {
    let game = create_turn_game();

    println!("  Board: Ks7h2c Tc");
    println!("  OOP hands: {}", game.private_cards(0).len());
    println!("  IP hands: {}", game.private_cards(1).len());

    let config = AbstractionConfig {
        num_buckets: 15,
        max_iterations: 100,
    };

    let start = Instant::now();
    let abstraction = AbstractionData::compute(&game, &config);
    let elapsed = start.elapsed();

    println!("  Computation time: {:?}", elapsed);
    println!("  Runouts stored: {} (should be ~48 rivers)", abstraction.num_runouts);
    println!("  Memory usage: {:.2} KB", abstraction.memory_usage() as f64 / 1024.0);

    // Turn has 52 - 3 (flop) - 1 (turn) = 48 possible rivers
    assert!(
        abstraction.num_runouts > 40 && abstraction.num_runouts <= 48,
        "Expected ~48 runouts for turn, got {}",
        abstraction.num_runouts
    );

    println!("  ✓ PASS: Turn abstraction computed correctly");
}

fn benchmark_precomputation() {
    let game = create_flop_game();

    println!("  Flop game: OOP={} hands, IP={} hands",
             game.private_cards(0).len(),
             game.private_cards(1).len());

    for &num_buckets in &[30, 50, 100] {
        let config = AbstractionConfig {
            num_buckets,
            max_iterations: 100,
        };

        let start = Instant::now();
        let abstraction = AbstractionData::compute(&game, &config);
        let elapsed = start.elapsed();

        println!(
            "  k={:3}: {:?}, {} runouts, {:.1} KB",
            num_buckets,
            elapsed,
            abstraction.num_runouts,
            abstraction.memory_usage() as f64 / 1024.0
        );
    }

    println!("  ✓ Benchmark complete");
}
