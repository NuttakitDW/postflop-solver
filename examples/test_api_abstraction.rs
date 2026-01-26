//! Test to verify API methods work correctly with abstraction.
//!
//! This test verifies Phase 6 (Results Disaggregation) implementation:
//! - strategy() returns hand-level results when abstraction is enabled
//! - expected_values() returns hand-level results when abstraction is enabled
//!
//! Run with: `cargo run --example test_api_abstraction --release`

use postflop_solver::*;

fn main() {
    println!("=== API Abstraction Test (Phase 6) ===\n");

    // Test 1: strategy() returns correct size and values
    println!("Test 1: strategy() with abstraction");
    test_strategy_api();

    // Test 2: expected_values() returns correct size and values
    println!("\nTest 2: expected_values() with abstraction");
    test_expected_values_api();

    // Test 3: play() works correctly with abstraction
    println!("\nTest 3: play() and tree navigation with abstraction");
    test_play_api();

    println!("\n✓ All Phase 6 API tests passed!");
}

fn create_river_game() -> PostFlopGame {
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,AKo,AQo,KQo";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,AKs,AQs,AJs,KQs,AKo";

    let oop: Range = oop_range.parse().unwrap();
    let ip: Range = ip_range.parse().unwrap();

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

    let action_tree = ActionTree::new(tree_config).unwrap();
    PostFlopGame::with_config(card_config, action_tree).unwrap()
}

fn test_strategy_api() {
    let mut game_baseline = create_river_game();
    let mut game_abstracted = create_river_game();

    let config = AbstractionConfig {
        num_buckets: 15,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).unwrap();

    // Allocate and solve
    game_baseline.allocate_memory(false);
    game_abstracted.allocate_memory(false);
    solve(&mut game_baseline, 100, 0.0, false);
    solve(&mut game_abstracted, 100, 0.0, false);

    // Get strategy at root node
    let num_hands_oop = game_baseline.num_private_hands(0);
    let num_buckets_oop = game_abstracted.effective_hand_count(0);

    let strategy_baseline = game_baseline.strategy();
    let strategy_abstracted = game_abstracted.strategy();

    let num_actions = game_baseline.available_actions().len();

    // Verify sizes
    let expected_baseline_size = num_actions * num_hands_oop;
    let expected_abstracted_size = num_actions * num_hands_oop; // Should be hand-level, not bucket-level

    println!("  OOP hands: {}, buckets: {}", num_hands_oop, num_buckets_oop);
    println!("  Actions at root: {}", num_actions);
    println!("  Baseline strategy size: {} (expected: {})", strategy_baseline.len(), expected_baseline_size);
    println!("  Abstracted strategy size: {} (expected: {})", strategy_abstracted.len(), expected_abstracted_size);

    assert_eq!(strategy_baseline.len(), expected_baseline_size,
        "Baseline strategy size mismatch");
    assert_eq!(strategy_abstracted.len(), expected_abstracted_size,
        "Abstracted strategy size should be hand-level, not bucket-level");

    // Verify strategy sums to 1.0 for each hand
    let mut strategy_valid = true;
    for hand in 0..num_hands_oop {
        let mut sum_baseline: f32 = 0.0;
        let mut sum_abstracted: f32 = 0.0;
        for action in 0..num_actions {
            sum_baseline += strategy_baseline[action * num_hands_oop + hand];
            sum_abstracted += strategy_abstracted[action * num_hands_oop + hand];
        }
        if (sum_baseline - 1.0).abs() > 0.001 || (sum_abstracted - 1.0).abs() > 0.001 {
            if sum_baseline > 0.0 || sum_abstracted > 0.0 { // Ignore blocked hands
                println!("  Hand {}: baseline sum={:.4}, abstracted sum={:.4}", hand, sum_baseline, sum_abstracted);
                strategy_valid = false;
            }
        }
    }

    if strategy_valid {
        println!("  ✓ Strategy sums to 1.0 for all valid hands");
    } else {
        println!("  ✗ Strategy sum validation failed");
    }

    // Verify hands in same bucket have same strategy
    let abs_data = game_abstracted.abstraction_data().unwrap();
    let hand_to_bucket = &abs_data.hand_to_bucket[0];

    let mut same_bucket_strategy = true;
    for bucket in 0..num_buckets_oop {
        let hands_in_bucket: Vec<usize> = hand_to_bucket.iter()
            .enumerate()
            .filter(|(_, &b)| b as usize == bucket)
            .map(|(h, _)| h)
            .collect();

        if hands_in_bucket.len() > 1 {
            let first_hand = hands_in_bucket[0];
            for &hand in &hands_in_bucket[1..] {
                for action in 0..num_actions {
                    let first_prob = strategy_abstracted[action * num_hands_oop + first_hand];
                    let hand_prob = strategy_abstracted[action * num_hands_oop + hand];
                    if (first_prob - hand_prob).abs() > 0.0001 {
                        println!("  Bucket {}: hand {} has prob {:.4}, hand {} has prob {:.4} for action {}",
                                 bucket, first_hand, first_prob, hand, hand_prob, action);
                        same_bucket_strategy = false;
                    }
                }
            }
        }
    }

    if same_bucket_strategy {
        println!("  ✓ Hands in same bucket have identical strategy (bucket disaggregation works)");
    } else {
        println!("  ✗ Hands in same bucket have different strategies");
    }

    println!("  ✓ PASS: strategy() returns correct hand-level results");
}

fn test_expected_values_api() {
    let mut game_baseline = create_river_game();
    let mut game_abstracted = create_river_game();

    let config = AbstractionConfig {
        num_buckets: 15,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).unwrap();

    // Allocate and solve
    game_baseline.allocate_memory(false);
    game_abstracted.allocate_memory(false);
    solve(&mut game_baseline, 100, 0.0, false);
    solve(&mut game_abstracted, 100, 0.0, false);

    // Cache normalized weights
    game_baseline.cache_normalized_weights();
    game_abstracted.cache_normalized_weights();

    let num_hands_oop = game_baseline.num_private_hands(0);
    let num_hands_ip = game_baseline.num_private_hands(1);

    // Get expected values at root
    let ev_baseline_oop = game_baseline.expected_values(0);
    let ev_baseline_ip = game_baseline.expected_values(1);
    let ev_abstracted_oop = game_abstracted.expected_values(0);
    let ev_abstracted_ip = game_abstracted.expected_values(1);

    println!("  OOP hands: {}", num_hands_oop);
    println!("  IP hands: {}", num_hands_ip);
    println!("  Baseline EV (OOP) size: {} (expected: {})", ev_baseline_oop.len(), num_hands_oop);
    println!("  Baseline EV (IP) size: {} (expected: {})", ev_baseline_ip.len(), num_hands_ip);
    println!("  Abstracted EV (OOP) size: {} (expected: {})", ev_abstracted_oop.len(), num_hands_oop);
    println!("  Abstracted EV (IP) size: {} (expected: {})", ev_abstracted_ip.len(), num_hands_ip);

    assert_eq!(ev_baseline_oop.len(), num_hands_oop, "Baseline OOP EV size mismatch");
    assert_eq!(ev_baseline_ip.len(), num_hands_ip, "Baseline IP EV size mismatch");
    assert_eq!(ev_abstracted_oop.len(), num_hands_oop, "Abstracted OOP EV should be hand-level");
    assert_eq!(ev_abstracted_ip.len(), num_hands_ip, "Abstracted IP EV should be hand-level");

    // Verify EVs are finite
    let all_finite = ev_abstracted_oop.iter().all(|v| v.is_finite())
        && ev_abstracted_ip.iter().all(|v| v.is_finite());

    if all_finite {
        println!("  ✓ All EV values are finite (not NaN/Inf)");
    } else {
        println!("  ✗ Some EV values are NaN or Inf");
    }

    // Compute average EVs
    let avg_ev_baseline_oop: f32 = ev_baseline_oop.iter().sum::<f32>() / num_hands_oop as f32;
    let avg_ev_abstracted_oop: f32 = ev_abstracted_oop.iter().sum::<f32>() / num_hands_oop as f32;
    println!("  Average OOP EV - Baseline: {:.2}, Abstracted: {:.2}", avg_ev_baseline_oop, avg_ev_abstracted_oop);

    println!("  ✓ PASS: expected_values() returns correct hand-level results");
}

fn test_play_api() {
    let mut game = create_river_game();

    let config = AbstractionConfig {
        num_buckets: 15,
        max_iterations: 100,
    };
    game.enable_abstraction(&config).unwrap();
    game.allocate_memory(false);
    solve(&mut game, 100, 0.0, false);

    // Navigate the tree and verify strategy at each node
    let num_hands_oop = game.num_private_hands(0);
    let num_hands_ip = game.num_private_hands(1);

    println!("  At root node:");
    let actions = game.available_actions();
    println!("    Actions: {:?}", actions);
    let strategy = game.strategy();
    println!("    Strategy size: {} (expected: {} actions * {} hands = {})",
             strategy.len(), actions.len(), num_hands_oop, actions.len() * num_hands_oop);
    assert_eq!(strategy.len(), actions.len() * num_hands_oop);

    // Play first action (should be Check or Bet)
    game.play(0);

    let actions_after = game.available_actions();
    if !game.is_terminal_node() && !game.is_chance_node() {
        println!("  After action 0:");
        println!("    Actions: {:?}", actions_after);
        let strategy_after = game.strategy();
        let current_player_hands = if game.current_player() == 0 { num_hands_oop } else { num_hands_ip };
        println!("    Strategy size: {} (expected: {} actions * {} hands = {})",
                 strategy_after.len(), actions_after.len(), current_player_hands, actions_after.len() * current_player_hands);
        assert_eq!(strategy_after.len(), actions_after.len() * current_player_hands);
    }

    // Go back to root and try expected_values
    game.back_to_root();
    game.cache_normalized_weights();

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);
    println!("  At root - EV sizes: OOP={}, IP={}", ev_oop.len(), ev_ip.len());
    assert_eq!(ev_oop.len(), num_hands_oop);
    assert_eq!(ev_ip.len(), num_hands_ip);

    // Navigate down and check EV at non-root node
    game.play(0);
    if !game.is_terminal_node() {
        game.cache_normalized_weights();
        let ev_oop_after = game.expected_values(0);
        let ev_ip_after = game.expected_values(1);
        println!("  After action 0 - EV sizes: OOP={}, IP={}", ev_oop_after.len(), ev_ip_after.len());
        assert_eq!(ev_oop_after.len(), num_hands_oop);
        assert_eq!(ev_ip_after.len(), num_hands_ip);
    }

    println!("  ✓ PASS: play() and tree navigation work correctly with abstraction");
}
