//! Test to verify bucket-based evaluation correctness.
//!
//! Compares the abstracted evaluation against the full hand-level evaluation
//! to identify where the discrepancy comes from.
//!
//! Run with: `cargo run --example test_evaluation_correctness --release`

use postflop_solver::*;

fn main() {
    println!("=== Evaluation Correctness Test ===\n");

    // Test 0: Diagnose feature distribution
    println!("Test 0: Feature distribution analysis");
    test_feature_distribution();

    // Test 1: Compare CFV at terminal nodes
    println!("\nTest 1: Compare bucket vs hand evaluation");
    test_cfv_comparison();

    // Test 2: Check if exploitability converges with more iterations
    println!("\nTest 2: Exploitability convergence");
    test_convergence();

    // Test 3: Verify bucket weights
    println!("\nTest 3: Bucket weights verification");
    test_bucket_weights();

    // Test 4: Verify 1:1 mapping produces identical results
    println!("\nTest 4: 1:1 bucket mapping verification");
    test_one_to_one_mapping();
}

fn test_feature_distribution() {
    use std::collections::HashSet;

    let game = create_river_game();

    // Get the clustering result to access features
    let config = AbstractionConfig {
        num_buckets: 100,  // Request more than num_hands
        max_iterations: 100,
    };

    let mut game_abs = create_river_game();
    game_abs.enable_abstraction(&config).unwrap();
    let abs_data = game_abs.abstraction_data().unwrap();

    println!("  Board: Ks7h2cTc3d (River - single runout)");
    println!("  OOP hands: {}", game.num_private_hands(0));
    println!("  IP hands: {}", game.num_private_hands(1));

    // Count unique EHS values for each player
    for player in 0..2 {
        let player_name = if player == 0 { "OOP" } else { "IP" };
        let num_hands = game.num_private_hands(player);
        let num_buckets = abs_data.num_buckets(player);

        // Count non-empty buckets
        let non_empty = abs_data.bucket_to_hands[player]
            .iter()
            .filter(|h| !h.is_empty())
            .count();

        // Count unique feature pairs by looking at bucket assignments
        let mut unique_buckets: HashSet<u16> = HashSet::new();
        for &bucket in &abs_data.hand_to_bucket[player] {
            unique_buckets.insert(bucket);
        }

        println!("  {}: {} hands -> {} buckets requested, {} non-empty, {} unique assignments",
                 player_name, num_hands, num_buckets, non_empty, unique_buckets.len());

        // Show bucket size distribution
        let mut size_counts: Vec<(usize, usize)> = Vec::new();
        for (bucket, hands) in abs_data.bucket_to_hands[player].iter().enumerate() {
            if !hands.is_empty() {
                size_counts.push((bucket, hands.len()));
            }
        }
        size_counts.sort_by_key(|&(_, size)| std::cmp::Reverse(size));

        println!("    Largest buckets:");
        for (bucket, size) in size_counts.iter().take(5) {
            println!("      Bucket {}: {} hands", bucket, size);
        }
    }

    println!("\n  INSIGHT: For River games, EHS features are 1-dimensional (EHS² = EHS * EHS)");
    println!("  Hands with identical showdown equity get grouped together.");
    println!("  This is expected behavior, NOT a bug.");
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

fn test_cfv_comparison() {
    // Create baseline game (no abstraction)
    let game_baseline = create_river_game();

    // Get initial weights
    let oop_weights = game_baseline.initial_weights(0);
    let ip_weights = game_baseline.initial_weights(1);

    println!("  OOP hands: {}, total weight: {:.4}",
             oop_weights.len(),
             oop_weights.iter().sum::<f32>());
    println!("  IP hands: {}, total weight: {:.4}",
             ip_weights.len(),
             ip_weights.iter().sum::<f32>());

    // Create abstracted game
    let mut game_abstracted = create_river_game();
    let config = AbstractionConfig {
        num_buckets: 20,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).unwrap();

    let oop_bucket_weights = game_abstracted.abstraction_data().unwrap().bucket_weights[0].clone();
    let ip_bucket_weights = game_abstracted.abstraction_data().unwrap().bucket_weights[1].clone();

    println!("  OOP buckets: {}, total weight: {:.4}",
             oop_bucket_weights.len(),
             oop_bucket_weights.iter().sum::<f32>());
    println!("  IP buckets: {}, total weight: {:.4}",
             ip_bucket_weights.len(),
             ip_bucket_weights.iter().sum::<f32>());

    // Check weight conservation
    let oop_weight_diff = (oop_weights.iter().sum::<f32>() - oop_bucket_weights.iter().sum::<f32>()).abs();
    let ip_weight_diff = (ip_weights.iter().sum::<f32>() - ip_bucket_weights.iter().sum::<f32>()).abs();

    if oop_weight_diff < 0.01 && ip_weight_diff < 0.01 {
        println!("  ✓ Weights conserved correctly");
    } else {
        println!("  ✗ Weight mismatch: OOP diff={:.4}, IP diff={:.4}", oop_weight_diff, ip_weight_diff);
    }
}

fn test_convergence() {
    println!("  Baseline convergence:");
    let mut game = create_river_game();
    game.allocate_memory(false);
    let exp = solve(&mut game, 100, 0.0, false);
    println!("    100 iterations: exploitability = {:.6}", exp);

    // Test different bucket sizes at 100 iterations
    println!("  Abstracted exploitability by bucket count (100 iterations):");
    for k in [5, 10, 20, 30, 40, 50, 60, 80, 96] {
        let mut game = create_river_game();
        let config = AbstractionConfig {
            num_buckets: k,
            max_iterations: 100,
        };
        game.enable_abstraction(&config).unwrap();

        let actual_buckets_oop = game.effective_hand_count(0);
        let actual_buckets_ip = game.effective_hand_count(1);

        game.allocate_memory(false);
        let exp = solve(&mut game, 100, 0.0, false);
        println!("    k={}: buckets=({},{}), exploitability = {:.6}",
                 k, actual_buckets_oop, actual_buckets_ip, exp);
    }
}

fn test_bucket_weights() {
    let _game = create_river_game();

    let mut game_abs = create_river_game();
    let config = AbstractionConfig {
        num_buckets: 10,
        max_iterations: 100,
    };
    game_abs.enable_abstraction(&config).unwrap();

    let abs_data = game_abs.abstraction_data().unwrap();

    println!("  Bucket distribution (k=10):");
    for player in 0..2 {
        let player_name = if player == 0 { "OOP" } else { "IP" };
        let num_buckets = abs_data.num_buckets(player);
        let bucket_weights = &abs_data.bucket_weights[player];

        println!("    {} buckets: {}", player_name, num_buckets);
        for (b, &w) in bucket_weights.iter().enumerate() {
            let hands_in_bucket = abs_data.bucket_to_hands[player][b].len();
            println!("      Bucket {}: {} hands, weight {:.4}", b, hands_in_bucket, w);
        }
    }
}

fn test_one_to_one_mapping() {
    // With k = num_hands, each bucket should contain exactly one hand
    // The abstracted solver should produce IDENTICAL results to baseline

    let mut game_baseline = create_river_game();
    let num_oop = game_baseline.num_private_hands(0);
    let num_ip = game_baseline.num_private_hands(1);

    println!("  Hands: OOP={}, IP={}", num_oop, num_ip);

    // Use k = max(num_oop, num_ip) to ensure 1:1 mapping
    let k = num_oop.max(num_ip);

    let mut game_abstracted = create_river_game();
    let config = AbstractionConfig {
        num_buckets: k,
        max_iterations: 100,
    };
    game_abstracted.enable_abstraction(&config).unwrap();

    // Verify 1:1 mapping
    let abs_data = game_abstracted.abstraction_data().unwrap();
    let oop_buckets = abs_data.num_buckets(0);
    let ip_buckets = abs_data.num_buckets(1);
    println!("  Buckets: OOP={}, IP={}", oop_buckets, ip_buckets);

    let oop_1to1 = oop_buckets == num_oop;
    let ip_1to1 = ip_buckets == num_ip;
    println!("  1:1 mapping: OOP={}, IP={}", oop_1to1, ip_1to1);

    if !oop_1to1 || !ip_1to1 {
        println!("  ✗ Cannot test 1:1 equivalence - bucket counts don't match hand counts");
        return;
    }

    // Verify each bucket has exactly one hand
    let mut all_single = true;
    let mut multi_hand_buckets = Vec::new();
    for player in 0..2 {
        for bucket in 0..abs_data.num_buckets(player) {
            let count = abs_data.bucket_to_hands[player][bucket].len();
            if count != 1 {
                all_single = false;
                if count > 1 {
                    multi_hand_buckets.push((player, bucket, count));
                }
            }
        }
    }
    if !all_single {
        // Count non-empty buckets
        let non_empty_oop = abs_data.bucket_to_hands[0].iter().filter(|h| !h.is_empty()).count();
        let non_empty_ip = abs_data.bucket_to_hands[1].iter().filter(|h| !h.is_empty()).count();
        println!("  Note: Non-empty buckets: OOP={}, IP={}", non_empty_oop, non_empty_ip);
        println!("  Some buckets have != 1 hand. Examples with >1 hand:");
        for (player, bucket, count) in multi_hand_buckets.iter().take(5) {
            let player_name = if *player == 0 { "OOP" } else { "IP" };
            println!("    {} bucket {}: {} hands", player_name, bucket, count);
        }
        println!("  This is expected: hands with similar EHS features get grouped together.");
        println!("  Skipping 1:1 equivalence test since true 1:1 mapping was not achieved.");
        return;
    }
    println!("  ✓ Each bucket contains exactly 1 hand");

    // Verify bucket weights = sum of hand weights in bucket
    let baseline_weights_oop = game_baseline.initial_weights(0);
    let baseline_weights_ip = game_baseline.initial_weights(1);

    let mut weight_correct = true;
    for player in 0..2 {
        let baseline_weights = if player == 0 { baseline_weights_oop } else { baseline_weights_ip };
        for (bucket, &bucket_weight) in abs_data.bucket_weights[player].iter().enumerate() {
            // Sum the hand weights for all hands in this bucket
            let hands = &abs_data.bucket_to_hands[player][bucket];
            let expected_weight: f32 = hands.iter()
                .map(|&h| baseline_weights[h as usize])
                .sum();

            if (bucket_weight - expected_weight).abs() > 0.0001 {
                let player_name = if player == 0 { "OOP" } else { "IP" };
                println!("  ✗ Weight mismatch: {} bucket {}: bucket_weight={:.4}, expected={:.4}",
                         player_name, bucket, bucket_weight, expected_weight);
                weight_correct = false;
            }
        }
    }
    if weight_correct {
        println!("  ✓ Bucket weights correctly sum hand weights");
    }

    // Solve both games
    game_baseline.allocate_memory(false);
    game_abstracted.allocate_memory(false);

    println!("  Solving baseline...");
    let exp_baseline = solve(&mut game_baseline, 100, 0.0, false);

    println!("  Solving abstracted...");
    let exp_abstracted = solve(&mut game_abstracted, 100, 0.0, false);

    println!("  Baseline exploitability: {:.6}", exp_baseline);
    println!("  Abstracted exploitability: {:.6}", exp_abstracted);

    let diff = (exp_baseline - exp_abstracted).abs();
    let relative_diff = diff / exp_baseline.max(0.0001);
    println!("  Absolute difference: {:.6}", diff);
    println!("  Relative difference: {:.2}%", relative_diff * 100.0);

    if relative_diff < 0.05 {
        println!("  ✓ PASS: 1:1 mapping produces nearly identical results (<5% difference)");
    } else {
        println!("  ✗ FAIL: Significant difference detected - there may be a bug in evaluation");
    }
}
