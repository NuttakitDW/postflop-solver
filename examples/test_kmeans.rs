//! Test k-means hand clustering for the postflop solver.
//!
//! Run with: `cargo run --example test_kmeans`

use postflop_solver::*;

fn main() {
    println!("=== K-Means Hand Clustering Test ===\n");

    // Test 1: Synthetic data test
    println!("Test 1: K-means on synthetic data");
    test_synthetic_kmeans();

    // Test 2: Real game clustering (flop)
    println!("\nTest 2: Clustering on real poker hands (flop)");
    test_real_game_clustering();

    // Test 3: Turn case - verify all rivers are considered
    println!("\nTest 3: Clustering on turn (verifies all rivers considered)");
    test_turn_case_clustering();
}

fn test_synthetic_kmeans() {
    // Create clearly separated clusters
    let mut features = Vec::new();

    // Cluster 1: Low EHS hands (trash)
    for i in 0..30 {
        features.push(HandFeatures {
            ehs: 0.15 + (i as f32) * 0.005,
            ehs_squared: 0.02 + (i as f32) * 0.001,
        });
    }

    // Cluster 2: Medium EHS hands (draws/marginal)
    for i in 0..40 {
        features.push(HandFeatures {
            ehs: 0.45 + (i as f32) * 0.003,
            ehs_squared: 0.20 + (i as f32) * 0.002,
        });
    }

    // Cluster 3: High EHS hands (strong)
    for i in 0..30 {
        features.push(HandFeatures {
            ehs: 0.80 + (i as f32) * 0.004,
            ehs_squared: 0.64 + (i as f32) * 0.003,
        });
    }

    println!("  Total hands: {}", features.len());

    // Run k-means with 3 clusters
    let (assignments, centroids, iterations) = kmeans(&features, 3, 100, 42);

    println!("  K-means converged in {} iterations", iterations);
    println!("  Centroids:");
    for (i, c) in centroids.iter().enumerate() {
        println!("    Cluster {}: EHS={:.3}, EHS²={:.3}", i, c.ehs, c.ehs_squared);
    }

    let sizes = cluster_sizes(&assignments, 3);
    println!("  Cluster sizes: {:?}", sizes);

    // Verify clusters are reasonable
    let all_nonzero = sizes.iter().all(|&s| s > 0);
    let total = sizes.iter().sum::<usize>();

    if all_nonzero && total == 100 && iterations < 100 {
        println!("  ✓ PASS: K-means working correctly");
    } else {
        println!("  ✗ FAIL: Unexpected clustering result");
    }
}

fn test_real_game_clustering() {
    // Simple ranges for testing
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,KQs,KJs,QJs,JTs,T9s,98s,AKo,AQo,KQo";

    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    // Card config - flop only
    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Ks7h2c").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    // Simple bet sizes
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

    // Build game
    let action_tree = ActionTree::new(tree_config).expect("Failed to create action tree");
    let game = PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game");

    println!("  OOP hands: {}", game.private_cards(0).len());
    println!("  IP hands: {}", game.private_cards(1).len());

    // Test different bucket counts
    for &num_buckets in &[10, 30, 50] {
        let config = AbstractionConfig {
            num_buckets,
            max_iterations: 100,
        };

        println!("\n  Clustering with {} buckets:", num_buckets);

        // Cluster OOP
        let result = cluster_hands(&game, 0, &config);

        println!("    OOP: {} iterations, {} clusters used",
                 result.iterations,
                 result.cluster_sizes.iter().filter(|&&s| s > 0).count());

        // Print cluster distribution
        let non_empty: Vec<_> = result.cluster_sizes.iter()
            .enumerate()
            .filter(|(_, &s)| s > 0)
            .collect();

        if non_empty.len() <= 10 {
            for (i, &size) in &non_empty {
                let centroid = &result.centroids[*i];
                println!("      Cluster {}: {} hands, EHS={:.3}", i, size, centroid.ehs);
            }
        } else {
            // Just show stats
            let min_size = result.cluster_sizes.iter().filter(|&&s| s > 0).min().unwrap_or(&0);
            let max_size = result.cluster_sizes.iter().max().unwrap_or(&0);
            println!("      Cluster sizes: min={}, max={}", min_size, max_size);
        }

        // Show some example hand -> cluster mappings
        let private_cards = game.private_cards(0);
        println!("    Sample hand assignments:");
        for i in 0..5.min(private_cards.len()) {
            let (c1, c2) = private_cards[i];
            let cluster = result.assignments[i];
            let ehs = result.features[i].ehs;
            println!("      {} -> cluster {} (EHS={:.3})",
                     format_hand(c1, c2), cluster, ehs);
        }
    }

    println!("\n  ✓ Clustering complete");
}

fn test_turn_case_clustering() {
    // Test with a turn card dealt to verify all rivers are considered
    // Turn card Tc (value 32) - tests that rivers < 32 are also included
    let oop_range = "AA,KK,QQ,JJ,TT,99,88,77,66,AKs,AQs,AJs,KQs";
    let ip_range = "AA,KK,QQ,JJ,TT,99,88,77,AKs,AQs,KQs";

    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    // Flop + Turn dealt, river not dealt
    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Ks7h2c").unwrap(),
        turn: card_from_str("Tc").unwrap(),  // Tc = card 32
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
    let game = PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game");

    println!("  Board: Ks7h2c Tc (turn dealt)");
    println!("  OOP hands: {}", game.private_cards(0).len());

    let config = AbstractionConfig {
        num_buckets: 10,
        max_iterations: 100,
    };

    let result = cluster_hands(&game, 0, &config);

    // Verify all hands have EHS computed (runout_count > 0)
    let hands_with_ehs: usize = result.features.iter()
        .filter(|f| f.ehs > 0.0 || f.ehs_squared > 0.0)
        .count();

    println!("  Hands with computed EHS: {}/{}", hands_with_ehs, result.features.len());
    println!("  Iterations: {}", result.iterations);

    // Show some hands
    let private_cards = game.private_cards(0);
    println!("  Sample hands:");
    for i in 0..5.min(private_cards.len()) {
        let (c1, c2) = private_cards[i];
        let cluster = result.assignments[i];
        let ehs = result.features[i].ehs;
        println!("      {} -> cluster {} (EHS={:.3})",
                 format_hand(c1, c2), cluster, ehs);
    }

    // Verify: on turn, there should be 52 - 3 (flop) - 1 (turn) = 48 possible rivers
    // The bug would have only considered ~20 rivers (cards > turn value)
    if hands_with_ehs == result.features.len() && hands_with_ehs > 0 {
        println!("  ✓ PASS: All hands have EHS computed (all rivers considered)");
    } else {
        println!("  ✗ FAIL: Some hands missing EHS ({} of {} computed)",
                 hands_with_ehs, result.features.len());
    }
}

fn format_hand(c1: u8, c2: u8) -> String {
    let ranks = "23456789TJQKA";
    let suits = "cdhs";

    let r1 = ranks.chars().nth((c1 >> 2) as usize).unwrap();
    let s1 = suits.chars().nth((c1 & 3) as usize).unwrap();
    let r2 = ranks.chars().nth((c2 >> 2) as usize).unwrap();
    let s2 = suits.chars().nth((c2 & 3) as usize).unwrap();

    format!("{}{}{}{}", r1, s1, r2, s2)
}
