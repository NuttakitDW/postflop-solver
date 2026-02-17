/// Compare strategies: solving from flop vs solving from turn with extracted ranges.
///
/// This test:
/// 1. Solves a small flop game to high accuracy
/// 2. Navigates to check-check on the flop, then a specific turn card
/// 3. Extracts the "realized ranges" (weights) for both players at that point
/// 4. Creates a new turn-only game with those extracted ranges
/// 5. Solves the turn game
/// 6. Compares the strategies at the same nodes
use postflop_solver::*;

fn main() {
    // Use small ranges to keep computation fast
    let oop_range = "QQ-88,AJs+,KQs,AQo+";
    let ip_range = "TT-22,ATs+,KJs+,QJs,AJo+,KQo";

    let flop = flop_from_str("Kh8d4c").unwrap();
    let turn_card = card_from_str("2s").unwrap();

    // Simple bet sizes: 33% pot and all-in
    let bet_sizes = BetSizeOptions::try_from(("33%, a", "60%")).unwrap();

    // ========================================
    // STEP 1: Solve from FLOP
    // ========================================
    println!("=== STEP 1: Solving from FLOP ===");

    let flop_card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let flop_tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 100,
        effective_stack: 200,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        max_raises_per_street: 0,
    };

    let flop_action_tree = ActionTree::new(flop_tree_config.clone()).unwrap();
    let mut flop_game = PostFlopGame::with_config(flop_card_config, flop_action_tree).unwrap();

    let (mem, mem_compressed) = flop_game.memory_usage();
    println!(
        "Flop game memory: {:.2}MB (compressed: {:.2}MB)",
        mem as f64 / 1024.0 / 1024.0,
        mem_compressed as f64 / 1024.0 / 1024.0
    );

    flop_game.allocate_memory(false);

    // Solve to very low exploitability for accurate comparison
    let target_exploit = flop_game.tree_config().starting_pot as f32 * 0.001; // 0.1% of pot
    let exploit = solve(&mut flop_game, 5000, target_exploit, true);
    println!("Flop game exploitability: {:.4}", exploit);

    // ========================================
    // STEP 2: Navigate to check-check + turn card, extract ranges
    // ========================================
    println!("\n=== STEP 2: Extract ranges after flop check-check, turn {} ===",
        card_to_string(turn_card).unwrap());

    flop_game.back_to_root();

    // OOP checks (action 0 should be Check)
    let actions = flop_game.available_actions();
    println!("OOP flop actions: {:?}", actions);
    assert!(matches!(actions[0], Action::Check), "Expected Check as first action for OOP");
    flop_game.play(0); // OOP Check

    // IP checks (action 0 should be Check)
    let actions = flop_game.available_actions();
    println!("IP flop actions: {:?}", actions);
    assert!(matches!(actions[0], Action::Check), "Expected Check as first action for IP");
    flop_game.play(0); // IP Check

    // Should be at chance node (turn)
    assert!(flop_game.is_chance_node(), "Expected chance node after check-check");

    // Deal the turn card
    flop_game.play(turn_card as usize);

    // Now at OOP's decision on the turn
    println!("Board: {:?}", flop_game.current_board().iter()
        .map(|&c| card_to_string(c).unwrap())
        .collect::<Vec<_>>());

    // Extract weights (realized ranges) for both players
    let oop_hands = flop_game.private_cards(0).to_vec();
    let oop_weights = flop_game.weights(0).to_vec();
    let ip_hands = flop_game.private_cards(1).to_vec();
    let ip_weights = flop_game.weights(1).to_vec();

    let oop_nonzero = oop_weights.iter().filter(|&&w| w > 0.0).count();
    let ip_nonzero = ip_weights.iter().filter(|&&w| w > 0.0).count();
    println!("OOP hands with weight > 0: {} / {}", oop_nonzero, oop_hands.len());
    println!("IP hands with weight > 0: {} / {}", ip_nonzero, ip_hands.len());

    // Show some sample weights
    let oop_hand_strs = holes_to_strings(&oop_hands).unwrap();
    println!("\nSample OOP weights after check-check:");
    for (i, (hand, &w)) in oop_hand_strs.iter().zip(oop_weights.iter()).enumerate() {
        if w > 0.0 && i < 20 {
            println!("  {}: {:.4}", hand, w);
        }
    }

    // Get the strategy at the turn root from flop solve (for comparison)
    let flop_turn_actions = flop_game.available_actions();
    println!("\nTurn actions (from flop game): {:?}", flop_turn_actions);

    let flop_turn_strategy = flop_game.strategy();
    let flop_turn_player = flop_game.current_player();
    let flop_turn_num_hands = flop_game.num_private_hands(flop_turn_player);
    let flop_turn_num_actions = flop_turn_actions.len();

    // Create Range objects from extracted weights
    let oop_turn_range = Range::from_hands_weights(&oop_hands, &oop_weights).unwrap();
    let ip_turn_range = Range::from_hands_weights(&ip_hands, &ip_weights).unwrap();

    // ========================================
    // STEP 3: Solve from TURN with extracted ranges
    // ========================================
    println!("\n=== STEP 3: Solving from TURN with extracted ranges ===");

    let turn_card_config = CardConfig {
        range: [oop_turn_range, ip_turn_range],
        flop,
        turn: turn_card,
        river: NOT_DEALT,
    };

    let turn_tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: 100,    // Same pot (no bets on flop)
        effective_stack: 200, // Same stack (no bets on flop)
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        river_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.15,
        merging_threshold: 0.1,
        max_raises_per_street: 0,
    };

    let turn_action_tree = ActionTree::new(turn_tree_config).unwrap();
    let mut turn_game = PostFlopGame::with_config(turn_card_config, turn_action_tree).unwrap();

    let (mem, _) = turn_game.memory_usage();
    println!(
        "Turn game memory: {:.2}MB",
        mem as f64 / 1024.0 / 1024.0
    );

    turn_game.allocate_memory(false);

    let target_exploit_turn = turn_game.tree_config().starting_pot as f32 * 0.001; // 0.1% of pot
    let exploit_turn = solve(&mut turn_game, 5000, target_exploit_turn, true);
    println!("Turn game exploitability: {:.4}", exploit_turn);

    // ========================================
    // STEP 4: Compare strategies
    // ========================================
    println!("\n=== STEP 4: Compare turn root strategies ===");

    turn_game.back_to_root();
    let turn_actions = turn_game.available_actions();
    println!("Turn actions (from turn game): {:?}", turn_actions);

    let turn_strategy = turn_game.strategy();
    let turn_player = turn_game.current_player();
    let turn_num_hands = turn_game.num_private_hands(turn_player);
    let turn_num_actions = turn_actions.len();
    let turn_hand_strs = holes_to_strings(turn_game.private_cards(turn_player)).unwrap();

    assert_eq!(flop_turn_num_actions, turn_num_actions, "Action count mismatch");
    println!(
        "Actions match: {} actions for both games",
        flop_turn_num_actions
    );

    // Build hand-to-index mapping for the flop game (at turn node)
    let flop_hand_strs = holes_to_strings(flop_game.private_cards(flop_turn_player)).unwrap();
    let flop_hand_map: std::collections::HashMap<&str, usize> = flop_hand_strs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    // Compare strategies hand-by-hand
    let mut total_diff = 0.0f64;
    let mut max_diff = 0.0f64;
    let mut max_diff_hand = String::new();
    let mut compared_count = 0;

    println!("\nPer-hand strategy comparison (showing hands with diff > 1%):");
    println!("{:<8} | {}", "Hand", (0..turn_num_actions)
        .map(|a| format!("Flop_{:?} / Turn_{:?}", flop_turn_actions[a], turn_actions[a]))
        .collect::<Vec<_>>()
        .join(" | "));
    println!("{}", "-".repeat(80));

    for (turn_idx, hand_str) in turn_hand_strs.iter().enumerate() {
        // Find this hand in flop game
        if let Some(&flop_idx) = flop_hand_map.get(hand_str.as_str()) {
            // Skip hands with zero weight (not in range)
            let flop_weight = flop_game.weights(flop_turn_player)[flop_idx];
            let turn_weight = turn_game.weights(turn_player)[turn_idx];
            if flop_weight == 0.0 && turn_weight == 0.0 {
                continue;
            }

            compared_count += 1;
            let mut hand_diff = 0.0f64;
            let mut action_strs = Vec::new();

            for a in 0..turn_num_actions {
                let flop_prob = flop_turn_strategy[a * flop_turn_num_hands + flop_idx];
                let turn_prob = turn_strategy[a * turn_num_hands + turn_idx];
                let diff = (flop_prob - turn_prob).abs() as f64;
                hand_diff += diff;
                action_strs.push(format!("{:.3} / {:.3}", flop_prob, turn_prob));
            }

            total_diff += hand_diff;
            if hand_diff > max_diff {
                max_diff = hand_diff;
                max_diff_hand = hand_str.clone();
            }

            if hand_diff > 0.01 {
                println!("{:<8} | {}", hand_str, action_strs.join(" | "));
            }
        }
    }

    let avg_diff = total_diff / compared_count.max(1) as f64;
    println!("\n=== SUMMARY ===");
    println!("Compared {} hands", compared_count);
    println!("Average strategy diff per hand: {:.6}", avg_diff);
    println!("Max strategy diff: {:.6} (hand: {})", max_diff, max_diff_hand);
    println!(
        "Conclusion: strategies are {}",
        if max_diff < 0.02 {
            "VERY CLOSE (< 2% max diff)"
        } else if max_diff < 0.05 {
            "CLOSE (< 5% max diff)"
        } else {
            "DIFFERENT (> 5% max diff)"
        }
    );

    // ========================================
    // STEP 5: Also compare deeper in the tree (after turn bet-call)
    // ========================================
    println!("\n=== STEP 5: Compare deeper node (after OOP bet on turn) ===");

    // Find a non-check action index (bet or all-in)
    let bet_action_idx = flop_turn_actions
        .iter()
        .position(|a| !matches!(a, Action::Check))
        .expect("Expected a non-check action on turn");

    println!("Playing action {} ({:?}) in both games...", bet_action_idx, flop_turn_actions[bet_action_idx]);

    // Play bet in flop game (already at turn root)
    flop_game.play(bet_action_idx);
    let flop_ip_actions = flop_game.available_actions();
    let flop_ip_strategy = flop_game.strategy();
    let flop_ip_player = flop_game.current_player();
    let flop_ip_num_hands = flop_game.num_private_hands(flop_ip_player);

    // Play bet in turn game
    turn_game.play(bet_action_idx);
    let turn_ip_actions = turn_game.available_actions();
    let turn_ip_strategy = turn_game.strategy();
    let turn_ip_player = turn_game.current_player();
    let turn_ip_num_hands = turn_game.num_private_hands(turn_ip_player);

    println!("IP actions from flop game: {:?}", flop_ip_actions);
    println!("IP actions from turn game: {:?}", turn_ip_actions);
    assert_eq!(flop_ip_actions.len(), turn_ip_actions.len(), "IP action count mismatch");

    // Compare IP's strategy
    let flop_ip_hand_strs = holes_to_strings(flop_game.private_cards(flop_ip_player)).unwrap();
    let turn_ip_hand_strs = holes_to_strings(turn_game.private_cards(turn_ip_player)).unwrap();

    let flop_ip_hand_map: std::collections::HashMap<&str, usize> = flop_ip_hand_strs
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut total_diff_ip = 0.0f64;
    let mut max_diff_ip = 0.0f64;
    let mut compared_ip = 0;

    for (turn_idx, hand_str) in turn_ip_hand_strs.iter().enumerate() {
        if let Some(&flop_idx) = flop_ip_hand_map.get(hand_str.as_str()) {
            let flop_w = flop_game.weights(flop_ip_player)[flop_idx];
            let turn_w = turn_game.weights(turn_ip_player)[turn_idx];
            if flop_w == 0.0 && turn_w == 0.0 {
                continue;
            }

            compared_ip += 1;
            let mut hand_diff = 0.0f64;
            for a in 0..flop_ip_actions.len() {
                let flop_prob = flop_ip_strategy[a * flop_ip_num_hands + flop_idx];
                let turn_prob = turn_ip_strategy[a * turn_ip_num_hands + turn_idx];
                hand_diff += (flop_prob - turn_prob).abs() as f64;
            }
            total_diff_ip += hand_diff;
            if hand_diff > max_diff_ip {
                max_diff_ip = hand_diff;
            }
        }
    }

    let avg_diff_ip = total_diff_ip / compared_ip.max(1) as f64;
    println!("\nIP strategy after OOP turn bet:");
    println!("Compared {} hands", compared_ip);
    println!("Average strategy diff per hand: {:.6}", avg_diff_ip);
    println!("Max strategy diff: {:.6}", max_diff_ip);

    // ========================================
    // STEP 6: Compare EV
    // ========================================
    println!("\n=== STEP 6: Compare EV at turn root ===");

    // Go back to turn root in both games
    flop_game.apply_history(&[0, 0, turn_card as usize]); // check, check, turn card
    turn_game.back_to_root();

    flop_game.cache_normalized_weights();
    turn_game.cache_normalized_weights();

    let flop_ev_oop = flop_game.expected_values(0);
    let turn_ev_oop = turn_game.expected_values(0);

    let flop_oop_weights = flop_game.normalized_weights(0);
    let turn_oop_weights = turn_game.normalized_weights(0);

    let flop_avg_ev = compute_average(&flop_ev_oop, flop_oop_weights);
    let turn_avg_ev = compute_average(&turn_ev_oop, turn_oop_weights);

    println!("OOP average EV (flop game): {:.4}", flop_avg_ev);
    println!("OOP average EV (turn game): {:.4}", turn_avg_ev);
    println!("EV difference: {:.4}", (flop_avg_ev - turn_avg_ev).abs());

    println!("\n=== DONE ===");
}
