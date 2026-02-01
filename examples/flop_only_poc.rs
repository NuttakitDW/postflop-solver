use postflop_solver::*;
use std::time::Instant;

fn main() {
    println!("=== Flop-Only Solving POC ===\n");

    // ranges of OOP and IP
    let oop_range = "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s";
    let ip_range = "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+";

    let card_config = CardConfig {
        range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("60%, e, a", "2.5x")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 100,
        effective_stack: 450,
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

    // ==========================================
    // FLOP-ONLY SOLVE
    // ==========================================
    println!("--- FLOP-ONLY SOLVE ---");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_flop_only = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();

    // Enable flop-only mode
    game_flop_only.set_solve_flop_only(true);
    game_flop_only.allocate_memory(false);

    let max_iterations = 500;
    let target_exploitability = game_flop_only.tree_config().starting_pot as f32 * 0.005;

    let start = Instant::now();
    let exploitability_flop = solve(&mut game_flop_only, max_iterations, target_exploitability, false);
    let duration_flop = start.elapsed();

    println!("  Time: {:?}", duration_flop);
    println!("  Exploitability: {:.4}", exploitability_flop);

    // ==========================================
    // FULL SOLVE (for comparison)
    // ==========================================
    println!("\n--- FULL SOLVE (for comparison) ---");
    let action_tree = ActionTree::new(tree_config).unwrap();
    let mut game_full = PostFlopGame::with_config(card_config, action_tree).unwrap();

    let (mem_usage_full, _) = game_full.memory_usage();
    println!("  Memory required: {:.2} MB", mem_usage_full as f64 / (1024.0 * 1024.0));

    game_full.allocate_memory(false);

    let start = Instant::now();
    let exploitability_full = solve(&mut game_full, max_iterations, target_exploitability, false);
    let duration_full = start.elapsed();

    println!("  Time: {:?}", duration_full);
    println!("  Exploitability: {:.4}", exploitability_full);

    // ==========================================
    // COMPARISON
    // ==========================================
    println!("\n--- COMPARISON ---");
    println!("  Speedup: {:.1}x faster", duration_full.as_secs_f64() / duration_flop.as_secs_f64());

    println!("\n=== POC Complete ===");
    println!("\nNote: Flop-only uses 50% equity for non-showdown terminals.");
    println!("This is a POC - accurate equity computation would improve results.");
}
