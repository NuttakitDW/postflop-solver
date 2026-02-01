use postflop_solver::*;
use std::time::Instant;

fn main() {
    println!("=== PDCFR+ Benchmark (medium.json settings) ===");
    println!("Threads: {}", rayon::current_num_threads());
    println!();

    // Ranges from medium.json
    let oop_range = "66:0.09,55:0.765,44:0.91,33:0.715,22:0.525,ATs-A2s,AJo:0.236,ATo:0.135,A9o:0.78,A8o:0.98,A7o:0.705,A6o:0.74,A5o:0.675,A4o:0.7,A3o:0.835,A2o:0.655,KJs:0.32,KTs-K2s,KQo:0.555,KJo:0.76,KTo:0.825,K9o:0.975,K8o:0.985,K7o:0.665,K6o:0.47,K5o:0.73,K4o-K3o:0.975,K2o,QJs:0.99,QTs-Q6s,Q5s:0.495,Q4s,Q3s:0.925,Q2s,QJo:0.73,QTo:0.89500004,Q9o:0.865,Q8o:0.84,Q7o:0.905,Q6o:0.67,Q5o:0.9,Q4o:0.97,Q3o,Q2o:0.875,J8s+,J7s:0.395,J6s-J2s,JTo:0.98,J9o:0.815,J8o:0.77,J7o:0.99,J6o-J5o,J4o:0.645,T8s+,T7s:0.35,T6s:0.365,T5s-T2s,T9o:0.785,T8o:0.76,T7o:0.985,T6o,98s:0.36,97s:0.325,96s:0.47,95s-92s,98o:0.92,97o-96o,87s:0.475,86s:0.52,85s-82s,86o+,76s:0.71,75s-72s,75o+,62s+,65o,64o:0.995,52s+,53o+,42s+,43o:0.255,32s";
    let ip_range = "22+,A2+,K2s+,K6o+,K5o:0.36,Q2s+,Q8o+,Q7o:0.97,J2s+,J8o+,T3s+,T8o+,T7o:0.36,95s+,98o,97o:0.52,85s+,87o,74s+,64s+,53s+";

    // Parse ranges
    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    // Card config - Td9d6h flop
    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    // Bet sizes from medium.json
    let oop_flop_bet = BetSizeOptions::try_from(("33%", "")).unwrap();
    let oop_flop_raise = BetSizeOptions::try_from(("", "33%, 60%")).unwrap();
    let ip_flop_bet = BetSizeOptions::try_from(("25%, 55%, 85%, 125%", "")).unwrap();
    let ip_flop_raise = BetSizeOptions::try_from(("", "33%, 60%")).unwrap();

    let turn_bet = BetSizeOptions::try_from(("33%, 75%, 125%", "33%, 60%")).unwrap();
    let river_bet = BetSizeOptions::try_from(("33%, 75%, 125%", "33%, 60%")).unwrap();

    // Tree config from medium.json
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 61,
        effective_stack: 477,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_flop_bet.clone(), ip_flop_bet],
        turn_bet_sizes: [turn_bet.clone(), turn_bet.clone()],
        river_bet_sizes: [river_bet.clone(), river_bet],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 6.0,
        force_allin_threshold: 0.5,
        merging_threshold: 0.1,
        max_raises_per_street: 4,
    };

    // Build action tree
    println!("Building action tree...");
    let tree_start = Instant::now();
    let action_tree = ActionTree::new(tree_config).expect("Failed to create action tree");
    println!("Tree built in {:.2}s", tree_start.elapsed().as_secs_f64());

    // Create game
    println!("Creating game...");
    let game_start = Instant::now();
    let mut game = PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game");
    println!("Game created in {:.2}s", game_start.elapsed().as_secs_f64());

    // Enable PDCFR+
    game.set_predictive_mode(true);
    println!("PDCFR+ mode: ENABLED");

    // Print game info
    println!();
    println!("=== Game Info ===");
    println!("OOP hands: {}", game.private_cards(0).len());
    println!("IP hands: {}", game.private_cards(1).len());
    let (mem_uncompressed, mem_compressed) = game.memory_usage();
    println!("Memory (uncompressed): {:.2} MB", mem_uncompressed as f64 / 1024.0 / 1024.0);
    println!("Memory (compressed): {:.2} MB", mem_compressed as f64 / 1024.0 / 1024.0);
    println!();

    // Allocate memory with compression (as per medium.json)
    println!("Allocating memory (compressed)...");
    let alloc_start = Instant::now();
    game.allocate_memory(true);
    println!("Memory allocated in {:.2}s", alloc_start.elapsed().as_secs_f64());
    println!();

    // Solve parameters from medium.json
    let max_iterations = 1000;
    let target_exploitability = 61.0 * 0.003; // 0.3% of pot

    println!("=== Solving with PDCFR+ ===");
    println!("Max iterations: {}", max_iterations);
    println!("Target exploitability: {:.4} ({:.1}% of pot)", target_exploitability, 0.3);
    println!();

    // Solve
    let solve_start = Instant::now();
    let exploitability = solve(&mut game, max_iterations, target_exploitability, true);
    let solve_time = solve_start.elapsed();

    // Results
    println!();
    println!("=== Results ===");
    println!("Solve time: {:.2}s", solve_time.as_secs_f64());
    println!("Final exploitability: {:.4}", exploitability);
    println!("Exploitability as % of pot: {:.3}%", exploitability / 61.0 * 100.0);
}
