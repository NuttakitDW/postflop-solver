//! Benchmark with abstraction enabled for comparison with baseline.
//!
//! Run with: cargo run --example benchmark_abstraction --release
//!
//! Or with specific bucket count:
//!   cargo run --example benchmark_abstraction --release -- 50

use postflop_solver::*;
use std::env;
use std::time::Instant;

fn main() {
    // Parse bucket count from args, default to 50
    let args: Vec<String> = env::args().collect();
    let num_buckets: usize = if args.len() > 1 {
        args[1].parse().unwrap_or(50)
    } else {
        50
    };

    println!("=== Postflop Solver Benchmark (ABSTRACTION) ===");
    println!("Threads: {} (set RAYON_NUM_THREADS to change)", rayon::current_num_threads());
    println!("Buckets: {}", num_buckets);
    println!();

    // OOP range from GTO Wizard 50bb config
    let oop_range = "66:0.09,55:0.766,44:0.91,33:0.716,22:0.526,KJs:0.32,K9s-K2s,KJo:0.76,K9o:0.976,K8o:0.986,K7o:0.666,K6o:0.47,K5o:0.73,K4o-K3o:0.976,K2o,J9s-J8s,J7s:0.396,J6s-J2s,J9o:0.816,J8o:0.77,J7o:0.99,J6o-J5o,J4o:0.646,98s:0.36,97s:0.326,96s:0.47,95s-92s,98o:0.92,97o-96o,87s:0.476,86s:0.52,85s-82s,86o+,76s:0.71,75s-72s,75o+,62s+,65o,64o:0.996,52s+,53o+,42s+,43o:0.256,32s,AsJh:0.236,AsJd:0.236,AsJc:0.236,AdJs:0.236,AdJh:0.236,AdJc:0.236,AcJs:0.236,AcJh:0.236,AcJd:0.236,AsTs,AdTd,AsTh:0.136,AsTd:0.136,AdTs:0.136,AdTh:0.136,AcTs:0.136,AcTh:0.136,AcTd:0.136,As9s,Ad9d,Ac9c,As9h:0.78,As9d:0.78,As9c:0.78,Ad9s:0.78,Ad9h:0.78,Ad9c:0.78,Ac9s:0.78,Ac9h:0.78,Ac9d:0.78,As8s,Ad8d,Ac8c,As8h:0.98,As8d:0.98,As8c:0.98,Ad8s:0.98,Ad8h:0.98,Ad8c:0.98,Ac8s:0.98,Ac8h:0.98,Ac8d:0.98,As7s,Ad7d,Ac7c,As7h:0.706,As7d:0.706,As7c:0.706,Ad7s:0.706,Ad7h:0.706,Ad7c:0.706,Ac7s:0.706,Ac7h:0.706,Ac7d:0.706,As6s,Ad6d,Ac6c,As6h:0.74,As6d:0.74,As6c:0.74,Ad6s:0.74,Ad6h:0.74,Ad6c:0.74,Ac6s:0.74,Ac6h:0.74,Ac6d:0.74,As5s,Ad5d,Ac5c,As5h:0.676,As5d:0.676,As5c:0.676,Ad5s:0.676,Ad5h:0.676,Ad5c:0.676,Ac5s:0.676,Ac5h:0.676,Ac5d:0.676,As4s,Ad4d,Ac4c,As4h:0.7,As4d:0.7,As4c:0.7,Ad4s:0.7,Ad4h:0.7,Ad4c:0.7,Ac4s:0.7,Ac4h:0.7,Ac4d:0.7,As3s,Ad3d,Ac3c,As3h:0.836,As3d:0.836,As3c:0.836,Ad3s:0.836,Ad3h:0.836,Ad3c:0.836,Ac3s:0.836,Ac3h:0.836,Ac3d:0.836,As2s,Ad2d,Ac2c,As2h:0.656,As2d:0.656,As2c:0.656,Ad2s:0.656,Ad2h:0.656,Ad2c:0.656,Ac2s:0.656,Ac2h:0.656,Ac2d:0.656,KsQh:0.556,KsQc:0.556,KhQs:0.556,KhQc:0.556,KdQs:0.556,KdQh:0.556,KdQc:0.556,KcQs:0.556,KcQh:0.556,KsTs,KhTh,KdTd,KsTh:0.826,KsTd:0.826,KhTs:0.826,KhTd:0.826,KdTs:0.826,KdTh:0.826,KcTs:0.826,KcTh:0.826,KcTd:0.826,QsJs:0.99,QhJh:0.99,QcJc:0.99,QsJh:0.73,QsJd:0.73,QsJc:0.73,QhJs:0.73,QhJd:0.73,QhJc:0.73,QcJs:0.73,QcJh:0.73,QcJd:0.73,QsTs,QhTh,QsTh:0.896,QsTd:0.896,QhTs:0.896,QhTd:0.896,QcTs:0.896,QcTh:0.896,QcTd:0.896,Qs9s,Qh9h,Qc9c,Qs9h:0.866,Qs9d:0.866,Qs9c:0.866,Qh9s:0.866,Qh9d:0.866,Qh9c:0.866,Qc9s:0.866,Qc9h:0.866,Qc9d:0.866,Qs8s,Qh8h,Qc8c,Qs8h:0.84,Qs8d:0.84,Qs8c:0.84,Qh8s:0.84,Qh8d:0.84,Qh8c:0.84,Qc8s:0.84,Qc8h:0.84,Qc8d:0.84,Qs7s,Qh7h,Qc7c,Qs7h:0.906,Qs7d:0.906,Qs7c:0.906,Qh7s:0.906,Qh7d:0.906,Qh7c:0.906,Qc7s:0.906,Qc7h:0.906,Qc7d:0.906,Qs6s,Qh6h,Qc6c,Qs6h:0.67,Qs6d:0.67,Qs6c:0.67,Qh6s:0.67,Qh6d:0.67,Qh6c:0.67,Qc6s:0.67,Qc6h:0.67,Qc6d:0.67,Qs5s:0.496,Qh5h:0.496,Qc5c:0.496,Qs5h:0.9,Qs5d:0.9,Qs5c:0.9,Qh5s:0.9,Qh5d:0.9,Qh5c:0.9,Qc5s:0.9,Qc5h:0.9,Qc5d:0.9,Qs4s,Qh4h,Qc4c,Qs4h:0.97,Qs4d:0.97,Qs4c:0.97,Qh4s:0.97,Qh4d:0.97,Qh4c:0.97,Qc4s:0.97,Qc4h:0.97,Qc4d:0.97,Qs3s:0.926,Qh3h:0.926,Qc3c:0.926,Qs3h,Qs3d,Qs3c,Qh3s,Qh3d,Qh3c,Qc3s,Qc3h,Qc3d,Qs2s,Qh2h,Qc2c,Qs2h:0.876,Qs2d:0.876,Qs2c:0.876,Qh2s:0.876,Qh2d:0.876,Qh2c:0.876,Qc2s:0.876,Qc2h:0.876,Qc2d:0.876,JsTs,JhTh,JdTd,JsTh:0.98,JsTd:0.98,JhTs:0.98,JhTd:0.98,JdTs:0.98,JdTh:0.98,JcTs:0.98,JcTh:0.98,JcTd:0.98,Ts9s,Th9h,Td9d,Ts9h:0.786,Ts9d:0.786,Ts9c:0.786,Th9s:0.786,Th9d:0.786,Th9c:0.786,Td9s:0.786,Td9h:0.786,Td9c:0.786,Ts8s,Th8h,Td8d,Ts8h:0.76,Ts8d:0.76,Ts8c:0.76,Th8s:0.76,Th8d:0.76,Th8c:0.76,Td8s:0.76,Td8h:0.76,Td8c:0.76,Ts7s:0.35,Th7h:0.35,Td7d:0.35,Ts7h:0.986,Ts7d:0.986,Ts7c:0.986,Th7s:0.986,Th7d:0.986,Th7c:0.986,Td7s:0.986,Td7h:0.986,Td7c:0.986,Ts6s:0.366,Th6h:0.366,Td6d:0.366,Ts6h,Ts6d,Ts6c,Th6s,Th6d,Th6c,Td6s,Td6h,Td6c,Ts5s,Th5h,Td5d,Ts4s,Th4h,Td4d,Ts3s,Th3h,Td3d,Ts2s,Th2h,Td2d";

    // IP range from GTO Wizard 50bb config
    let ip_range = "KK,JJ,99-22,KJs,K9s-K2s,KJo,K9o-K6o,K5o:0.36,J9s-J2s,J9o-J8o,95s+,98o,97o:0.52,85s+,87o,74s+,64s+,53s+,AsAd,AsAc,AdAc,QsQh,QsQc,QhQc,TsTh,TsTd,ThTd,AsKs,AdKd,AcKc,AsKh,AsKd,AsKc,AdKs,AdKh,AdKc,AcKs,AcKh,AcKd,AsQs,AcQc,AsQh,AsQc,AdQs,AdQh,AdQc,AcQs,AcQh,AsJs,AdJd,AcJc,AsJh,AsJd,AsJc,AdJs,AdJh,AdJc,AcJs,AcJh,AcJd,AsTs,AdTd,AsTh,AsTd,AdTs,AdTh,AcTs,AcTh,AcTd,As9s,Ad9d,Ac9c,As9h,As9d,As9c,Ad9s,Ad9h,Ad9c,Ac9s,Ac9h,Ac9d,As8s,Ad8d,Ac8c,As8h,As8d,As8c,Ad8s,Ad8h,Ad8c,Ac8s,Ac8h,Ac8d,As7s,Ad7d,Ac7c,As7h,As7d,As7c,Ad7s,Ad7h,Ad7c,Ac7s,Ac7h,Ac7d,As6s,Ad6d,Ac6c,As6h,As6d,As6c,Ad6s,Ad6h,Ad6c,Ac6s,Ac6h,Ac6d,As5s,Ad5d,Ac5c,As5h,As5d,As5c,Ad5s,Ad5h,Ad5c,Ac5s,Ac5h,Ac5d,As4s,Ad4d,Ac4c,As4h,As4d,As4c,Ad4s,Ad4h,Ad4c,Ac4s,Ac4h,Ac4d,As3s,Ad3d,Ac3c,As3h,As3d,As3c,Ad3s,Ad3h,Ad3c,Ac3s,Ac3h,Ac3d,As2s,Ad2d,Ac2c,As2h,As2d,As2c,Ad2s,Ad2h,Ad2c,Ac2s,Ac2h,Ac2d,KsQs,KhQh,KcQc,KsQh,KsQc,KhQs,KhQc,KdQs,KdQh,KdQc,KcQs,KcQh,KsTs,KhTh,KdTd,KsTh,KsTd,KhTs,KhTd,KdTs,KdTh,KcTs,KcTh,KcTd,QsJs,QhJh,QcJc,QsJh,QsJd,QsJc,QhJs,QhJd,QhJc,QcJs,QcJh,QcJd,QsTs,QhTh,QsTh,QsTd,QhTs,QhTd,QcTs,QcTh,QcTd,Qs9s,Qh9h,Qc9c,Qs9h,Qs9d,Qs9c,Qh9s,Qh9d,Qh9c,Qc9s,Qc9h,Qc9d,Qs8s,Qh8h,Qc8c,Qs8h,Qs8d,Qs8c,Qh8s,Qh8d,Qh8c,Qc8s,Qc8h,Qc8d,Qs7s,Qh7h,Qc7c,Qs7h:0.97,Qs7d:0.97,Qs7c:0.97,Qh7s:0.97,Qh7d:0.97,Qh7c:0.97,Qc7s:0.97,Qc7h:0.97,Qc7d:0.97,Qs6s,Qh6h,Qc6c,Qs5s,Qh5h,Qc5c,Qs4s,Qh4h,Qc4c,Qs3s,Qh3h,Qc3c,Qs2s,Qh2h,Qc2c,JsTs,JhTh,JdTd,JsTh,JsTd,JhTs,JhTd,JdTs,JdTh,JcTs,JcTh,JcTd,Ts9s,Th9h,Td9d,Ts9h,Ts9d,Ts9c,Th9s,Th9d,Th9c,Td9s,Td9h,Td9c,Ts8s,Th8h,Td8d,Ts8h,Ts8d,Ts8c,Th8s,Th8d,Th8c,Td8s,Td8h,Td8c,Ts7s,Th7h,Td7d,Ts7h:0.36,Ts7d:0.36,Ts7c:0.36,Th7s:0.36,Th7d:0.36,Th7c:0.36,Td7s:0.36,Td7h:0.36,Td7c:0.36,Ts6s,Th6h,Td6d,Ts5s,Th5h,Td5d,Ts4s,Th4h,Td4d,Ts3s,Th3h,Td3d";

    // Parse ranges
    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    // Card config - starting from flop (Td9d6h same as before for comparison)
    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    // GTO Wizard 50bb bet sizes
    // OOP: bet "33, a", raise "33, 55, 83, 125, a"
    // IP: bet "20, 33, 55, 83, 125, a", raise "33, 55, 83, 125, a"
    let oop_flop_bet = BetSizeOptions::try_from(("33%, a", "33%, 55%, 83%, 125%, a")).unwrap();
    let ip_flop_bet = BetSizeOptions::try_from(("20%, 33%, 55%, 83%, 125%, a", "33%, 55%, 83%, 125%, a")).unwrap();

    // Turn: OOP bet "20, 33, 55, 83, 125, 200, a", raise "33, 55, 83, 125, a"
    //       IP bet "20, 33, 55, 83, 125, 200, a", raise "33, 55, 83, 125, a"
    let oop_turn_bet = BetSizeOptions::try_from(("20%, 33%, 55%, 83%, 125%, 200%, a", "33%, 55%, 83%, 125%, a")).unwrap();
    let ip_turn_bet = BetSizeOptions::try_from(("20%, 33%, 55%, 83%, 125%, 200%, a", "33%, 55%, 83%, 125%, a")).unwrap();

    // River: OOP bet "11, 35, 60, 85, 149, a", raise "35, 55, 126, a"
    //        IP bet "11, 35, 60, 85, 149, a", raise "35, 56, 126, a"
    let oop_river_bet = BetSizeOptions::try_from(("11%, 35%, 60%, 85%, 149%, a", "35%, 55%, 126%, a")).unwrap();
    let ip_river_bet = BetSizeOptions::try_from(("11%, 35%, 60%, 85%, 149%, a", "35%, 56%, 126%, a")).unwrap();

    // Donk sizes
    // Turn donk: "20, 50, a"
    // River donk: "11, 35, 60, 85, 149, a"
    let turn_donk = DonkSizeOptions::try_from("20%, 50%, a").ok();
    let river_donk = DonkSizeOptions::try_from("11%, 35%, 60%, 85%, 149%, a").ok();

    // Tree config from GTO Wizard 50bb
    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 6,
        effective_stack: 47,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_flop_bet, ip_flop_bet],
        turn_bet_sizes: [oop_turn_bet, ip_turn_bet],
        river_bet_sizes: [oop_river_bet, ip_river_bet],
        turn_donk_sizes: turn_donk,
        river_donk_sizes: river_donk,
        add_allin_threshold: 1.5,       // 150%
        force_allin_threshold: 0.2,     // 20%
        merging_threshold: 0.1,         // 10%
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

    // Print baseline info
    println!();
    println!("=== Game Info (Before Abstraction) ===");
    println!("OOP hands: {}", game.private_cards(0).len());
    println!("IP hands: {}", game.private_cards(1).len());
    let (mem_uncompressed_baseline, _) = game.memory_usage();
    println!("Baseline memory (uncompressed): {:.2} MB", mem_uncompressed_baseline as f64 / 1024.0 / 1024.0);

    // Enable abstraction
    println!();
    println!("=== Enabling Abstraction ===");
    let abs_start = Instant::now();
    let config = AbstractionConfig {
        num_buckets,
        max_iterations: 100,
    };
    game.enable_abstraction(&config).expect("Failed to enable abstraction");
    println!("Abstraction computed in {:.2}s", abs_start.elapsed().as_secs_f64());

    // Print abstracted info
    println!();
    println!("=== Game Info (After Abstraction) ===");
    println!("OOP buckets: {}", game.effective_hand_count(0));
    println!("IP buckets: {}", game.effective_hand_count(1));
    let (mem_uncompressed_abs, mem_compressed_abs) = game.memory_usage();
    println!("Memory (uncompressed): {:.2} MB", mem_uncompressed_abs as f64 / 1024.0 / 1024.0);
    println!("Memory (compressed): {:.2} MB", mem_compressed_abs as f64 / 1024.0 / 1024.0);
    let memory_reduction = (1.0 - mem_uncompressed_abs as f64 / mem_uncompressed_baseline as f64) * 100.0;
    println!("Memory reduction: {:.1}%", memory_reduction);
    println!();

    // Allocate memory
    println!("Allocating memory (32-bit FP, no compression)...");
    let alloc_start = Instant::now();
    game.allocate_memory(false);
    println!("Memory allocated in {:.2}s", alloc_start.elapsed().as_secs_f64());
    println!();

    // Target exploitability: 0.08
    let max_iterations = 10000;
    let target_exploitability = 0.08;

    println!("=== Solving ===");
    println!("Max iterations: {}", max_iterations);
    println!("Target exploitability: {:.4}", target_exploitability);
    println!();

    // Solve
    let solve_start = Instant::now();
    let exploitability = solve(&mut game, max_iterations, target_exploitability, true);
    let solve_time = solve_start.elapsed();

    // Results
    println!();
    println!("=== Results ===");
    println!("Solve time: {:.2}s ({:.0} ms)", solve_time.as_secs_f64(), solve_time.as_millis());
    println!("Final exploitability: {:.4}", exploitability);
    println!("Exploitability as % of pot: {:.3}%", exploitability / game.tree_config().starting_pot as f32 * 100.0);
    println!();

    // Summary
    println!("=== Summary ===");
    println!("Config: GTO Wizard 50bb");
    println!("Flop: Td9d6h");
    println!("Buckets: {}", num_buckets);
    println!("Threads: {}", rayon::current_num_threads());
    println!("Memory reduction: {:.1}%", memory_reduction);
    println!("Solve time: {:.2}s", solve_time.as_secs_f64());
    println!("Exploitability: {:.4}", exploitability);
}
