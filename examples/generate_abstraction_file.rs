//! Generate a solved game file with abstraction enabled.
//!
//! Run with: cargo run --example generate_abstraction_file --release --features bincode
//!
//! This creates a file with the same settings as abstraction_results.txt:
//! - Board: Td9d6h (flop)
//! - 50 buckets per player
//! - Starting pot: 55, Effective stack: 180
//! - Bet sizes: 25%, 50%, 100% / 50% raise

use postflop_solver::*;
use std::time::Instant;

fn main() {
    println!("=== Generate Abstraction File ===");
    println!("Threads: {}", rayon::current_num_threads());
    println!();

    // Same OOP range as benchmark_abstraction
    let oop_range = "A9s-A3s,A8o:0.906,A7o:0.826,A6o:0.78,A5o:0.49,A4o:0.07,QJs,Q9s-Q3s,QJo,Q9o,Q8o:0.996,Q7o,Q6o:0.99,Q5o:0.92,Q4o:0.916,Q3o:0.866,J9s:0.89,J8s:0.99,J7s-J3s,J9o-J8o,J7o:0.9,J6o:0.98,J5o:0.86,J4o:0.936,J3o:0.98,97s:0.99,96s-93s,96o+,95o:0.936,87s:0.026,86s-83s,85o+,76s:0.946,75s-73s,74o+,63s+,64o+,63o:0.836,53+,43,AhTh:0.37,AdTd:0.37,AcTc:0.37,As2s,Ah2h,Ac2c,KhQh,KdQd,KcQc,KhQs:0.496,KhQd:0.496,KhQc:0.496,KdQs:0.496,KdQh:0.496,KdQc:0.496,KcQs:0.496,KcQh:0.496,KcQd:0.496,KhJh,KdJd,KcJc,KhJs,KhJd,KhJc,KdJs,KdJh,KdJc,KcJs,KcJh,KcJd,KhTh,KdTd,KcTc,KhTd,KhTc,KdTh,KdTc,KcTh,KcTd,Kh9h,Kd9d,Kc9c,Kh9s,Kh9d,Kh9c,Kd9s,Kd9h,Kd9c,Kc9s,Kc9h,Kc9d,Kh8h,Kd8d,Kc8c,Kh8s,Kh8d,Kh8c,Kd8s,Kd8h,Kd8c,Kc8s,Kc8h,Kc8d,Kh7s:0.92,Kh7d:0.92,Kh7c:0.92,Kd7s:0.92,Kd7h:0.92,Kd7c:0.92,Kc7s:0.92,Kc7h:0.92,Kc7d:0.92,Kh6s:0.96,Kh6d:0.96,Kh6c:0.96,Kd6s:0.96,Kd6h:0.96,Kd6c:0.96,Kc6s:0.96,Kc6h:0.96,Kc6d:0.96,Kh5h:0.826,Kd5d:0.826,Kc5c:0.826,Kh5s:0.95,Kh5d:0.95,Kh5c:0.95,Kd5s:0.95,Kd5h:0.95,Kd5c:0.95,Kc5s:0.95,Kc5h:0.95,Kc5d:0.95,Kh4h,Kd4d,Kc4c,Kh4s:0.9,Kh4d:0.9,Kh4c:0.9,Kd4s:0.9,Kd4h:0.9,Kd4c:0.9,Kc4s:0.9,Kc4h:0.9,Kc4d:0.9,Kh3h,Kd3d,Kc3c,Kh3s:0.92,Kh3d:0.92,Kh3c:0.92,Kd3s:0.92,Kd3h:0.92,Kd3c:0.92,Kc3s:0.92,Kc3h:0.92,Kc3d:0.92,Kh2h,Kc2c,Kh2s:0.966,Kh2c:0.966,Kd2s:0.966,Kd2h:0.966,Kd2c:0.966,Kc2s:0.966,Kc2h:0.966,QhTh,QdTd,QcTc,QsTh,QsTd,QsTc,QhTd,QhTc,QdTh,QdTc,QcTh,QcTd,Qs2s,Qh2h,Qc2c,Qs2h:0.906,Qs2c:0.906,Qh2s:0.906,Qh2c:0.906,Qd2s:0.906,Qd2h:0.906,Qd2c:0.906,Qc2s:0.906,Qc2h:0.906,JhTh:0.686,JdTd:0.686,JcTc:0.686,JsTh,JsTd,JsTc,JhTd,JhTc,JdTh,JdTc,JcTh,JcTd,Js2s,Jh2h,Jc2c,Js2h:0.986,Js2c:0.986,Jh2s:0.986,Jh2c:0.986,Jd2s:0.986,Jd2h:0.986,Jd2c:0.986,Jc2s:0.986,Jc2h:0.986,Th9h:0.026,Td9d:0.026,Tc9c:0.026,Th9s,Th9d,Th9c,Td9s,Td9h,Td9c,Tc9s,Tc9h,Tc9d,Th8h:0.376,Td8d:0.376,Tc8c:0.376,Th8s,Th8d,Th8c,Td8s,Td8h,Td8c,Tc8s,Tc8h,Tc8d,Th7h,Td7d,Tc7c,Th7s,Th7d,Th7c,Td7s,Td7h,Td7c,Tc7s,Tc7h,Tc7d,Th6h,Td6d,Tc6c,Th6s:0.866,Th6d:0.866,Th6c:0.866,Td6s:0.866,Td6h:0.866,Td6c:0.866,Tc6s:0.866,Tc6h:0.866,Tc6d:0.866,Th5h,Td5d,Tc5c,Th5s:0.49,Th5d:0.49,Th5c:0.49,Td5s:0.49,Td5h:0.49,Td5c:0.49,Tc5s:0.49,Tc5h:0.49,Tc5d:0.49,Th4h,Td4d,Tc4c,Th4s:0.46,Th4d:0.46,Th4c:0.46,Td4s:0.46,Td4h:0.46,Td4c:0.46,Tc4s:0.46,Tc4h:0.46,Tc4d:0.46,Th3h,Td3d,Tc3c,Th2h,Tc2c,9s2s,9h2h,9c2c,8s2s,8h2h,8c2c,7s2s,7h2h,7c2c,6s2s,6h2h,6c2c,5s2s,5h2h,5c2c,5s2h:0.58,5s2c:0.58,5h2s:0.58,5h2c:0.58,5d2s:0.58,5d2h:0.58,5d2c:0.58,5c2s:0.58,5c2h:0.58,4s2s,4h2h,4c2c,3s2s,3h2h,3c2c";

    // Same IP range as benchmark_abstraction
    let ip_range = "AA:0.826,QQ:0.876,JJ:0.816,99:0.72,88:0.846,77:0.946,66:0.93,55:0.82,AQs:0.766,AJs:0.69,A9s,A8s:0.916,A7s:0.486,A6s:0.466,A5s:0.26,AQo:0.786,AJo:0.78,A9o:0.136,A8o:0.09,A7o:0.306,A6o:0.946,A5o:0.72,A4o:0.67,A3o:0.586,QJs,Q9s:0.3,Q8s:0.65,Q7s:0.75,Q6s:0.89,Q5s:0.95,Q4s:0.63,QJo:0.62,Q9o:0.746,Q8o:0.36,J9s:0.5,J8s:0.816,J7s:0.91,J6s:0.75,J5s:0.59,J4s:0.2,J9o:0.73,J8o:0.336,98s:0.806,97s:0.8,96s:0.666,98o:0.4,87s:0.63,86s:0.61,85s:0.746,76s:0.666,75s:0.556,65s:0.496,54s:0.42,KhKd:0.846,KhKc:0.846,KdKc:0.846,ThTd:0.73,ThTc:0.73,TdTc:0.73,AhKh:0.96,AdKd:0.96,AcKc:0.96,AsKh:0.8,AsKd:0.8,AsKc:0.8,AhKd:0.8,AhKc:0.8,AdKh:0.8,AdKc:0.8,AcKh:0.8,AcKd:0.8,AhTh:0.516,AdTd:0.516,AcTc:0.516,AsTh:0.8,AsTd:0.8,AsTc:0.8,AhTd:0.8,AhTc:0.8,AdTh:0.8,AdTc:0.8,AcTh:0.8,AcTd:0.8,As2h:0.24,As2c:0.24,Ah2s:0.24,Ah2c:0.24,Ad2s:0.24,Ad2h:0.24,Ad2c:0.24,Ac2s:0.24,Ac2h:0.24,KhQh:0.84,KdQd:0.84,KcQc:0.84,KhQs:0.58,KhQd:0.58,KhQc:0.58,KdQs:0.58,KdQh:0.58,KdQc:0.58,KcQs:0.58,KcQh:0.58,KcQd:0.58,KhJh:0.916,KdJd:0.916,KcJc:0.916,KhJs:0.516,KhJd:0.516,KhJc:0.516,KdJs:0.516,KdJh:0.516,KdJc:0.516,KcJs:0.516,KcJh:0.516,KcJd:0.516,KhTh:0.346,KdTd:0.346,KcTc:0.346,KhTd:0.826,KhTc:0.826,KdTh:0.826,KdTc:0.826,KcTh:0.826,KcTd:0.826,Kh9h:0.276,Kd9d:0.276,Kc9c:0.276,Kh9s:0.736,Kh9d:0.736,Kh9c:0.736,Kd9s:0.736,Kd9h:0.736,Kd9c:0.736,Kc9s:0.736,Kc9h:0.736,Kc9d:0.736,Kh8h:0.44,Kd8d:0.44,Kc8c:0.44,Kh8s:0.73,Kh8d:0.73,Kh8c:0.73,Kd8s:0.73,Kd8h:0.73,Kd8c:0.73,Kc8s:0.73,Kc8h:0.73,Kc8d:0.73,Kh7h:0.486,Kd7d:0.486,Kc7c:0.486,Kh7s:0.586,Kh7d:0.586,Kh7c:0.586,Kd7s:0.586,Kd7h:0.586,Kd7c:0.586,Kc7s:0.586,Kc7h:0.586,Kc7d:0.586,Kh6h:0.686,Kd6d:0.686,Kc6c:0.686,Kh5h:0.926,Kd5d:0.926,Kc5c:0.926,Kh4h:0.89,Kd4d:0.89,Kc4c:0.89,Kh3h:0.566,Kd3d:0.566,Kc3c:0.566,QhTh:0.176,QdTd:0.176,QcTc:0.176,QsTh:0.756,QsTd:0.756,QsTc:0.756,QhTd:0.756,QhTc:0.756,QdTh:0.756,QdTc:0.756,QcTh:0.756,QcTd:0.756,JsTh:0.81,JsTd:0.81,JsTc:0.81,JhTd:0.81,JhTc:0.81,JdTh:0.81,JdTc:0.81,JcTh:0.81,JcTd:0.81,Th9h:0.56,Td9d:0.56,Tc9c:0.56,Th9s:0.7,Th9d:0.7,Th9c:0.7,Td9s:0.7,Td9h:0.7,Td9c:0.7,Tc9s:0.7,Tc9h:0.7,Tc9d:0.7,Th8h:0.84,Td8d:0.84,Tc8c:0.84,Th8s:0.3,Th8d:0.3,Th8c:0.3,Td8s:0.3,Td8h:0.3,Td8c:0.3,Tc8s:0.3,Tc8h:0.3,Tc8d:0.3,Th7h:0.81,Td7d:0.81,Tc7c:0.81,Th6h:0.5,Td6d:0.5,Tc6c:0.5";

    let oop: Range = oop_range.parse().expect("Failed to parse OOP range");
    let ip: Range = ip_range.parse().expect("Failed to parse IP range");

    let card_config = CardConfig {
        range: [oop, ip],
        flop: flop_from_str("Td9d6h").unwrap(),
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let bet_sizes = BetSizeOptions::try_from(("25%, 50%, 100%", "50%")).unwrap();

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 55,
        effective_stack: 180,
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

    // Build action tree
    println!("Building action tree...");
    let action_tree = ActionTree::new(tree_config).expect("Failed to create action tree");

    // Create game
    println!("Creating game...");
    let mut game = PostFlopGame::with_config(card_config, action_tree).expect("Failed to create game");

    println!("OOP hands: {}", game.private_cards(0).len());
    println!("IP hands: {}", game.private_cards(1).len());

    // Enable abstraction with 50 buckets
    println!();
    println!("Enabling abstraction (50 buckets)...");
    let abs_start = Instant::now();
    let config = AbstractionConfig {
        num_buckets: 50,
        max_iterations: 100,
    };
    game.enable_abstraction(&config).expect("Failed to enable abstraction");
    println!("Abstraction computed in {:.2}s", abs_start.elapsed().as_secs_f64());

    println!("OOP buckets: {}", game.effective_hand_count(0));
    println!("IP buckets: {}", game.effective_hand_count(1));

    let (mem_uncompressed, mem_compressed) = game.memory_usage();
    println!("Memory (uncompressed): {:.2} MB", mem_uncompressed as f64 / 1024.0 / 1024.0);
    println!("Memory (compressed): {:.2} MB", mem_compressed as f64 / 1024.0 / 1024.0);

    // Allocate memory
    println!();
    println!("Allocating memory...");
    game.allocate_memory(false);

    // Solve
    let max_iterations = 1000;
    let target_exploitability = game.tree_config().starting_pot as f32 * 0.005;

    println!();
    println!("Solving (max {} iterations, target exploitability {:.4})...", max_iterations, target_exploitability);
    let solve_start = Instant::now();
    let exploitability = solve(&mut game, max_iterations, target_exploitability, true);
    let solve_time = solve_start.elapsed();

    println!();
    println!("Solve time: {:.2}s", solve_time.as_secs_f64());
    println!("Final exploitability: {:.4}", exploitability);

    // Save to file
    let filename = "abstraction_game.flop";
    let memo = format!(
        "Td9d6h flop, 50 buckets, pot=55, stack=180, exploitability={:.4}",
        exploitability
    );

    println!();
    println!("Saving to {}...", filename);
    save_data_to_file(&game, &memo, filename, None).expect("Failed to save file");
    println!("File saved successfully!");

    // Verify the file can be loaded
    println!();
    println!("Verifying file can be loaded...");
    let (loaded_game, loaded_memo): (PostFlopGame, String) =
        load_data_from_file(filename, None).expect("Failed to load file");

    println!("Loaded memo: {}", loaded_memo);
    println!("Abstraction enabled: {}", loaded_game.is_abstraction_enabled());
    if let Some(abs_data) = loaded_game.abstraction_data() {
        println!("OOP buckets: {}", abs_data.num_buckets(0));
        println!("IP buckets: {}", abs_data.num_buckets(1));
    }

    println!();
    println!("=== Success! ===");
    println!("File '{}' created with abstraction settings.", filename);
    println!("You can load this file in desktop-postflop.");
}
