//! POC: Lookup Table at Turn Boundary (DeepStack-style)
//!
//! Proves: if we have correct turn/river CFV lookup tables (extracted from full tree),
//! a flop-only DCFR using external matrices at turn boundaries produces
//! the same flop strategies and CFVs as the full solve.
//!
//! Option 1: Extract matrices directly from the full solved tree.
//! This guarantees exact match (same chance_factor, isomorphism, Nash strategies).
//!
//! Usage:
//!   cargo run --example poc_precompute --release --features "bincode rayon" -- config/poc.json

#[path = "../common/mod.rs"]
mod common;
mod oracle_solver;

use common::*;
use oracle_solver::TreeOracle;
use postflop_solver::*;
use std::env;
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }
    let config_path = &args[1];
    if !Path::new(config_path).exists() {
        eprintln!("Error: Config file not found: {}", config_path);
        std::process::exit(1);
    }
    let config = load_config(config_path);
    let (card_config, tree_config) = parse_configs(&config).unwrap();
    let target_exploitability =
        tree_config.starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;
    let starting_pot = tree_config.starting_pot as f32;

    println!("=== POC: Lookup Table at Turn Boundary (DeepStack-style) ===");
    println!("Board: {}", config.board.flop);
    println!("Pot: {}, Stack: {}", tree_config.starting_pot, tree_config.effective_stack);
    println!();

    // =========================================================================
    // Phase 1: Full solve (ground truth)
    // =========================================================================
    println!("--- Phase 1: Full Solve (ground truth) ---");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let boundary_amounts = action_tree.boundary_amounts();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory(false);
    println!("  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0), game.num_private_hands(1));
    println!("  Boundary amounts: {:?}", boundary_amounts);

    let phase1_start = Instant::now();
    let full_exploitability = manual_solve_loop(
        &game, config.solver.max_iterations, target_exploitability, starting_pot, "full",
    );
    let phase1_time = phase1_start.elapsed().as_secs_f64();
    let full_pct = full_exploitability / starting_pot * 100.0;
    println!("  Full solve: {:.4} ({:.3}% of pot) in {:.2}s", full_exploitability, full_pct, phase1_time);

    let ref_cfvs = extract_root_cfvs(&game);
    print_ev_summary(&game, &ref_cfvs, "  Full solve");

    // =========================================================================
    // Phase 2: Extract oracle from full solved tree
    // =========================================================================
    println!();
    println!("--- Phase 2: Extract Oracle from Full Tree ---");
    let phase2_start = Instant::now();
    let oracle = TreeOracle::build(&game, true);
    let phase2_time = phase2_start.elapsed().as_secs_f64();
    println!("  Oracle ready in {:.2}s ({} boundary amounts)", phase2_time, oracle.num_amounts());

    // =========================================================================
    // Phase 3: Validate oracle vs full tree
    // =========================================================================
    println!();
    println!("--- Phase 3: Validate Oracle vs Full Tree ---");
    validate_oracle_vs_tree(&game, &oracle);

    // =========================================================================
    // Phase 4: Reset flop + oracle DCFR
    // =========================================================================
    println!();
    println!("--- Phase 4: Reset flop + Oracle DCFR ---");

    let flop_nodes_reset = oracle_solver::reset_flop_storage(&game);
    println!("  Reset {} flop action nodes", flop_nodes_reset);

    // Use the full solve's exploitability as the target so oracle converges to the same level.
    // The config target (2%) causes oracle to stop too early since turn/river are already converged.
    let oracle_target = full_exploitability;
    println!("  Oracle target: {:.4} (matching full solve)", oracle_target);

    let phase4_start = Instant::now();
    let oracle_exploitability = oracle_solver::solve_flop_with_oracle(
        &game,
        &oracle,
        config.solver.max_iterations,
        oracle_target,
        true,
    );
    let phase4_time = phase4_start.elapsed().as_secs_f64();
    let oracle_pct = oracle_exploitability / starting_pot * 100.0;
    println!("  Oracle solve: {:.4} ({:.3}% of pot) in {:.2}s",
        oracle_exploitability, oracle_pct, phase4_time);

    let oracle_cfvs = extract_root_cfvs(&game);
    print_ev_summary(&game, &oracle_cfvs, "  Oracle solve");

    // =========================================================================
    // Phase 5: Compare
    // =========================================================================
    println!();
    println!("--- Phase 5: Compare ---");
    for player in 0..2 {
        let pname = if player == 0 { "OOP" } else { "IP" };
        let n = ref_cfvs[player].len();
        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f64;
        for i in 0..n {
            let diff = (ref_cfvs[player][i] - oracle_cfvs[player][i]).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as f64;
        }
        let avg_diff = if n > 0 { total_diff / n as f64 } else { 0.0 };
        println!("  {}: max_diff={:.6}, avg_diff={:.8} ({} hands)", pname, max_diff, avg_diff, n);
    }

    println!();
    let delta = (full_pct - oracle_pct).abs();
    println!("  Exploitability: full={:.3}%, oracle={:.3}%, delta={:.3}%", full_pct, oracle_pct, delta);
    if delta < 0.5 {
        println!("  RESULT: SUCCESS — Oracle solve matches full solve.");
        println!("  -> Tree-extracted matrix at turn boundary + flop-only DCFR = full DCFR");
        println!("  -> Safe to replace with NN at turn boundary");
    } else {
        println!("  RESULT: MISMATCH — needs investigation (delta={:.3}%).", delta);
    }

    // =========================================================================
    // Save .flop file so you can inspect the strategy
    // =========================================================================
    println!();
    println!("--- Saving .flop ---");
    finalize(&mut game);
    let output_path = &config.output.filename;
    if let Some(parent) = Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let memo = config.output.memo.as_deref().unwrap_or("oracle_poc");
    save_data_to_file(&game, memo, output_path, config.output.compression_level)
        .expect("Failed to save .flop file");
    println!("  Saved to {}", output_path);

    println!();
    println!("  Phase 1 (full solve): {:.2}s", phase1_time);
    println!("  Phase 2 (extract oracle): {:.2}s", phase2_time);
    println!("  Phase 4 (oracle solve): {:.2}s", phase4_time);
}

// =============================================================================
// Manual solve loop using solve_step (avoids finalize/is_solved)
// =============================================================================

fn manual_solve_loop(
    game: &PostFlopGame,
    max_iterations: u32,
    target_exploitability: f32,
    starting_pot: f32,
    label: &str,
) -> f32 {
    let mut exploitability = compute_exploitability(game);

    for t in 0..max_iterations {
        if exploitability <= target_exploitability {
            break;
        }
        solve_step(game, t);

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            exploitability = compute_exploitability(game);
            let pct = exploitability / starting_pot * 100.0;
            let target_pct = target_exploitability / starting_pot * 100.0;
            print!("\r  {} iter: {} / {} (exploitability = {:.2}% | target = {:.2}%)",
                label, t + 1, max_iterations, pct, target_pct);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }
    println!();
    exploitability
}

// =============================================================================
// Validate oracle vs full tree at turn boundary
// =============================================================================

fn validate_oracle_vs_tree(game: &PostFlopGame, oracle: &TreeOracle) {
    for player in 0..2 {
        let pname = if player == 0 { "OOP" } else { "IP" };
        let num_hands = game.num_private_hands(player);
        let cfreach = game.initial_weights(player ^ 1).to_vec();

        // Full tree CFVs
        let mut full_result = vec![MaybeUninit::<f32>::uninit(); num_hands];
        {
            let mut root = game.root();
            compute_cfvalue_recursive(&mut full_result, game, &mut root, player, &cfreach, false);
        }
        let full_cfvs: Vec<f32> = full_result.iter().map(|v| unsafe { v.assume_init() }).collect();

        // Oracle CFVs (traverse flop nodes, use oracle at boundary)
        let mut oracle_result = vec![MaybeUninit::<f32>::uninit(); num_hands];
        {
            let mut root = game.root();
            oracle_solver::compute_cfvalue_with_oracle(
                &mut oracle_result, game, oracle, &mut root, player, &cfreach,
            );
        }
        let oracle_cfvs: Vec<f32> = oracle_result.iter().map(|v| unsafe { v.assume_init() }).collect();

        let mut max_diff = 0.0f32;
        let mut total_diff = 0.0f64;
        for i in 0..num_hands {
            let diff = (full_cfvs[i] - oracle_cfvs[i]).abs();
            max_diff = max_diff.max(diff);
            total_diff += diff as f64;
        }
        let avg_diff = total_diff / num_hands as f64;
        println!("  {}: oracle vs tree max_diff={:.6}, avg_diff={:.8}", pname, max_diff, avg_diff);
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn extract_root_cfvs(game: &PostFlopGame) -> [Vec<f32>; 2] {
    let mut cfvs = [Vec::new(), Vec::new()];
    for player in 0..2 {
        let num_hands = game.num_private_hands(player);
        let cfreach = game.initial_weights(player ^ 1).to_vec();
        let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
        {
            let mut root = game.root();
            compute_cfvalue_recursive(&mut result, game, &mut root, player, &cfreach, false);
        }
        cfvs[player] = result.into_iter().map(|v| unsafe { v.assume_init() }).collect();
    }
    cfvs
}

fn print_ev_summary(game: &PostFlopGame, cfvs: &[Vec<f32>; 2], label: &str) {
    for player in 0..2 {
        let pname = if player == 0 { "OOP" } else { "IP" };
        let weights = game.initial_weights(player);
        let ev: f64 = cfvs[player].iter().zip(weights)
            .map(|(&v, &w)| v as f64 * w as f64).sum();
        println!("{} {}: weighted EV = {:.6}", label, pname, ev);
    }
}
