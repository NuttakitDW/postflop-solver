//! Build boundary training data (.bt1) using the library's own DCFR.
//!
//! Same flow as build_pairs_v2 but also records cfreach (opponent reach
//! probabilities) at each turn boundary. The cfreach is needed for NN training.
//!
//! .bt1 format per boundary per iteration:
//!   cfv[player_hands]      — boundary CFV (same as .dpairs2)
//!   cfreach[opponent_hands] — opponent reach at this boundary (NEW)
//!
//! Usage:
//!   cargo run --example build_bt1 --release --features "bincode rayon" -- config/9s6d6c.json

#[path = "../common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

// =============================================================================
// Count turn boundary nodes by walking the flop tree
// =============================================================================

fn count_turn_boundaries(game: &PostFlopGame) -> usize {
    fn walk(node: &mut PostFlopNode, count: &mut usize) {
        if node.is_terminal() { return; }
        if node.is_chance() && node.turn() == NOT_DEALT {
            *count += 1;
            return;
        }
        let num_actions = node.num_actions();
        for a in 0..num_actions {
            walk(&mut node.play(a), count);
        }
    }
    let mut root = game.root();
    let mut count = 0;
    walk(&mut root, &mut count);
    count
}

// =============================================================================
// Save .bt1 format
// =============================================================================

struct IterationRecord {
    iteration: u32,
    exploitability: f32,
    convergence_mode: bool,
    /// boundary_cfvs[player][boundary_idx] = cfv vector
    boundary_cfvs: [Vec<Vec<f32>>; 2],
    /// boundary_cfreaches[player][boundary_idx] = opponent reach vector
    boundary_cfreaches: [Vec<Vec<f32>>; 2],
}

fn save_bt1(
    path: &str,
    num_hands: [usize; 2],
    num_boundaries: usize,
    starting_pot: f32,
    records: &[IterationRecord],
) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);

    // Header (32 bytes, same layout as dpairs2)
    f.write_all(b"BT1\0\0\0\0\0")?;                               // magic (8 bytes)
    f.write_all(&1u32.to_le_bytes())?;                              // version
    f.write_all(&(num_hands[0] as u32).to_le_bytes())?;             // num_oop
    f.write_all(&(num_hands[1] as u32).to_le_bytes())?;             // num_ip
    f.write_all(&(num_boundaries as u32).to_le_bytes())?;           // num_boundaries
    f.write_all(&(records.len() as u32).to_le_bytes())?;            // num_iterations
    f.write_all(&starting_pot.to_le_bytes())?;                      // starting_pot

    for rec in records {
        f.write_all(&rec.iteration.to_le_bytes())?;
        f.write_all(&rec.exploitability.to_le_bytes())?;
        f.write_all(&(rec.convergence_mode as u32).to_le_bytes())?;

        for b in 0..num_boundaries {
            for player in 0..2 {
                let opponent = player ^ 1;

                // CFV: indexed by player's hands
                let cfv = &rec.boundary_cfvs[player][b];
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(cfv.as_ptr() as *const u8, cfv.len() * 4)
                };
                f.write_all(bytes)?;

                // cfreach: opponent's reach, indexed by opponent's hands
                let cfreach = &rec.boundary_cfreaches[player][b];
                assert_eq!(cfreach.len(), num_hands[opponent],
                    "cfreach len {} != expected {} (opponent hands)", cfreach.len(), num_hands[opponent]);
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(cfreach.as_ptr() as *const u8, cfreach.len() * 4)
                };
                f.write_all(bytes)?;
            }
        }
    }
    f.flush()?;
    Ok(())
}

// =============================================================================
// Main
// =============================================================================

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
    let starting_pot = tree_config.starting_pot as f32;
    let target_exploitability =
        starting_pot * config.solver.target_exploitability_percent / 100.0;

    let config_filename = Path::new(config_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let bt1_path = format!("data/bt1/{}.bt1", config_filename);
    std::fs::create_dir_all("data/bt1").ok();

    let total_start = Instant::now();

    println!("=== Build BT1 (boundary training data) ===");
    println!("Config: {}", config_path);
    println!();

    // Build game tree
    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory(false);

    let num_hands = [game.num_private_hands(0), game.num_private_hands(1)];
    let (mem_usage, _) = game.memory_usage();
    let memory_mb = mem_usage as f64 / 1024.0 / 1024.0;

    println!("  Board: {}", config.board.flop);
    println!("  Pot: {}, Stack: {}", tree_config.starting_pot, tree_config.effective_stack);
    println!("  OOP hands: {}, IP hands: {}", num_hands[0], num_hands[1]);
    println!("  Memory: {:.2} MB", memory_mb);
    println!();

    // Count turn boundary nodes
    let num_boundaries = count_turn_boundaries(&game);
    println!("  Turn boundary nodes: {}", num_boundaries);
    println!();

    // Main loop: library DCFR with boundary recording
    println!("--- Recording boundary data (CFV + cfreach) ---");
    let mut records: Vec<IterationRecord> = Vec::new();
    let mut exploitability = compute_exploitability(&game);
    let mut convergence_mode = false;

    let solve_start = Instant::now();

    for t in 0..config.solver.max_iterations {
        if exploitability <= target_exploitability {
            println!();
            println!("  Converged at iteration {} (exploitability={:.4}%)",
                t, exploitability / starting_pot * 100.0);
            break;
        }

        // Match library's convergence_mode logic
        let is_power_of_4 = t > 0 && t == 1u32 << ((t.leading_zeros() ^ 31) & !1);
        if is_power_of_4 {
            exploitability = compute_exploitability(&game);
        }

        if starting_pot > 0.0 && (exploitability / starting_pot * 100.0) < 1.0 {
            convergence_mode = true;
        }

        // Player 0: record boundary CFVs + cfreach + DCFR update
        let (p0_cfvs, p0_cfreaches) = solve_step_for_player_recording_with_cfreach(&game, t, 0, convergence_mode);
        assert_eq!(p0_cfvs.len(), num_boundaries,
            "Player 0 boundary count mismatch: {} vs {}", p0_cfvs.len(), num_boundaries);
        assert_eq!(p0_cfreaches.len(), num_boundaries,
            "Player 0 cfreach count mismatch: {} vs {}", p0_cfreaches.len(), num_boundaries);

        // Player 1: record boundary CFVs + cfreach + DCFR update
        let (p1_cfvs, p1_cfreaches) = solve_step_for_player_recording_with_cfreach(&game, t, 1, convergence_mode);
        assert_eq!(p1_cfvs.len(), num_boundaries,
            "Player 1 boundary count mismatch: {} vs {}", p1_cfvs.len(), num_boundaries);
        assert_eq!(p1_cfreaches.len(), num_boundaries,
            "Player 1 cfreach count mismatch: {} vs {}", p1_cfreaches.len(), num_boundaries);

        // Compute exploitability
        let check_exploitability = (t + 1) % 10 == 0
            || t + 1 == config.solver.max_iterations;
        if check_exploitability {
            exploitability = compute_exploitability(&game);
        }

        records.push(IterationRecord {
            iteration: t,
            exploitability,
            convergence_mode,
            boundary_cfvs: [p0_cfvs, p1_cfvs],
            boundary_cfreaches: [p0_cfreaches, p1_cfreaches],
        });

        let elapsed = solve_start.elapsed().as_secs_f64();
        let pct = exploitability / starting_pot * 100.0;
        print!("\r  iter {}: exploit={:.4}%, conv_mode={}, total={:.1}s    ",
            t, pct, convergence_mode, elapsed);
        io::stdout().flush().unwrap();
    }
    println!();

    let solve_total = solve_start.elapsed().as_secs_f64();

    // Save .bt1
    println!("--- Saving ---");
    save_bt1(&bt1_path, num_hands, num_boundaries, starting_pot, &records)
        .expect("Failed to save .bt1");
    let bt1_size = std::fs::metadata(&bt1_path).map(|m| m.len()).unwrap_or(0);

    // Save standard-solved .flop (ground truth)
    let tree_path = format!("data/out/{}-standard.flop", config_filename);
    if let Some(parent) = Path::new(&tree_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    finalize(&mut game);
    let memo = config.output.memo.as_deref().unwrap_or("standard-bt1");
    save_data_to_file(&game, memo, &tree_path, None)
        .expect("Failed to save .flop file");
    let flop_size = std::fs::metadata(&tree_path).map(|m| m.len()).unwrap_or(0);

    let total_time = total_start.elapsed().as_secs_f64();

    println!();
    println!("=== Summary ===");
    println!("Board: {}", config.board.flop);
    println!("Boundary nodes: {}", num_boundaries);
    println!("Iterations: {}", records.len());
    println!("Final exploitability: {:.4}%",
        records.last().map(|r| r.exploitability / starting_pot * 100.0).unwrap_or(0.0));
    println!("Convergence mode: {}", convergence_mode);
    println!("Solve+record time: {:.2}s", solve_total);
    println!("Total time: {:.2}s", total_time);
    println!();
    println!("BT1: {} ({:.2} MB)", bt1_path, bt1_size as f64 / 1048576.0);
    println!("Ground truth: {} ({:.2} MB)", tree_path, flop_size as f64 / 1048576.0);
    println!("Memory: {:.2} MB", memory_mb);
}
