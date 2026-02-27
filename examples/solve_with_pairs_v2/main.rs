//! Phase 1 v2: Solve a flop game using pre-recorded boundary pairs (.dpairs2).
//!
//! Uses `solve_step_for_player_replay` — the library's exact DCFR code with
//! pre-recorded boundary CFVs injected at turn boundaries. This guarantees
//! float-identical flop strategies compared to the standard solver.
//!
//! Usage:
//!   cargo run --example solve_with_pairs_v2 --release --features "bincode rayon" -- config/toy.json
//!
//! Prerequisite: build the training data first:
//!   cargo run --example build_pairs_v2 --release --features "bincode rayon" -- config/toy.json

#[path = "../common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::io::{self, Read as _, Write as _};
use std::path::Path;
use std::time::Instant;

// =============================================================================
// Boundary pairs v2 loader
// =============================================================================

struct BoundaryPairsV2 {
    num_hands: [usize; 2],
    num_boundaries: usize,
    starting_pot: f32,
    /// iterations[t].exploitability
    iterations: Vec<IterationData>,
}

struct IterationData {
    _iteration: u32,
    _exploitability: f32,
    convergence_mode: bool,
    /// boundary_cfvs[boundary_idx * 2 + player] = cfv vector
    boundary_cfvs: Vec<Vec<f32>>,
}

impl BoundaryPairsV2 {
    fn load(path: &str) -> io::Result<Self> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"DPAIRS2\0" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic (expected DPAIRS2)"));
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);
        f.read_exact(&mut buf4)?;
        let num_oop = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_ip = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_boundaries = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_iterations = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let starting_pot = f32::from_le_bytes(buf4);

        let num_hands = [num_oop, num_ip];
        let mut iterations = Vec::with_capacity(num_iterations);

        for _ in 0..num_iterations {
            f.read_exact(&mut buf4)?;
            let iteration = u32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let exploitability = f32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let convergence_mode = u32::from_le_bytes(buf4) != 0;

            let mut boundary_cfvs = Vec::with_capacity(num_boundaries * 2);
            for _b in 0..num_boundaries {
                for player in 0..2 {
                    let n_player = num_hands[player];
                    let mut cfv = vec![0.0f32; n_player];
                    let cfv_bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            cfv.as_mut_ptr() as *mut u8,
                            n_player * 4,
                        )
                    };
                    f.read_exact(cfv_bytes)?;
                    boundary_cfvs.push(cfv);
                }
            }

            iterations.push(IterationData {
                _iteration: iteration,
                _exploitability: exploitability,
                convergence_mode,
                boundary_cfvs,
            });
        }

        Ok(Self {
            num_hands,
            num_boundaries,
            starting_pot,
            iterations,
        })
    }

    fn get_cfv(&self, iteration: usize, boundary_idx: usize, player: usize) -> &[f32] {
        &self.iterations[iteration].boundary_cfvs[boundary_idx * 2 + player]
    }

    fn convergence_mode_at(&self, iteration: usize) -> bool {
        self.iterations[iteration].convergence_mode
    }
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

    let config_filename = Path::new(config_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let pairs_path = format!("data/oracles/{}.dpairs2", config_filename);
    if !Path::new(&pairs_path).exists() {
        eprintln!("Error: Training data not found: {}", pairs_path);
        eprintln!("Build it first:");
        eprintln!(
            "  cargo run --example build_pairs_v2 --release --features \"bincode rayon\" -- {}",
            config_path
        );
        std::process::exit(1);
    }

    let total_start = Instant::now();

    println!("=== Phase 1 v2: Solve with Boundary Pairs ===");
    println!("Config: {}", config_path);
    println!();

    // Load boundary pairs
    println!("Loading pairs: {}", pairs_path);
    let load_start = Instant::now();
    let pairs = BoundaryPairsV2::load(&pairs_path).expect("Failed to load pairs");
    let load_time = load_start.elapsed().as_secs_f64();
    let file_size = std::fs::metadata(&pairs_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {} boundaries, {} iterations, OOP={}, IP={}",
        pairs.num_boundaries,
        pairs.iterations.len(),
        pairs.num_hands[0],
        pairs.num_hands[1],
    );
    println!("  starting_pot: {}", pairs.starting_pot);
    println!(
        "  File size: {:.2} MB, load time: {:.3}s",
        file_size as f64 / 1048576.0,
        load_time,
    );
    println!();

    // Build flop game tree
    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    let (mem_usage, _) = game.memory_usage();
    let memory_mb = mem_usage as f64 / 1024.0 / 1024.0;
    println!(
        "  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0),
        game.num_private_hands(1)
    );
    println!("  Memory: {:.2} MB", memory_mb);
    game.allocate_memory(false);

    if game.num_private_hands(0) != pairs.num_hands[0]
        || game.num_private_hands(1) != pairs.num_hands[1]
    {
        eprintln!("Error: Hand counts don't match between game and pairs file");
        std::process::exit(1);
    }
    println!();

    // Solve: flop-only DCFR with pre-recorded boundary CFVs
    let max_iterations = pairs.iterations.len();
    println!(
        "Solving ({} iterations from recorded pairs)...",
        max_iterations,
    );
    let solve_start = Instant::now();

    for t in 0..max_iterations {
        // Use the exact convergence_mode that was used during build
        let convergence_mode = pairs.convergence_mode_at(t);

        for player in 0..2 {
            // Extract boundary CFVs for this iteration/player
            let boundary_cfvs: Vec<Vec<f32>> = (0..pairs.num_boundaries)
                .map(|b| pairs.get_cfv(t, b, player).to_vec())
                .collect();

            // Use library's exact DCFR with injected boundary CFVs
            solve_step_for_player_replay(
                &game,
                t as u32,
                player,
                convergence_mode,
                &boundary_cfvs,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            let elapsed = solve_start.elapsed().as_secs_f64();
            let per_iter = elapsed / (t + 1) as f64;
            print!(
                "\r  iteration: {} / {} ({:.2}s, {:.4}s/iter, conv_mode={})",
                t + 1,
                max_iterations,
                elapsed,
                per_iter,
                convergence_mode,
            );
            io::stdout().flush().unwrap();
        }
    }
    println!();
    let solve_time = solve_start.elapsed().as_secs_f64();

    // Finalize
    println!("Finalizing...");
    let finalize_start = Instant::now();
    finalize(&mut game);
    let finalize_time = finalize_start.elapsed().as_secs_f64();
    println!("  Finalize: {:.2}s", finalize_time);

    // Save
    let output_path = format!("data/out/{}-pairs2.flop", config_filename);
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let save_start = Instant::now();
    save_data_to_file(&game, "pairs-v2", &output_path, None)
        .expect("Failed to save .flop file");
    let save_time = save_start.elapsed().as_secs_f64();
    println!("  Save: {:.2}s ({})", save_time, output_path);

    let total_time = total_start.elapsed().as_secs_f64();

    println!();
    println!("=== Summary ===");
    println!("Output: {}", output_path);
    println!("Pairs load: {:.3}s", load_time);
    println!(
        "Solve time: {:.2}s ({} iterations, {:.4}s/iter)",
        solve_time,
        max_iterations,
        solve_time / max_iterations as f64,
    );
    println!("Finalize: {:.2}s", finalize_time);
    println!("Save: {:.2}s", save_time);
    println!("Total time: {:.2}s", total_time);
    println!("Memory: {:.2} MB", memory_mb);
}
