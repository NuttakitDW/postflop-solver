//! Verify .bt1 data: replay boundary CFVs from .bt1 file (ignoring cfreach).
//!
//! Same as solve_with_pairs_v2 but reads from .bt1 instead of .dpairs2.
//! This verifies that bt1's recorded CFVs are correct by producing a .flop
//! file that should be byte-identical to the pairs2/standard result.
//!
//! Usage:
//!   cargo run --example solve_with_bt1 --release --features "bincode rayon" -- config/KcQh7s.json

#[path = "../common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::io::{self, Read as _, Write as _};
use std::path::Path;
use std::time::Instant;

// =============================================================================
// .bt1 loader (CFVs only — cfreach is skipped)
// =============================================================================

struct Bt1Data {
    num_hands: [usize; 2],
    num_boundaries: usize,
    starting_pot: f32,
    /// iterations[t].boundary_cfvs[boundary_idx * 2 + player] = cfv vector
    iterations: Vec<Bt1Iteration>,
}

struct Bt1Iteration {
    _iteration: u32,
    _exploitability: f32,
    boundary_cfvs: Vec<Vec<f32>>,
}

impl Bt1Data {
    fn load(path: &str) -> io::Result<Self> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"BT1\0\0\0\0\0" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic (expected BT1)"));
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
            let _reserved = u32::from_le_bytes(buf4);

            let mut boundary_cfvs = Vec::with_capacity(num_boundaries * 2);
            for _b in 0..num_boundaries {
                for player in 0..2 {
                    let opponent = player ^ 1;
                    let n_player = num_hands[player];
                    let n_opponent = num_hands[opponent];

                    // Read CFV
                    let mut cfv = vec![0.0f32; n_player];
                    let cfv_bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            cfv.as_mut_ptr() as *mut u8,
                            n_player * 4,
                        )
                    };
                    f.read_exact(cfv_bytes)?;
                    boundary_cfvs.push(cfv);

                    // Skip cfreach (we don't need it for replay)
                    let mut skip = vec![0u8; n_opponent * 4];
                    f.read_exact(&mut skip)?;
                }
            }

            iterations.push(Bt1Iteration {
                _iteration: iteration,
                _exploitability: exploitability,
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
    let bt1_path = format!("data/bt1/{}.bt1", config_filename);
    if !Path::new(&bt1_path).exists() {
        eprintln!("Error: .bt1 file not found: {}", bt1_path);
        eprintln!("Build it first:");
        eprintln!(
            "  cargo run --example build_bt1 --release --features \"bincode rayon\" -- {}",
            config_path
        );
        std::process::exit(1);
    }

    let total_start = Instant::now();

    println!("=== Verify .bt1: Replay Boundary CFVs ===");
    println!("Config: {}", config_path);
    println!();

    // Load .bt1
    println!("Loading .bt1: {}", bt1_path);
    let load_start = Instant::now();
    let bt1 = Bt1Data::load(&bt1_path).expect("Failed to load .bt1");
    let load_time = load_start.elapsed().as_secs_f64();
    let file_size = std::fs::metadata(&bt1_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {} boundaries, {} iterations, OOP={}, IP={}",
        bt1.num_boundaries,
        bt1.iterations.len(),
        bt1.num_hands[0],
        bt1.num_hands[1],
    );
    println!("  starting_pot: {}", bt1.starting_pot);
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

    if game.num_private_hands(0) != bt1.num_hands[0]
        || game.num_private_hands(1) != bt1.num_hands[1]
    {
        eprintln!("Error: Hand counts don't match between game and .bt1 file");
        std::process::exit(1);
    }
    println!();

    // Solve: flop-only DCFR with bt1's recorded boundary CFVs
    let max_iterations = bt1.iterations.len();
    println!(
        "Solving ({} iterations from .bt1 CFVs)...",
        max_iterations,
    );
    let solve_start = Instant::now();

    for t in 0..max_iterations {
        for player in 0..2 {
            let boundary_cfvs: Vec<Vec<f32>> = (0..bt1.num_boundaries)
                .map(|b| bt1.get_cfv(t, b, player).to_vec())
                .collect();

            solve_step_for_player_replay(
                &game,
                t as u32,
                player,
                &boundary_cfvs,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            let elapsed = solve_start.elapsed().as_secs_f64();
            let per_iter = elapsed / (t + 1) as f64;
            print!(
                "\r  iteration: {} / {} ({:.2}s, {:.4}s/iter)",
                t + 1,
                max_iterations,
                elapsed,
                per_iter,
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
    let output_path = format!("data/out/{}-bt1-replay.flop", config_filename);
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let save_start = Instant::now();
    save_data_to_file(&game, "bt1-replay", &output_path, None)
        .expect("Failed to save .flop file");
    let save_time = save_start.elapsed().as_secs_f64();
    println!("  Save: {:.2}s ({})", save_time, output_path);

    let total_time = total_start.elapsed().as_secs_f64();

    println!();
    println!("=== Summary ===");
    println!("Output: {}", output_path);
    println!("Load: {:.3}s", load_time);
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
    println!();
    println!("Compare against baseline:");
    println!(
        "  cargo run --example compare_flop_files --release --features \"bincode rayon\" -- \
data/out/dpair2/{}-standard.flop {}",
        config_filename, output_path
    );
}
