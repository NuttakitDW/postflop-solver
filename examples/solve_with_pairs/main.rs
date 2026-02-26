//! Phase 1: Solve a flop game using pre-recorded boundary pairs (.dpairs).
//!
//! The .dpairs file contains the exact (cfreach, cfv) pairs that flowed through
//! each turn boundary during a standard DCFR solve. Since DCFR is deterministic,
//! replaying the recorded cfv at each iteration reproduces the standard solver's
//! flop strategy exactly.
//!
//! Usage:
//!   cargo run --example solve_with_pairs --release --features "bincode rayon" -- config/toy.json
//!
//! Prerequisite: build the training data first:
//!   cargo run --example build_dynamic_oracle --release --features "bincode rayon" -- config/toy.json

#[path = "../common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::collections::HashMap;
use std::env;
use std::io::{self, Read as _, Write as _};
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

// =============================================================================
// DCFR parameters (matches library)
// =============================================================================

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    fn new(t: u32) -> Self {
        let nearest_lower_power_of_4 = match t {
            0 => 0,
            x => 1u32 << ((x.leading_zeros() ^ 31) & !1),
        };
        let t_alpha = (t as i32 - 1).max(0) as f64;
        let t_gamma = (t - nearest_lower_power_of_4) as f64;
        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);
        Self {
            alpha_t: (pow_alpha / (pow_alpha + 1.0)) as f32,
            beta_t: 0.5,
            gamma_t: pow_gamma as f32,
        }
    }
}

fn regret_matching(regrets: &[f32], num_actions: usize, num_hands: usize) -> Vec<f32> {
    let mut strategy = vec![0.0f32; num_actions * num_hands];
    for (s, &r) in strategy.iter_mut().zip(regrets.iter()) {
        *s = r.max(0.0);
    }
    for h in 0..num_hands {
        let mut denom = 0.0f32;
        for a in 0..num_actions {
            denom += strategy[a * num_hands + h];
        }
        if denom > 0.0 {
            for a in 0..num_actions {
                strategy[a * num_hands + h] /= denom;
            }
        } else {
            let uniform = 1.0 / num_actions as f32;
            for a in 0..num_actions {
                strategy[a * num_hands + h] = uniform;
            }
        }
    }
    strategy
}

// =============================================================================
// Boundary pairs loader
// =============================================================================

struct BoundaryPairs {
    num_hands: [usize; 2],
    /// (iteration, amount, player) → cfv vector
    lookup: HashMap<(u32, i32, u32), Vec<f32>>,
    max_iteration: u32,
}

impl BoundaryPairs {
    fn load(path: &str) -> io::Result<Self> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"DPAIRS\0\0" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);
        f.read_exact(&mut buf4)?;
        let num_oop = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_ip = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let total_pairs = u32::from_le_bytes(buf4) as usize;

        let num_hands = [num_oop, num_ip];
        let mut lookup = HashMap::with_capacity(total_pairs);
        let mut max_iteration = 0u32;

        for _ in 0..total_pairs {
            f.read_exact(&mut buf4)?;
            let iteration = u32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let _exploitability = f32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let amount = i32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let player = u32::from_le_bytes(buf4);

            // cfreach [n_opp floats] — read but not stored (we only need cfv)
            let n_opp = num_hands[player as usize ^ 1];
            let mut cfreach_buf = vec![0u8; n_opp * 4];
            f.read_exact(&mut cfreach_buf)?;

            // cfv [n_player floats]
            let n_player = num_hands[player as usize];
            let mut cfv = vec![0.0f32; n_player];
            let cfv_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(cfv.as_mut_ptr() as *mut u8, n_player * 4)
            };
            f.read_exact(cfv_bytes)?;

            if iteration > max_iteration {
                max_iteration = iteration;
            }
            lookup.insert((iteration, amount, player), cfv);
        }

        Ok(Self {
            num_hands,
            lookup,
            max_iteration,
        })
    }

    fn get_cfv(&self, iteration: u32, amount: i32, player: usize) -> &[f32] {
        self.lookup
            .get(&(iteration, amount, player as u32))
            .expect(&format!(
                "Missing pair: iter={}, amount={}, player={}",
                iteration, amount, player
            ))
    }
}

// =============================================================================
// Flop-only DCFR traversal with pre-recorded boundary CFVs
// =============================================================================

fn solve_recursive_with_pairs(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    pairs: &BoundaryPairs,
    iteration: u32,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Turn boundary → return pre-recorded cfv
    if node.is_chance() && node.turn() == NOT_DEALT {
        let cfv = pairs.get_cfv(iteration, node.amount(), player);
        for (r, &v) in result.iter_mut().zip(cfv) {
            r.write(v);
        }
        return;
    }

    // Passthrough
    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        solve_recursive_with_pairs(
            result, game, pairs, iteration, &mut child, player, cfreach, params,
        );
        return;
    }

    let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);

    if node.player() == player {
        for action in 0..num_actions {
            let mut child = node.play(action);
            let cfv = cfv_actions.lock();
            let offset = action * num_hands;
            let action_result = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut MaybeUninit<f32>).add(offset),
                    num_hands,
                )
            };
            solve_recursive_with_pairs(
                action_result, game, pairs, iteration, &mut child, player, cfreach, params,
            );
        }

        let cfv_actions = cfv_actions.lock();
        let strategy = regret_matching(node.regrets(), num_actions, num_hands);

        for h in 0..num_hands {
            let mut weighted = 0.0f32;
            for a in 0..num_actions {
                weighted += strategy[a * num_hands + h] * cfv_actions[a * num_hands + h];
            }
            result[h].write(weighted);
        }
        let result_f32 = unsafe { &*(result as *const _ as *const [f32]) };

        // Update cumulative strategy
        let gamma = params.gamma_t;
        let cum_strategy = node.strategy_mut();
        for (x, y) in cum_strategy.iter_mut().zip(&strategy) {
            *x = *x * gamma + *y;
        }

        // Update cumulative regret (matches library exactly)
        let (alpha, beta) = (params.alpha_t, params.beta_t);
        let cum_regret = node.regrets_mut();
        for (x, y) in cum_regret.iter_mut().zip(cfv_actions.iter()) {
            let coef = if x.is_sign_positive() { alpha } else { beta };
            *x = *x * coef + *y;
        }
        for chunk in cum_regret.chunks_exact_mut(num_hands) {
            for (x, y) in chunk.iter_mut().zip(result_f32) {
                *x -= *y;
            }
        }
    } else {
        // Opponent node
        let opp_strategy = regret_matching(node.regrets(), num_actions, cfreach.len());
        let row_size = cfreach.len();
        let mut cfreach_actions = opp_strategy;
        for a in 0..num_actions {
            for h in 0..row_size {
                cfreach_actions[a * row_size + h] *= cfreach[h];
            }
        }

        for action in 0..num_actions {
            let mut child = node.play(action);
            let cfv = cfv_actions.lock();
            let offset = action * num_hands;
            let action_result = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut MaybeUninit<f32>).add(offset),
                    num_hands,
                )
            };
            solve_recursive_with_pairs(
                action_result, game, pairs, iteration, &mut child, player,
                &cfreach_actions[action * row_size..(action + 1) * row_size],
                params,
            );
        }

        let cfv_actions = cfv_actions.lock();
        for h in 0..num_hands {
            let mut total = 0.0f32;
            for a in 0..num_actions {
                total += cfv_actions[a * num_hands + h];
            }
            result[h].write(total);
        }
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
    let pairs_path = format!("data/oracles/{}.dpairs", config_filename);
    if !Path::new(&pairs_path).exists() {
        eprintln!("Error: Training data not found: {}", pairs_path);
        eprintln!("Build it first:");
        eprintln!(
            "  cargo run --example build_dynamic_oracle --release --features \"bincode rayon\" -- {}",
            config_path
        );
        std::process::exit(1);
    }

    let total_start = Instant::now();

    println!("=== Phase 1: Solve with Boundary Pairs ===");
    println!("Config: {}", config_path);
    println!();

    // Load boundary pairs
    println!("Loading pairs: {}", pairs_path);
    let load_start = Instant::now();
    let pairs = BoundaryPairs::load(&pairs_path).expect("Failed to load pairs");
    let load_time = load_start.elapsed().as_secs_f64();
    let file_size = std::fs::metadata(&pairs_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  {} pairs, iterations 0..{}, OOP={}, IP={}",
        pairs.lookup.len(),
        pairs.max_iteration,
        pairs.num_hands[0],
        pairs.num_hands[1]
    );
    println!(
        "  File size: {:.2} KB, load time: {:.3}s",
        file_size as f64 / 1024.0,
        load_time
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
    let max_iterations = pairs.max_iteration + 1;
    println!(
        "Solving ({} iterations from recorded pairs)...",
        max_iterations
    );
    let solve_start = Instant::now();

    for t in 0..max_iterations {
        let params = DiscountParams::new(t);

        for player in 0..2 {
            let num_hands = game.num_private_hands(player);
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            let mut root = game.root();
            solve_recursive_with_pairs(
                &mut result,
                &game,
                &pairs,
                t,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
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
                per_iter
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
    let output_path = format!("data/out/{}-pairs.flop", config_filename);
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let save_start = Instant::now();
    save_data_to_file(&game, "pairs-oracle", &output_path, None)
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
        solve_time / max_iterations as f64
    );
    println!("Finalize: {:.2}s", finalize_time);
    println!("Save: {:.2}s", save_time);
    println!("Total time: {:.2}s", total_time);
    println!("Memory: {:.2} MB", memory_mb);
}
