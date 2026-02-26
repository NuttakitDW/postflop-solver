//! Solve a flop game using a dynamic oracle (.doracle) at the turn boundary.
//!
//! Instead of a fixed matrix, uses per-iteration matrices M_t extracted from
//! the standard solver. At iteration t, the oracle provides the same boundary
//! CFVs that the standard solver would produce at iteration t.
//!
//! Usage:
//!   cargo run --example solve_with_dynamic_oracle --release --features "bincode rayon" -- config/toy.json
//!
//! Prerequisite: build the dynamic oracle first:
//!   cargo run --example build_dynamic_oracle --release --features "bincode rayon" -- config/toy.json

#[path = "../common/mod.rs"]
mod common;
#[path = "../poc_precompute/oracle_solver.rs"]
mod oracle_solver;

use common::*;
use oracle_solver::TreeOracle;
use postflop_solver::*;
use std::collections::HashMap;
use std::env;
use std::io::{self, Read as _, Write as _};
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

// =============================================================================
// Dynamic Oracle loader (matches build_dynamic_oracle format)
// =============================================================================

struct IterationSnapshot {
    iteration: u32,
    matrices: HashMap<i32, [Vec<f32>; 2]>,
}

struct DynamicOracle {
    num_hands: [usize; 2],
    boundary_amounts: Vec<i32>,
    snapshots: Vec<IterationSnapshot>,
}

impl DynamicOracle {
    fn load(path: &str) -> io::Result<Self> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"DORACLE\0" {
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
        let num_amounts = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_snapshots = u32::from_le_bytes(buf4) as usize;

        let num_hands = [num_oop, num_ip];
        let mut boundary_amounts = Vec::with_capacity(num_amounts);
        for _ in 0..num_amounts {
            f.read_exact(&mut buf4)?;
            boundary_amounts.push(i32::from_le_bytes(buf4));
        }

        let mut snapshots = Vec::with_capacity(num_snapshots);
        for _ in 0..num_snapshots {
            f.read_exact(&mut buf4)?;
            let iteration = u32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let _exploitability = f32::from_le_bytes(buf4);

            let mut matrices = HashMap::with_capacity(num_amounts);
            for &amount in &boundary_amounts {
                let mut player_matrices: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                for player in 0..2 {
                    let n_player = num_hands[player];
                    let n_opp = num_hands[player ^ 1];
                    let mut data = vec![0.0f32; n_player * n_opp];
                    let bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(
                            data.as_mut_ptr() as *mut u8,
                            data.len() * 4,
                        )
                    };
                    f.read_exact(bytes)?;
                    player_matrices[player] = data;
                }
                matrices.insert(amount, player_matrices);
            }

            snapshots.push(IterationSnapshot {
                iteration,
                matrices,
            });
        }

        Ok(Self {
            num_hands,
            boundary_amounts,
            snapshots,
        })
    }

    /// Get the snapshot index for iteration t.
    /// If t exceeds recorded snapshots, clamp to the last one.
    fn snapshot_for_iter(&self, t: u32) -> usize {
        let idx = t as usize;
        idx.min(self.snapshots.len() - 1)
    }

    /// Build a TreeOracle from the snapshot at iteration t.
    fn oracle_at_iter(&self, t: u32) -> TreeOracle {
        let idx = self.snapshot_for_iter(t);
        let snap = &self.snapshots[idx];
        let matrices = snap.matrices.clone();
        TreeOracle::from_matrices(matrices, self.num_hands)
    }
}

// =============================================================================
// DCFR solve with dynamic oracle (one iteration at a time)
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

fn solve_recursive_with_oracle(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    oracle: &TreeOracle,
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

    // Turn chance node → oracle
    if node.is_chance() && node.turn() == NOT_DEALT {
        oracle.evaluate_turn_boundary(result, node.amount(), player, cfreach);
        return;
    }

    // Passthrough
    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        solve_recursive_with_oracle(result, game, oracle, &mut child, player, cfreach, params);
        return;
    }

    let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);

    if node.player() == player {
        for action in 0..num_actions {
            let mut child = node.play(action);
            let cfv = cfv_actions.lock();
            let offset = action * num_hands;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut MaybeUninit<f32>).add(offset),
                    num_hands,
                )
            };
            solve_recursive_with_oracle(
                dst, game, oracle, &mut child, player, cfreach, params,
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

        let gamma = params.gamma_t;
        let cum_strategy = node.strategy_mut();
        for (x, y) in cum_strategy.iter_mut().zip(&strategy) {
            *x = *x * gamma + *y;
        }

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
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut MaybeUninit<f32>).add(offset),
                    num_hands,
                )
            };
            solve_recursive_with_oracle(
                dst, game, oracle, &mut child, player,
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
    let oracle_path = format!("data/oracles/{}.doracle", config_filename);
    if !Path::new(&oracle_path).exists() {
        eprintln!("Error: Dynamic oracle not found: {}", oracle_path);
        eprintln!("Build it first:");
        eprintln!("  cargo run --example build_dynamic_oracle --release --features \"bincode rayon\" -- {}", config_path);
        std::process::exit(1);
    }

    let total_start = Instant::now();

    println!("=== Dynamic Oracle Solver ===");
    println!("Config: {}", config_path);
    println!("Mode: Dynamic Oracle (flop-only DCFR with per-iteration matrices)");
    println!();

    // Load dynamic oracle
    println!("Loading dynamic oracle: {}", oracle_path);
    let load_start = Instant::now();
    let doracle = DynamicOracle::load(&oracle_path).expect("Failed to load dynamic oracle");
    let load_time = load_start.elapsed().as_secs_f64();
    let file_size = std::fs::metadata(&oracle_path).map(|m| m.len()).unwrap_or(0);
    println!("  {} boundary amounts, {} snapshots (iter 0..{})",
        doracle.boundary_amounts.len(),
        doracle.snapshots.len(),
        doracle.snapshots.last().map(|s| s.iteration).unwrap_or(0));
    println!("  OOP={} hands, IP={} hands", doracle.num_hands[0], doracle.num_hands[1]);
    println!("  File size: {:.2} MB, load time: {:.3}s",
        file_size as f64 / 1_048_576.0, load_time);
    println!();

    // Build game tree
    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();

    let (mem_usage, _) = game.memory_usage();
    let memory_mb = mem_usage as f64 / 1024.0 / 1024.0;
    println!("  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0), game.num_private_hands(1));
    println!("  Memory: {:.2} MB", memory_mb);

    game.allocate_memory(false);

    if game.num_private_hands(0) != doracle.num_hands[0]
        || game.num_private_hands(1) != doracle.num_hands[1]
    {
        eprintln!("Error: Oracle hand counts don't match game");
        std::process::exit(1);
    }
    println!();

    // Solve: flop-only DCFR with dynamic per-iteration oracle
    // Cap iterations to snapshot count — beyond that, clamped snapshots = static Nash oracle
    let max_iterations = (config.solver.max_iterations as usize)
        .min(doracle.snapshots.len()) as u32;
    println!("Solving (dynamic oracle DCFR, {} iterations capped to {} snapshots)...",
        config.solver.max_iterations, max_iterations);
    let solve_start = Instant::now();

    for t in 0..max_iterations {
        let oracle = doracle.oracle_at_iter(t);
        let params = DiscountParams::new(t);

        for player in 0..2 {
            let num_hands = game.num_private_hands(player);
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            let mut root = game.root();
            solve_recursive_with_oracle(
                &mut result,
                &game,
                &oracle,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            let elapsed = solve_start.elapsed().as_secs_f64();
            let per_iter = elapsed / (t + 1) as f64;
            print!("\r  iteration: {} / {} ({:.2}s, {:.4}s/iter)",
                t + 1, max_iterations, elapsed, per_iter);
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
    let output_path = format!("data/out/{}-dynamic.flop", config_filename);
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let save_start = Instant::now();
    save_data_to_file(&game, "dynamic-oracle", &output_path, None)
        .expect("Failed to save .flop file");
    let save_time = save_start.elapsed().as_secs_f64();
    println!("  Save: {:.2}s ({})", save_time, output_path);

    let total_time = total_start.elapsed().as_secs_f64();

    println!();
    println!("=== Summary ===");
    println!("Output: {}", output_path);
    println!("Oracle load: {:.2}s", load_time);
    println!("Solve time: {:.2}s ({} iterations, {:.4}s/iter)",
        solve_time, max_iterations, solve_time / max_iterations as f64);
    println!("Finalize: {:.2}s", finalize_time);
    println!("Save: {:.2}s", save_time);
    println!("Total time: {:.2}s", total_time);
    println!("Memory: {:.2} MB", memory_mb);
}
