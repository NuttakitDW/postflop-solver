//! Build boundary pairs (.dpairs) by recording (cfreach, cfv) at each turn
//! boundary during a full-tree DCFR solve.
//!
//! This is the fast version — no matrix extraction, just full-tree traversal
//! with pair recording. ~22x faster than build_dynamic_oracle.
//!
//! Outputs:
//!   - .dpairs: actual (cfreach, cfv) pairs (for solve_with_pairs / NN training)
//!   - .flop:   standard-solved tree (ground truth for comparison)
//!
//! Usage:
//!   cargo run --example build_pairs --release --features "bincode rayon" -- config/toy.json

#[path = "../common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::collections::HashSet;
use std::env;
use std::io::{self, Write as _};
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

fn apply_swap(slice: &mut [f32], swaps: &[(u16, u16)]) {
    for &(i, j) in swaps {
        slice.swap(i as usize, j as usize);
    }
}

// =============================================================================
// Boundary pair recording
// =============================================================================

struct BoundaryPair {
    amount: i32,
    player: usize,
    cfreach: Vec<f32>,
    cfv: Vec<f32>,
}

// =============================================================================
// Full-tree DCFR traversal for a SINGLE player (with regret updates + recording)
// =============================================================================

fn solve_recursive_full(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
    recorder: &mut Vec<BoundaryPair>,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        solve_recursive_full(result, game, &mut child, player, cfreach, params, recorder);
        return;
    }

    let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);

    if node.is_chance() {
        let is_turn_boundary = node.turn() == NOT_DEALT;
        let boundary_amount = if is_turn_boundary { Some(node.amount()) } else { None };
        let boundary_cfreach = if is_turn_boundary { Some(cfreach.to_vec()) } else { None };

        let chance_factor = game.chance_factor(node);
        let cfreach_scaled: Vec<f32> = cfreach.iter()
            .map(|&v| v / chance_factor as f32)
            .collect();

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
            solve_recursive_full(
                action_result, game, &mut child, player, &cfreach_scaled, params, recorder,
            );
        }

        let mut cfv_actions = cfv_actions.lock();
        let mut result_f64 = vec![0.0f64; num_hands];
        for action in 0..num_actions {
            for h in 0..num_hands {
                result_f64[h] += cfv_actions[action * num_hands + h] as f64;
            }
        }

        let iso_chances = game.isomorphic_chances(node);
        for (i, &iso_idx) in iso_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let start = iso_idx as usize * num_hands;
            let tmp = &mut cfv_actions[start..start + num_hands];
            apply_swap(tmp, swap_list);
            for h in 0..num_hands {
                result_f64[h] += tmp[h] as f64;
            }
            apply_swap(tmp, swap_list);
        }

        for h in 0..num_hands {
            result[h].write(result_f64[h] as f32);
        }

        // Record boundary pair
        if let (Some(amount), Some(cfr)) = (boundary_amount, boundary_cfreach) {
            let cfv: Vec<f32> = result.iter()
                .map(|v| unsafe { v.assume_init() })
                .collect();
            recorder.push(BoundaryPair { amount, player, cfreach: cfr, cfv });
        }
    } else if node.player() == player {
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
            solve_recursive_full(
                action_result, game, &mut child, player, cfreach, params, recorder,
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
            let action_result = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut MaybeUninit<f32>).add(offset),
                    num_hands,
                )
            };
            solve_recursive_full(
                action_result, game, &mut child, player,
                &cfreach_actions[action * row_size..(action + 1) * row_size],
                params, recorder,
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
// Discover boundary amounts
// =============================================================================

fn discover_boundary_amounts(node: &mut PostFlopNode) -> Vec<i32> {
    let mut amounts = HashSet::new();
    discover_recursive(node, &mut amounts);
    let mut sorted: Vec<i32> = amounts.into_iter().collect();
    sorted.sort();
    sorted
}

fn discover_recursive(node: &mut PostFlopNode, amounts: &mut HashSet<i32>) {
    if node.is_terminal() { return; }
    if node.is_chance() && node.turn() == NOT_DEALT {
        amounts.insert(node.amount());
        return;
    }
    for action in 0..node.num_actions() {
        let mut child = node.play(action);
        discover_recursive(&mut child, amounts);
    }
}

// =============================================================================
// Save .dpairs
// =============================================================================

fn save_dpairs(
    path: &str,
    num_hands: [usize; 2],
    records: &[(u32, f32, Vec<BoundaryPair>)],
) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    let total_pairs: usize = records.iter().map(|(_, _, p)| p.len()).sum();

    f.write_all(b"DPAIRS\0\0")?;
    f.write_all(&1u32.to_le_bytes())?;
    f.write_all(&(num_hands[0] as u32).to_le_bytes())?;
    f.write_all(&(num_hands[1] as u32).to_le_bytes())?;
    f.write_all(&(total_pairs as u32).to_le_bytes())?;

    for (iteration, exploitability, pairs) in records {
        for pair in pairs {
            f.write_all(&iteration.to_le_bytes())?;
            f.write_all(&exploitability.to_le_bytes())?;
            f.write_all(&pair.amount.to_le_bytes())?;
            f.write_all(&(pair.player as u32).to_le_bytes())?;
            let cfreach_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    pair.cfreach.as_ptr() as *const u8, pair.cfreach.len() * 4,
                )
            };
            f.write_all(cfreach_bytes)?;
            let cfv_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    pair.cfv.as_ptr() as *const u8, pair.cfv.len() * 4,
                )
            };
            f.write_all(cfv_bytes)?;
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
    let target_exploitability =
        tree_config.starting_pot as f32 * config.solver.target_exploitability_percent / 100.0;
    let starting_pot = tree_config.starting_pot as f32;

    let config_filename = Path::new(config_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let dpairs_path = format!("data/oracles/{}.dpairs", config_filename);
    std::fs::create_dir_all("data/oracles").ok();

    let total_start = Instant::now();

    println!("=== Build Boundary Pairs (fast, no matrix extraction) ===");
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

    let boundary_amounts = discover_boundary_amounts(&mut game.root());
    println!("  Boundary amounts: {:?}", boundary_amounts);
    println!();

    // Main solve loop
    println!("--- Solving + recording pairs ---");
    let mut pair_records: Vec<(u32, f32, Vec<BoundaryPair>)> = Vec::new();
    let mut exploitability;

    let solve_start = Instant::now();

    for t in 0..config.solver.max_iterations {
        let params = DiscountParams::new(t);
        let mut pairs: Vec<BoundaryPair> = Vec::new();

        // Player 0 full-tree traversal (records pairs)
        {
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands[0]];
            let mut root = game.root();
            solve_recursive_full(
                &mut result, &game, &mut root, 0,
                game.initial_weights(1), &params, &mut pairs,
            );
        }

        // Player 1 full-tree traversal (records pairs)
        {
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands[1]];
            let mut root = game.root();
            solve_recursive_full(
                &mut result, &game, &mut root, 1,
                game.initial_weights(0), &params, &mut pairs,
            );
        }

        // Compute exploitability
        exploitability = compute_exploitability(&game);
        let pct = exploitability / starting_pot * 100.0;

        pair_records.push((t, exploitability, pairs));

        let elapsed = solve_start.elapsed().as_secs_f64();
        let per_iter = elapsed / (t + 1) as f64;
        if (t + 1) % 10 == 0 || t + 1 == config.solver.max_iterations || exploitability <= target_exploitability {
            print!("\r  iter {}: exploit={:.2}%, {:.2}s ({:.4}s/iter)    ",
                t, pct, elapsed, per_iter);
            io::stdout().flush().unwrap();
        }

        if exploitability <= target_exploitability {
            println!();
            println!("  Converged at iteration {} (exploitability={:.2}%)", t, pct);
            break;
        }
    }
    println!();

    let solve_total = solve_start.elapsed().as_secs_f64();
    let num_iters = pair_records.len();
    let total_pairs: usize = pair_records.iter().map(|(_, _, p)| p.len()).sum();
    let final_exploit = pair_records.last().map(|(_, e, _)| *e).unwrap_or(0.0);
    let final_pct = final_exploit / starting_pot * 100.0;

    // Save .dpairs
    println!("--- Saving ---");
    save_dpairs(&dpairs_path, num_hands, &pair_records)
        .expect("Failed to save .dpairs");
    let dpairs_size = std::fs::metadata(&dpairs_path).map(|m| m.len()).unwrap_or(0);

    // Save .flop (ground truth)
    let tree_path = format!("data/out/{}.flop", config_filename);
    if let Some(parent) = Path::new(&tree_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    finalize(&mut game);
    let memo = config.output.memo.as_deref().unwrap_or("build-pairs");
    save_data_to_file(&game, memo, &tree_path, None)
        .expect("Failed to save .flop file");
    let flop_size = std::fs::metadata(&tree_path).map(|m| m.len()).unwrap_or(0);

    let total_time = total_start.elapsed().as_secs_f64();

    println!();
    println!("=== Summary ===");
    println!("Board: {}", config.board.flop);
    println!("Boundary amounts: {:?}", boundary_amounts);
    println!("Iterations: {}", num_iters);
    println!("Pairs: {} ({} per iter)", total_pairs, total_pairs / num_iters.max(1));
    println!("Final exploitability: {:.2}%", final_pct);
    println!("Solve time: {:.2}s ({:.4}s/iter)", solve_total, solve_total / num_iters as f64);
    println!("Total time: {:.2}s", total_time);
    println!();
    println!("Training pairs: {} ({:.2} KB)", dpairs_path, dpairs_size as f64 / 1024.0);
    println!("Solved tree: {} ({:.2} MB)", tree_path, flop_size as f64 / 1048576.0);
}
