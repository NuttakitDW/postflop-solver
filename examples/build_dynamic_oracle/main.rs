//! Build a dynamic oracle by extracting turn boundary matrices under the
//! CURRENT strategy (regret_matching) at each DCFR iteration.
//!
//! Per-player extraction timing (v3): player 1's matrix is extracted AFTER
//! player 0's traversal has updated regrets, matching the library's alternating
//! update order. This produces a perfect match (max diff 0.000003).
//!
//! Outputs:
//!   - .doracle: full per-iteration matrices (for solve_with_dynamic_oracle)
//!   - .dpairs:  actual (cfreach, cfv) pairs (for NN training)
//!   - .flop:    standard-solved tree (ground truth for comparison)
//!
//! Usage:
//!   cargo run --example build_dynamic_oracle --release --features "bincode rayon" -- config/toy.json

#[path = "../common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::io::{self, Write as _};
use std::mem::MaybeUninit;
use std::path::Path;
use std::time::Instant;

// =============================================================================
// DCFR parameters (matches library's DiscountParams)
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

// =============================================================================
// Regret matching (matches library)
// =============================================================================

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
// Boundary pair recording (for .dpairs output)
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
// CFV computation using CURRENT strategy (read-only, for matrix extraction)
// =============================================================================

fn compute_cfv_current_strategy(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        compute_cfv_current_strategy(result, game, &mut child, player, cfreach);
        return;
    }

    if node.is_chance() {
        let chance_factor = game.chance_factor(node);
        let cfreach_scaled: Vec<f32> = cfreach.iter()
            .map(|&v| v / chance_factor as f32)
            .collect();

        let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);
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
            compute_cfv_current_strategy(action_result, game, &mut child, player, &cfreach_scaled);
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
    } else if node.player() == player {
        let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);
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
            compute_cfv_current_strategy(action_result, game, &mut child, player, cfreach);
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
    } else {
        let opp_strategy = regret_matching(node.regrets(), num_actions, cfreach.len());
        let row_size = cfreach.len();
        let mut cfreach_actions = opp_strategy;
        for a in 0..num_actions {
            for h in 0..row_size {
                cfreach_actions[a * row_size + h] *= cfreach[h];
            }
        }

        let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);
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
            compute_cfv_current_strategy(
                action_result, game, &mut child, player,
                &cfreach_actions[action * row_size..(action + 1) * row_size],
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
// Boundary matrix extraction (per-player)
// =============================================================================

fn extract_matrix_for_player(
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    num_hands: [usize; 2],
    player: usize,
) -> Vec<f32> {
    let n_player = num_hands[player];
    let n_opp = num_hands[player ^ 1];

    let node_wrapper = MutexLike::new(node as *mut PostFlopNode as usize);
    let columns: Vec<Vec<f32>> = (0..n_opp)
        .into_par_iter()
        .map(|j| {
            let mut basis = vec![0.0f32; n_opp];
            basis[j] = 1.0;
            let mut result = vec![MaybeUninit::<f32>::uninit(); n_player];
            let node_ptr = *node_wrapper.lock() as *mut PostFlopNode;
            compute_cfv_current_strategy(
                &mut result, game, unsafe { &mut *node_ptr }, player, &basis,
            );
            result.iter().map(|v| unsafe { v.assume_init() }).collect()
        })
        .collect();

    let mut matrix = vec![0.0f32; n_player * n_opp];
    for j in 0..n_opp {
        for i in 0..n_player {
            matrix[i * n_opp + j] = columns[j][i];
        }
    }
    matrix
}

fn extract_boundary_for_player(
    game: &PostFlopGame,
    num_hands: [usize; 2],
    player: usize,
) -> HashMap<i32, Vec<f32>> {
    let mut matrices = HashMap::new();
    let mut root = game.root();
    extract_for_player_recursive(game, &mut root, &mut matrices, num_hands, player);
    matrices
}

fn extract_for_player_recursive(
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    matrices: &mut HashMap<i32, Vec<f32>>,
    num_hands: [usize; 2],
    player: usize,
) {
    if node.is_terminal() { return; }
    if node.is_chance() && node.turn() == NOT_DEALT {
        let amount = node.amount();
        if !matrices.contains_key(&amount) {
            let mat = extract_matrix_for_player(game, node, num_hands, player);
            matrices.insert(amount, mat);
        }
        return;
    }
    for action in 0..node.num_actions() {
        let mut child = node.play(action);
        extract_for_player_recursive(game, &mut child, matrices, num_hands, player);
    }
}

// =============================================================================
// Save formats
// =============================================================================

struct IterationSnapshot {
    iteration: u32,
    exploitability: f32,
    matrices: HashMap<i32, [Vec<f32>; 2]>,
}

fn save_doracle(
    path: &str,
    num_hands: [usize; 2],
    boundary_amounts: &[i32],
    snapshots: &[IterationSnapshot],
) -> io::Result<()> {
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"DORACLE\0")?;
    f.write_all(&3u32.to_le_bytes())?;
    f.write_all(&(num_hands[0] as u32).to_le_bytes())?;
    f.write_all(&(num_hands[1] as u32).to_le_bytes())?;
    f.write_all(&(boundary_amounts.len() as u32).to_le_bytes())?;
    f.write_all(&(snapshots.len() as u32).to_le_bytes())?;

    for &amount in boundary_amounts {
        f.write_all(&amount.to_le_bytes())?;
    }

    for snap in snapshots {
        f.write_all(&snap.iteration.to_le_bytes())?;
        f.write_all(&snap.exploitability.to_le_bytes())?;
        for &amount in boundary_amounts {
            let mats = &snap.matrices[&amount];
            for player in 0..2 {
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        mats[player].as_ptr() as *const u8,
                        mats[player].len() * 4,
                    )
                };
                f.write_all(bytes)?;
            }
        }
    }
    f.flush()?;
    Ok(())
}

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
    let doracle_path = format!("data/oracles/{}.doracle", config_filename);
    let dpairs_path = format!("data/oracles/{}.dpairs", config_filename);
    std::fs::create_dir_all("data/oracles").ok();

    let total_start = Instant::now();

    println!("=== Build Dynamic Oracle v3 (per-player extraction timing) ===");
    println!("Config: {}", config_path);
    #[cfg(feature = "rayon")]
    println!("Threads: {}", rayon::current_num_threads());
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

    // Main loop
    println!("--- Recording boundary evolution (per-player timing) ---");
    let mut snapshots: Vec<IterationSnapshot> = Vec::new();
    let mut pair_records: Vec<(u32, f32, Vec<BoundaryPair>)> = Vec::new();
    let mut exploitability;
    let mut boundary_amounts: Vec<i32> = Vec::new();

    let solve_start = Instant::now();

    for t in 0..config.solver.max_iterations {
        let params = DiscountParams::new(t);
        let mut pairs: Vec<BoundaryPair> = Vec::new();

        // Step 1: Extract M_t[player=0] BEFORE any traversal
        let extract_start = Instant::now();
        let p0_matrices = extract_boundary_for_player(&game, num_hands, 0);
        let p0_extract_time = extract_start.elapsed().as_secs_f64();

        if boundary_amounts.is_empty() {
            boundary_amounts = p0_matrices.keys().copied().collect();
            boundary_amounts.sort();
        }

        // Step 2: Player 0 full-tree traversal (updates player 0's regrets, records pairs)
        {
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands[0]];
            let mut root = game.root();
            solve_recursive_full(
                &mut result, &game, &mut root, 0,
                game.initial_weights(1), &params, &mut pairs,
            );
        }

        // Step 3: Extract M_t[player=1] AFTER player 0's traversal
        let extract_start = Instant::now();
        let p1_matrices = extract_boundary_for_player(&game, num_hands, 1);
        let p1_extract_time = extract_start.elapsed().as_secs_f64();

        // Step 4: Player 1 full-tree traversal (updates player 1's regrets, records pairs)
        {
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands[1]];
            let mut root = game.root();
            solve_recursive_full(
                &mut result, &game, &mut root, 1,
                game.initial_weights(0), &params, &mut pairs,
            );
        }

        // Step 5: Compute exploitability
        exploitability = compute_exploitability(&game);
        let pct = exploitability / starting_pot * 100.0;

        // Combine per-player matrices into snapshot
        let mut combined = HashMap::new();
        for &amount in &boundary_amounts {
            let p0 = p0_matrices[&amount].clone();
            let p1 = p1_matrices[&amount].clone();
            combined.insert(amount, [p0, p1]);
        }

        snapshots.push(IterationSnapshot {
            iteration: t,
            exploitability,
            matrices: combined,
        });
        pair_records.push((t, exploitability, pairs));

        let elapsed = solve_start.elapsed().as_secs_f64();
        print!("\r  iter {}: exploit={:.2}%, extract=({:.2}s+{:.2}s), total={:.1}s    ",
            t, pct, p0_extract_time, p1_extract_time, elapsed);
        io::stdout().flush().unwrap();

        if exploitability <= target_exploitability {
            println!();
            println!("  Converged at iteration {} (exploitability={:.2}%)", t, pct);
            break;
        }
    }
    println!();

    let solve_total = solve_start.elapsed().as_secs_f64();
    let final_exploit = snapshots.last().map(|s| s.exploitability).unwrap_or(0.0);
    let final_pct = final_exploit / starting_pot * 100.0;

    // Save .doracle (for solve_with_dynamic_oracle)
    println!("--- Saving ---");
    save_doracle(&doracle_path, num_hands, &boundary_amounts, &snapshots)
        .expect("Failed to save .doracle");
    let doracle_size = std::fs::metadata(&doracle_path).map(|m| m.len()).unwrap_or(0);

    // Save .dpairs (for NN training / solve_with_pairs)
    save_dpairs(&dpairs_path, num_hands, &pair_records)
        .expect("Failed to save .dpairs");
    let dpairs_size = std::fs::metadata(&dpairs_path).map(|m| m.len()).unwrap_or(0);

    // Save .flop (ground truth)
    let tree_path = format!("data/out/{}.flop", config_filename);
    if let Some(parent) = Path::new(&tree_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    finalize(&mut game);
    let memo = config.output.memo.as_deref().unwrap_or("dynamic-oracle-v3");
    save_data_to_file(&game, memo, &tree_path, None)
        .expect("Failed to save .flop file");
    let flop_size = std::fs::metadata(&tree_path).map(|m| m.len()).unwrap_or(0);

    let total_time = total_start.elapsed().as_secs_f64();
    let total_pairs: usize = pair_records.iter().map(|(_, _, p)| p.len()).sum();

    println!();
    println!("=== Summary ===");
    println!("Board: {}", config.board.flop);
    println!("Boundary amounts: {:?}", boundary_amounts);
    println!("Snapshots: {} (iter 0..{})", snapshots.len(),
        snapshots.last().map(|s| s.iteration).unwrap_or(0));
    println!("Pairs: {} ({} per iter)", total_pairs, total_pairs / snapshots.len().max(1));
    println!("Final exploitability: {:.2}%", final_pct);
    println!("Solve+extract time: {:.2}s", solve_total);
    println!("Total time: {:.2}s", total_time);
    println!();
    println!("Dynamic oracle: {} ({:.2} MB)", doracle_path, doracle_size as f64 / 1048576.0);
    println!("Training pairs: {} ({:.2} KB)", dpairs_path, dpairs_size as f64 / 1024.0);
    println!("Solved tree: {} ({:.2} MB)", tree_path, flop_size as f64 / 1048576.0);
}
