use crate::card::NOT_DEALT;
use crate::game::{PostFlopGame, PostFlopNode};
use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;
#[cfg(feature = "logging")]
use std::time::Instant;

#[cfg(feature = "logging")]
use log::debug;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

// =============================================================================
// CFR Debug Logging (env var CFR_LOG=1|2|3)
//   CFR_LOG=1  iteration-level + flop nodes (depth ≤ 2)
//   CFR_LOG=2  all nodes in tree
//   CFR_LOG=3  all nodes + per-hand detail
//   CFR_LOG_ITERS=N  log only first N iterations (default 2)
//
// Logs are saved to: logs/<mode>_<timestamp>.log
// =============================================================================

fn cfr_log_level() -> u32 {
    use std::sync::OnceLock;
    static LEVEL: OnceLock<u32> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("CFR_LOG").ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn cfr_log_max_iters() -> u32 {
    use std::sync::OnceLock;
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CFR_LOG_ITERS").ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

use std::sync::Mutex;

static CFR_LOG_FILE: std::sync::OnceLock<Mutex<std::fs::File>> = std::sync::OnceLock::new();

fn cfr_log_init(mode: &str) {
    if cfr_log_level() == 0 { return; }
    CFR_LOG_FILE.get_or_init(|| {
        let _ = std::fs::create_dir_all("logs");
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let path = format!("logs/{}_{}.log", mode, secs);
        eprintln!("[CFR_LOG] Saving log to: {}", path);
        let file = std::fs::File::create(&path)
            .expect("Failed to create CFR log file");
        Mutex::new(file)
    });
}

fn cfr_log_write(msg: &str) {
    if let Some(file) = CFR_LOG_FILE.get() {
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(f, "{}", msg);
            let _ = f.flush();
        }
    }
}

macro_rules! cfr_log {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        eprintln!("{}", msg);
        cfr_log_write(&msg);
    }};
}

thread_local! {
    static CFR_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn cfr_depth() -> usize {
    CFR_DEPTH.with(|d| d.get())
}

fn cfr_set_depth(d: usize) {
    CFR_DEPTH.with(|c| c.set(d));
}

/// Should we log at the current depth?
fn cfr_should_log() -> bool {
    let level = cfr_log_level();
    if level == 0 { return false; }
    if level >= 2 { return true; }
    // level 1: only log at depth ≤ 2 (flop nodes + turn chance)
    cfr_depth() <= 2
}

fn cfr_indent() -> String {
    "  ".repeat(cfr_depth())
}

/// Summary statistics for a float slice
fn slice_stats(s: &[f32]) -> String {
    if s.is_empty() { return "[]".to_string(); }
    let sum: f64 = s.iter().map(|&x| x as f64).sum();
    let mean = sum / s.len() as f64;
    let min = s.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nz = s.iter().filter(|&&x| x.abs() > 1e-10).count();
    format!("[n={} nz={} sum={:.4} mean={:.6} min={:.4} max={:.4}]",
            s.len(), nz, sum, mean, min, max)
}

/// Per-action mean strategy probabilities
fn strategy_action_means(strategy: &[f32], num_actions: usize) -> String {
    let num_hands = strategy.len() / num_actions;
    if num_hands == 0 { return "[]".to_string(); }
    let means: Vec<String> = (0..num_actions).map(|a| {
        let slice = &strategy[a * num_hands..(a + 1) * num_hands];
        let mean: f64 = slice.iter().map(|&x| x as f64).sum::<f64>() / num_hands as f64;
        format!("a{}={:.4}", a, mean)
    }).collect();
    format!("[{}]", means.join(", "))
}

/// Per-action CFV mean values
fn cfv_action_means(cfv_actions: &[f32], num_actions: usize) -> String {
    let num_hands = cfv_actions.len() / num_actions;
    if num_hands == 0 { return "[]".to_string(); }
    let parts: Vec<String> = (0..num_actions).map(|a| {
        let slice = &cfv_actions[a * num_hands..(a + 1) * num_hands];
        let sum: f64 = slice.iter().map(|&x| x as f64).sum();
        let mean = sum / num_hands as f64;
        let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        format!("a{}:mean={:.4},min={:.4},max={:.4}", a, mean, min, max)
    }).collect();
    format!("[{}]", parts.join(" | "))
}

/// Per-action regret mean values (after update, regret = cfv_action - weighted_cfv)
fn regret_action_means(cfv_actions: &[f32], weighted_cfv: &[f32], num_actions: usize) -> String {
    let num_hands = cfv_actions.len() / num_actions;
    if num_hands == 0 { return "[]".to_string(); }
    let parts: Vec<String> = (0..num_actions).map(|a| {
        let slice = &cfv_actions[a * num_hands..(a + 1) * num_hands];
        let regret_mean: f64 = slice.iter().zip(weighted_cfv.iter()).map(|(&c, &w)| {
            (c - w) as f64
        }).sum::<f64>() / num_hands as f64;
        format!("a{}={:.4}", a, regret_mean)
    }).collect();
    format!("[{}]", parts.join(", "))
}

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    pub fn new(current_iteration: u32) -> Self {
        // 0, 1, 4, 16, 64, 256, ...
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };

        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;

        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);

        // Pure DCFR discounting (no convergence mode overrides)
        let alpha_t = (pow_alpha / (pow_alpha + 1.0)) as f32;
        let beta_t = 0.5;
        let gamma_t = pow_gamma as f32;

        Self {
            alpha_t,
            beta_t,
            gamma_t,
        }
    }
}

/// Performs Discounted CFR algorithm until the given number of iterations or exploitability is
/// satisfied.
///
/// This method returns the exploitability of the obtained strategy.
pub fn solve<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);
    let starting_pot = game.starting_pot() as f32;
    let target_percent = if starting_pot > 0.0 {
        target_exploitability / starting_pot * 100.0
    } else {
        0.0
    };
    #[cfg(feature = "logging")]
    let solve_start = Instant::now();
    if print_progress {
        print!("iteration: 0 / {max_num_iterations} ");
        if starting_pot > 0.0 {
            let current_percent = exploitability / starting_pot * 100.0;
            print!("(exploitability = {current_percent:.2}% | target = {target_percent:.2}%)");
        } else {
            print!("(exploitability = {exploitability:.4e})");
        }
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        // Check if this iteration is a power of 4 (reset iteration)
        // Recalculate exploitability before reset decision to avoid using stale values
        let is_power_of_4 = t > 0 && t == 1u32 << ((t.leading_zeros() ^ 31) & !1);
        if is_power_of_4 {
            #[cfg(feature = "logging")]
            let old_exploitability = exploitability;
            exploitability = compute_exploitability(game);
            #[cfg(feature = "logging")]
            debug!(
                "[{:.2}s] iter={}: POWER_OF_4 detected, recalculated exploitability: {:.4}% -> {:.4}%",
                solve_start.elapsed().as_secs_f64(),
                t,
                old_exploitability / starting_pot * 100.0,
                exploitability / starting_pot * 100.0
            );
        }

        let params = DiscountParams::new(t);

        // alternating updates
        if cfr_log_level() >= 1 && t < cfr_log_max_iters() {
            cfr_log_init("standard");
            cfr_log!("\n{}", "=".repeat(80));
            cfr_log!("=== [STANDARD CFR] ITERATION {} | alpha={:.4} beta={:.4} gamma={:.4} ===",
                      t, params.alpha_t, params.beta_t, params.gamma_t);
        }
        for player in 0..2 {
            if cfr_log_level() >= 1 && t < cfr_log_max_iters() {
                let pname = if player == 0 { "OOP" } else { "IP" };
                cfr_log!("\n--- iter={} player={} ({}) num_hands={} ---",
                          t, player, pname, game.num_private_hands(player));
                cfr_log!("  initial_weights(player): {}",
                          slice_stats(game.initial_weights(player)));
                cfr_log!("  initial_weights(opponent): {}",
                          slice_stats(game.initial_weights(player ^ 1)));
            }
            CFR_DEPTH.with(|d| d.set(0));
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        // Log root node scale factors and regret stats after update (compression only)
        #[cfg(feature = "logging")]
        {
            let in_spike_window = t >= 165 && t <= 200;
            if game.is_compression_enabled() && in_spike_window {
                let strategy_scale = root.strategy_scale();
                let regret_scale = root.regret_scale();

                // Get regret statistics
                let regrets = root.regrets_compressed();
                let (min_regret, max_regret, pos_count, neg_count) = if !regrets.is_empty() {
                    let min = regrets.iter().min().copied().unwrap_or(0);
                    let max = regrets.iter().max().copied().unwrap_or(0);
                    let pos = regrets.iter().filter(|&&r| r > 0).count();
                    let neg = regrets.iter().filter(|&&r| r < 0).count();
                    (min, max, pos, neg)
                } else {
                    (0, 0, 0, 0)
                };

                debug!(
                    "[{:.2}s] iter={}: ROOT NODE: strategy_scale={:.4}, regret_scale={:.4}, regrets(min={}, max={}, pos={}, neg={})",
                    solve_start.elapsed().as_secs_f64(),
                    t,
                    strategy_scale,
                    regret_scale,
                    min_regret,
                    max_regret,
                    pos_count,
                    neg_count
                );
            }
        }

        // Calculate exploitability - more frequently in spike window for debugging
        #[cfg(feature = "logging")]
        let check_exploitability = {
            let in_spike_window = t >= 165 && t <= 200;
            (t + 1) % 10 == 0 || t + 1 == max_num_iterations || in_spike_window
        };
        #[cfg(not(feature = "logging"))]
        let check_exploitability = (t + 1) % 10 == 0 || t + 1 == max_num_iterations;

        if check_exploitability {
            #[cfg(feature = "logging")]
            let old_exploitability = exploitability;
            exploitability = compute_exploitability(game);
            #[cfg(feature = "logging")]
            {
                let old_percent = old_exploitability / starting_pot * 100.0;
                let new_percent = exploitability / starting_pot * 100.0;
                let delta = new_percent - old_percent;

                // Always log in spike window, or if spike detected
                let in_spike_window = t >= 165 && t <= 200;
                if in_spike_window {
                    debug!(
                        "[{:.2}s] iter={}: EXPLOITABILITY: {:.4}% -> {:.4}% (delta: {:+.4}%)",
                        solve_start.elapsed().as_secs_f64(),
                        t + 1,
                        old_percent,
                        new_percent,
                        delta
                    );
                } else if delta > 0.1 {
                    debug!(
                        "[{:.2}s] iter={}: SPIKE DETECTED! exploitability: {:.4}% -> {:.4}% (delta: +{:.4}%)",
                        solve_start.elapsed().as_secs_f64(),
                        t + 1,
                        old_percent,
                        new_percent,
                        delta
                    );
                }
            }
        }

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            if starting_pot > 0.0 {
                let current_percent = exploitability / starting_pot * 100.0;
                print!("(exploitability = {current_percent:.2}% | target = {target_percent:.2}%)");
            } else {
                print!("(exploitability = {exploitability:.4e})");
            }
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
        io::stdout().flush().unwrap();
    }

    finalize(game);

    exploitability
}

/// Proceeds Discounted CFR algorithm for one iteration.
#[inline]
pub fn solve_step<T: Game>(game: &T, current_iteration: u32) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);

    // alternating updates
    for player in 0..2 {
        let mut result = Vec::with_capacity(game.num_private_hands(player));
        solve_recursive(
            result.spare_capacity_mut(),
            game,
            &mut root,
            player,
            game.initial_weights(player ^ 1),
            &params,
        );
    }
}

/// Proceeds Discounted CFR algorithm for a single player within one iteration.
///
/// This enables per-player control needed for boundary pair recording:
/// 1. Record boundary CFVs for player 0 (before any update)
/// 2. Call `solve_step_for_player(game, t, 0)` — updates player 0's regrets
/// 3. Record boundary CFVs for player 1 (after player 0's update)
/// 4. Call `solve_step_for_player(game, t, 1)` — updates player 1's regrets
#[inline]
pub fn solve_step_for_player<T: Game>(
    game: &T,
    current_iteration: u32,
    player: usize,
) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);

    let mut result = Vec::with_capacity(game.num_private_hands(player));
    solve_recursive(
        result.spare_capacity_mut(),
        game,
        &mut root,
        player,
        game.initial_weights(player ^ 1),
        &params,
    );
}

/// Proceeds DCFR for a single player, recording boundary CFVs at turn boundaries.
///
/// This records the CFV that `solve_recursive` computes at each turn boundary
/// chance node (where `turn == NOT_DEALT`), in DFS traversal order.
/// The recorded CFVs are the EXACT values the library computes internally.
pub fn solve_step_for_player_recording(
    game: &PostFlopGame,
    current_iteration: u32,
    player: usize,
) -> Vec<Vec<f32>> {
    if game.is_solved() {
        panic!("Game is already solved");
    }
    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);
    let mut result = Vec::with_capacity(game.num_private_hands(player));
    let mut boundaries = Vec::new();
    let mut boundary_cfreaches = Vec::new();
    solve_recursive_recording(
        result.spare_capacity_mut(),
        game,
        &mut root,
        player,
        game.initial_weights(player ^ 1),
        &params,
        &mut boundaries,
        &mut boundary_cfreaches,
    );
    boundaries
}

/// Like `solve_step_for_player_recording` but also returns opponent reach
/// probabilities (cfreach) at each turn boundary, for NN training data.
///
/// Returns (boundary_cfvs, boundary_cfreaches).
/// - boundary_cfvs[i]: CFV vector for the i-th turn boundary (indexed by player's hands)
/// - boundary_cfreaches[i]: opponent reach probabilities at the i-th turn boundary
pub fn solve_step_for_player_recording_with_cfreach(
    game: &PostFlopGame,
    current_iteration: u32,
    player: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    if game.is_solved() {
        panic!("Game is already solved");
    }
    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);
    let mut result = Vec::with_capacity(game.num_private_hands(player));
    let mut boundaries = Vec::new();
    let mut boundary_cfreaches = Vec::new();
    solve_recursive_recording(
        result.spare_capacity_mut(),
        game,
        &mut root,
        player,
        game.initial_weights(player ^ 1),
        &params,
        &mut boundaries,
        &mut boundary_cfreaches,
    );
    (boundaries, boundary_cfreaches)
}

/// Proceeds DCFR for a single player using pre-recorded boundary CFVs (flop-only).
///
/// At turn boundary chance nodes, uses the pre-recorded CFV from `boundary_cfvs`
/// instead of recursing into the turn/river subtree. All other DCFR operations
/// use the library's exact code, ensuring float-identical results.
pub fn solve_step_for_player_replay(
    game: &PostFlopGame,
    current_iteration: u32,
    player: usize,
    boundary_cfvs: &[Vec<f32>],
) {
    if game.is_solved() {
        panic!("Game is already solved");
    }
    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = DiscountParams::new(current_iteration);
    let mut result = Vec::with_capacity(game.num_private_hands(player));
    let mut boundary_counter = 0usize;
    solve_recursive_replay(
        result.spare_capacity_mut(),
        game,
        &mut root,
        player,
        game.initial_weights(player ^ 1),
        &params,
        boundary_cfvs,
        &mut boundary_counter,
    );
    assert_eq!(boundary_counter, boundary_cfvs.len(),
        "Boundary count mismatch: visited {} expected {}", boundary_counter, boundary_cfvs.len());
}

/// Collects opponent cfreach at each turn boundary node using current strategies.
///
/// Traverses the flop tree without modifying any state (no regret/strategy updates).
/// Returns one cfreach vector per boundary in DFS order.
pub fn collect_boundary_cfreaches(
    game: &PostFlopGame,
    player: usize,
) -> Vec<Vec<f32>> {
    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let mut boundaries = Vec::new();
    collect_cfreaches_recursive(
        game,
        &mut root,
        player,
        game.initial_weights(player ^ 1),
        &mut boundaries,
    );
    boundaries
}

fn collect_cfreaches_recursive(
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    boundaries: &mut Vec<Vec<f32>>,
) {
    if node.is_terminal() {
        return;
    }

    let num_actions = node.num_actions();

    // Turn boundary: record cfreach and stop
    if node.is_chance() && node.turn() == NOT_DEALT {
        boundaries.push(cfreach.to_vec());
        return;
    }

    if num_actions == 1 && !node.is_chance() {
        collect_cfreaches_recursive(game, &mut node.play(0), player, cfreach, boundaries);
        return;
    }

    if node.is_chance() {
        let chance_factor = game.chance_factor(node);
        let cfreach_updated: Vec<f32> = cfreach.iter()
            .map(|&v| v / chance_factor as f32)
            .collect();
        for action in 0..num_actions {
            collect_cfreaches_recursive(
                game, &mut node.play(action), player, &cfreach_updated, boundaries,
            );
        }
    } else if node.player() == player {
        // Current player's node: cfreach passes through unchanged
        for action in 0..num_actions {
            collect_cfreaches_recursive(
                game, &mut node.play(action), player, cfreach, boundaries,
            );
        }
    } else {
        // Opponent's node: split cfreach by opponent strategy
        let mut cfreach_actions = regret_matching(node.regrets(), num_actions);
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        for action in 0..num_actions {
            collect_cfreaches_recursive(
                game, &mut node.play(action), player,
                row(&cfreach_actions, action, row_size),
                boundaries,
            );
        }
    }
}

/// Like `solve_recursive` but records boundary CFVs at turn boundary chance nodes.
fn solve_recursive_recording(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
    boundaries: &mut Vec<Vec<f32>>,
    boundary_cfreaches: &mut Vec<Vec<f32>>,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive_recording(result, game, child, player, cfreach, params, boundaries, boundary_cfreaches);
        return;
    }

    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    if node.is_chance() {
        let chance_factor = game.chance_factor(node);

        #[cfg(feature = "custom-alloc")]
        let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / chance_factor as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // Process children sequentially (deterministic DFS order for boundary recording)
        for action in 0..num_actions {
            solve_recursive_recording(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
                boundaries,
                boundary_cfreaches,
            );
        }

        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        let isomorphic_chances = game.isomorphic_chances(node);
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);
            apply_swap(tmp, swap_list);
            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });
            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });

        // Record boundary CFV and cfreach at turn boundary
        if node.turn() == NOT_DEALT {
            let cfv: Vec<f32> = result.iter()
                .map(|v| unsafe { v.assume_init() })
                .collect();
            boundaries.push(cfv);
            boundary_cfreaches.push(cfreach.to_vec());
        }
    } else if node.player() == player {
        for action in 0..num_actions {
            solve_recursive_recording(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
                boundaries,
                boundary_cfreaches,
            );
        }

        let mut strategy = regret_matching(node.regrets(), num_actions);
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        // update cumulative strategy
        let gamma = params.gamma_t;
        let cum_strategy = node.strategy_mut();
        cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
            *x = *x * gamma + *y;
        });

        // update cumulative regret
        let (alpha, beta) = (params.alpha_t, params.beta_t);
        let cum_regret = node.regrets_mut();
        cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
            let coef = if x.is_sign_positive() { alpha } else { beta };
            *x = *x * coef + *y;
        });
        cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
            sub_slice(row, result);
        });
    } else {
        let mut cfreach_actions = regret_matching(node.regrets(), num_actions);
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        for action in 0..num_actions {
            solve_recursive_recording(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
                boundaries,
                boundary_cfreaches,
            );
        }

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

/// Like `solve_recursive` but uses pre-recorded CFVs at turn boundary chance nodes.
fn solve_recursive_replay(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
    boundary_cfvs: &[Vec<f32>],
    boundary_counter: &mut usize,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Turn boundary → return pre-recorded CFV
    if node.is_chance() && node.turn() == NOT_DEALT {
        let idx = *boundary_counter;
        *boundary_counter += 1;
        let cfv = &boundary_cfvs[idx];
        for (r, &v) in result.iter_mut().zip(cfv.iter()) {
            r.write(v);
        }
        return;
    }

    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive_replay(result, game, child, player, cfreach, params, boundary_cfvs, boundary_counter);
        return;
    }

    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    if node.is_chance() {
        // Non-turn chance node in flop tree (shouldn't normally happen, but handle it)
        let chance_factor = game.chance_factor(node);

        #[cfg(feature = "custom-alloc")]
        let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / chance_factor as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        for action in 0..num_actions {
            solve_recursive_replay(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
                boundary_cfvs,
                boundary_counter,
            );
        }

        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        let isomorphic_chances = game.isomorphic_chances(node);
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);
            apply_swap(tmp, swap_list);
            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });
            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });
    } else if node.player() == player {
        for action in 0..num_actions {
            solve_recursive_replay(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
                boundary_cfvs,
                boundary_counter,
            );
        }

        let mut strategy = regret_matching(node.regrets(), num_actions);
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        // update cumulative strategy
        let gamma = params.gamma_t;
        let cum_strategy = node.strategy_mut();
        cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
            *x = *x * gamma + *y;
        });

        // update cumulative regret
        let (alpha, beta) = (params.alpha_t, params.beta_t);
        let cum_regret = node.regrets_mut();
        cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
            let coef = if x.is_sign_positive() { alpha } else { beta };
            *x = *x * coef + *y;
        });
        cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
            sub_slice(row, result);
        });
    } else {
        let mut cfreach_actions = regret_matching(node.regrets(), num_actions);
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        for action in 0..num_actions {
            solve_recursive_replay(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
                boundary_cfvs,
                boundary_counter,
            );
        }

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

/// Recursively solves the counterfactual values.
fn solve_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    let log = cfr_should_log();
    let depth = cfr_depth();

    // return the counterfactual values when the `node` is terminal
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        if log {
            let r = unsafe { &*(result as *const _ as *const [f32]) };
            cfr_log!("{}[d{}] TERMINAL p={} cfv={} cfreach={}",
                      cfr_indent(), depth, player, slice_stats(r), slice_stats(cfreach));
        }
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
        if log {
            cfr_log!("{}[d{}] PASSTHROUGH (1 action) p={}",
                      cfr_indent(), depth, player);
        }
        let child = &mut node.play(0);
        solve_recursive(result, game, child, player, cfreach, params);
        return;
    }

    // allocate memory for storing the counterfactual values
    #[cfg(feature = "custom-alloc")]
    let cfv_actions = MutexLike::new(Vec::with_capacity_in(num_actions * num_hands, StackAlloc));
    #[cfg(not(feature = "custom-alloc"))]
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    // if the `node` is chance
    if node.is_chance() {
        let chance_factor = game.chance_factor(node);
        if log {
            cfr_log!("{}[d{}] CHANCE p={} actions={} chance_factor={} cfreach={}",
                      cfr_indent(), depth, player, num_actions, chance_factor,
                      slice_stats(cfreach));
        }

        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / chance_factor as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            cfr_set_depth(depth + 1);
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
            );
        });
        cfr_set_depth(depth);

        // use 64-bit floating point values
        #[cfg(feature = "custom-alloc")]
        let mut result_f64 = Vec::with_capacity_in(num_hands, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut result_f64 = Vec::with_capacity(num_hands);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_f64_uninit(result_f64.spare_capacity_mut(), &cfv_actions);
        unsafe { result_f64.set_len(num_hands) };

        // get information about isomorphic chances
        let isomorphic_chances = game.isomorphic_chances(node);

        // process isomorphic chances
        for (i, &isomorphic_index) in isomorphic_chances.iter().enumerate() {
            let swap_list = &game.isomorphic_swap(node, i)[player];
            let tmp = row_mut(&mut cfv_actions, isomorphic_index as usize, num_hands);

            apply_swap(tmp, swap_list);

            result_f64.iter_mut().zip(&*tmp).for_each(|(r, &v)| {
                *r += v as f64;
            });

            apply_swap(tmp, swap_list);
        }

        result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
            r.write(v as f32);
        });

        if log {
            let r = unsafe { &*(result as *const _ as *const [f32]) };
            cfr_log!("{}[d{}] CHANCE RESULT p={} cfv={} iso_chances={}",
                      cfr_indent(), depth, player, slice_stats(r), isomorphic_chances.len());
        }
    }
    // if the current player is `player`
    else if node.player() == player {
        if log {
            let pname = if node.player() == 0 { "OOP" } else { "IP" };
            cfr_log!("{}[d{}] PLAYER_NODE ({}) p={} actions={} cfreach={}",
                      cfr_indent(), depth, pname, player, num_actions,
                      slice_stats(cfreach));
        }

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            cfr_set_depth(depth + 1);
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
            );
        });
        cfr_set_depth(depth);

        // compute the strategy by regret-maching algorithm
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        if log {
            cfr_log!("{}[d{}]   strategy_means: {}",
                      cfr_indent(), depth, strategy_action_means(&strategy, num_actions));
        }

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };

        if log {
            cfr_log!("{}[d{}]   cfv_per_action: {}",
                      cfr_indent(), depth, cfv_action_means(&cfv_actions, num_actions));
        }

        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        if log {
            cfr_log!("{}[d{}]   weighted_cfv: {}",
                      cfr_indent(), depth, slice_stats(result));
            cfr_log!("{}[d{}]   instant_regret: {}",
                      cfr_indent(), depth, regret_action_means(&cfv_actions, result, num_actions));
        }

        if game.is_compression_enabled() {
            // update the cumulative strategy
            let scale = node.strategy_scale();
            let decoder = params.gamma_t * scale / u16::MAX as f32;
            let cum_strategy = node.strategy_compressed_mut();

            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x += (*y as f32) * decoder;
            });

            if !locking.is_empty() {
                strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() {
                        *d = 0.0;
                    }
                })
            }

            let new_scale = encode_unsigned_slice(cum_strategy, &strategy);
            node.set_strategy_scale(new_scale);

            // update the cumulative regret
            let scale = node.regret_scale();
            let alpha_decoder = params.alpha_t * scale / i16::MAX as f32;
            let beta_decoder = params.beta_t * scale / i16::MAX as f32;
            let cum_regret = node.regrets_compressed_mut();

            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                *x += *y as f32 * if *y >= 0 { alpha_decoder } else { beta_decoder };
            });

            cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });

            if !locking.is_empty() {
                cfv_actions.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() {
                        *d = 0.0;
                    }
                })
            }

            let new_scale = encode_signed_slice(cum_regret, &cfv_actions);
            node.set_regret_scale(new_scale);
        } else {
            // update the cumulative strategy
            let gamma = params.gamma_t;
            let cum_strategy = node.strategy_mut();
            cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
                *x = *x * gamma + *y;
            });

            // update the cumulative regret
            let (alpha, beta) = (params.alpha_t, params.beta_t);
            let cum_regret = node.regrets_mut();
            cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
                let coef = if x.is_sign_positive() { alpha } else { beta };
                *x = *x * coef + *y;
            });
            cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });
        }
    }
    // if the current player is not `player`
    else {
        let opp_name = if node.player() == 0 { "OOP" } else { "IP" };
        if log {
            cfr_log!("{}[d{}] OPP_NODE ({}) p={} actions={} cfreach={}",
                      cfr_indent(), depth, opp_name, player, num_actions,
                      slice_stats(cfreach));
        }

        // compute the strategy by regret-matching algorithm
        let mut cfreach_actions = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        if log {
            cfr_log!("{}[d{}]   opp_strategy_means: {}",
                      cfr_indent(), depth, strategy_action_means(&cfreach_actions, num_actions));
        }

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            cfr_set_depth(depth + 1);
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
            );
        });
        cfr_set_depth(depth);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);

        if log {
            let r = unsafe { &*(result as *const _ as *const [f32]) };
            cfr_log!("{}[d{}] OPP_RESULT ({}) cfv={}",
                      cfr_indent(), depth, opp_name, slice_stats(r));
        }
    }
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32, StackAlloc> {
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    unsafe { strategy.set_len(regret.len()) };

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    let uninit = strategy.spare_capacity_mut();
    uninit.iter_mut().zip(regret).for_each(|(s, r)| {
        s.write(max(*r, 0.0));
    });
    unsafe { strategy.set_len(regret.len()) };

    let row_size = regret.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32, StackAlloc> {
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity_in(row_size, StackAlloc);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}

/// Computes the strategy by regret-matching algorithm.
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32> {
    let mut strategy = Vec::with_capacity(regret.len());
    strategy.extend(regret.iter().map(|&r| r.max(0) as f32));

    let row_size = strategy.len() / num_actions;
    let mut denom = Vec::with_capacity(row_size);
    sum_slices_uninit(denom.spare_capacity_mut(), &strategy);
    unsafe { denom.set_len(row_size) };

    let default = 1.0 / num_actions as f32;
    strategy.chunks_exact_mut(row_size).for_each(|row| {
        div_slice(row, &denom, default);
    });

    strategy
}


