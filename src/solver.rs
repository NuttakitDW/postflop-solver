use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::time::Instant;

#[cfg(feature = "onnx")]
use crate::action_tree::Action;
#[cfg(feature = "onnx")]
use crate::card::*;
#[cfg(feature = "onnx")]
use crate::game::*;
#[cfg(feature = "onnx")]
use crate::oracle;

#[cfg(feature = "logging")]
use log::debug;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

struct DiscountParams {
    alpha_t: f32,
    beta_t: f32,
    gamma_t: f32,
}

impl DiscountParams {
    pub fn new(current_iteration: u32, convergence_mode: bool) -> Self {
        // 0, 1, 4, 16, 64, 256, ...
        let nearest_lower_power_of_4 = match current_iteration {
            0 => 0,
            x => 1 << ((x.leading_zeros() ^ 31) & !1),
        };

        let t_alpha = (current_iteration as i32 - 1).max(0) as f64;
        let t_gamma = (current_iteration - nearest_lower_power_of_4) as f64;

        let pow_alpha = t_alpha * t_alpha.sqrt();
        let pow_gamma = (t_gamma / (t_gamma + 1.0)).powi(3);

        // In convergence mode (exploitability < 1%):
        // - Higher alpha_t floor (0.9) = more conservative positive regret updates
        // - Higher beta_t (0.9 vs 0.5) = preserve negative regrets to prevent imbalance
        // - Higher gamma_t floor (0.9) = more strategy retention
        // This prevents oscillations/spikes when close to optimal solution
        let (alpha_t, beta_t, gamma_t) = if convergence_mode {
            let alpha = ((pow_alpha / (pow_alpha + 1.0)) as f32).max(0.9);
            let beta = 0.9; // Preserve negative regrets (default is 0.5 which causes imbalance)
            let gamma = (pow_gamma as f32).max(0.9);
            (alpha, beta, gamma)
        } else {
            let alpha = (pow_alpha / (pow_alpha + 1.0)) as f32;
            let beta = 0.5;
            let gamma = pow_gamma as f32;
            (alpha, beta, gamma)
        };

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
    let solve_start = Instant::now();
    // Once exploitability drops below 1%, enter convergence mode
    // This applies higher floors to alpha_t and gamma_t to prevent oscillations
    let mut convergence_mode = false;

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

        // Once exploitability drops below 1.0%, enter convergence mode
        // This applies min floors: alpha_t >= 0.9, gamma_t >= 0.9 to prevent spikes
        let current_percent = exploitability / starting_pot * 100.0;
        if starting_pot > 0.0 && current_percent < 1.0 {
            if !convergence_mode {
                #[cfg(feature = "logging")]
                debug!(
                    "[{:.2}s] iter={}: exploitability dropped below 1% ({:.4}%), entering CONVERGENCE MODE (alpha_t >= 0.9, gamma_t >= 0.9)",
                    solve_start.elapsed().as_secs_f64(),
                    t,
                    current_percent
                );
            }
            convergence_mode = true;
        }
        let params = DiscountParams::new(t, convergence_mode);

        // Log parameters for every iteration in debug mode
        #[cfg(feature = "logging")]
        {
            // Log at power-of-4, or every iteration in spike investigation window
            let in_spike_window = t >= 165 && t <= 200;
            if is_power_of_4 || in_spike_window {
                debug!(
                    "[{:.2}s] iter={}: convergence_mode={}, current_percent={:.4}%, alpha_t={:.6}, beta_t={:.6}, gamma_t={:.6}",
                    solve_start.elapsed().as_secs_f64(),
                    t,
                    convergence_mode,
                    current_percent,
                    params.alpha_t,
                    params.beta_t,
                    params.gamma_t
                );
            }
        }

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
    let params = DiscountParams::new(current_iteration, false);

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

/// Recursively solves the counterfactual values.
fn solve_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // return the counterfactual values when the `node` is terminal
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // simply recurse when the number of actions is one
    if num_actions == 1 && !node.is_chance() {
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
        // update the reach probabilities
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_updated = Vec::with_capacity_in(cfreach.len(), StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
            );
        });

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
    }
    // if the current player is `player`
    else if node.player() == player {
        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
            );
        });

        // compute the strategy by regret-maching algorithm
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

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
        // compute the strategy by regret-matching algorithm
        let mut cfreach_actions = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        // update the reach probabilities
        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        // compute the counterfactual values of each action
        for_each_child(node, |action| {
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
            );
        });

        // sum up the counterfactual values
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
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

// =============================================================================
// Deepstack solver: uses ONNX oracle at turn chance nodes
// =============================================================================

#[cfg(feature = "onnx")]
/// Solves a postflop game using deepstack approach: at turn chance nodes, the oracle
/// neural network predicts CFVs instead of recursing into turn+river subtrees.
///
/// Returns 0.0 (no exploitability computation in deepstack mode).
pub fn solve_deepstack(
    game: &mut PostFlopGame,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
    oracle: &oracle::OnnxOracle,
) -> f32 {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    // Precompute combo-to-hand mappings for both players
    let combo_to_hand = [
        oracle::build_combo_to_hand(game, 0),
        oracle::build_combo_to_hand(game, 1),
    ];

    let mut root = game.root();
    let solve_timer = Instant::now();
    let mut exploitability = compute_exploitability(game);
    let starting_pot = game.tree_config().starting_pot as f32;
    let target_percent = if starting_pot > 0.0 {
        target_exploitability / starting_pot * 100.0
    } else {
        0.0
    };

    if print_progress {
        print!("iteration: 0 / {max_num_iterations} ");
        if starting_pot > 0.0 {
            let current_percent = exploitability / starting_pot * 100.0;
            print!(
                "(exploitability = {current_percent:.2}% | target = {target_percent:.2}%) [{:.1}s] [deepstack]",
                solve_timer.elapsed().as_secs_f64()
            );
        }
        io::stdout().flush().unwrap();
    }

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t, false);

        for player in 0..2 {
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive_deepstack(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
                oracle,
                &combo_to_hand,
            );
        }

        // Compute exploitability every 50 iterations or on the last iteration
        // (exploitability check traverses full tree including turn+river, so it's expensive)
        let check_exploitability = (t + 1) % 50 == 0 || t + 1 == max_num_iterations;
        if check_exploitability {
            exploitability = compute_exploitability(game);
        }

        oracle.reset_stats();

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            if starting_pot > 0.0 {
                let current_percent = exploitability / starting_pot * 100.0;
                print!(
                    "(exploitability = {current_percent:.2}% | target = {target_percent:.2}%) [{:.1}s] [deepstack]",
                    solve_timer.elapsed().as_secs_f64(),
                );
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

#[cfg(feature = "onnx")]
fn solve_recursive_deepstack(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
    oracle: &oracle::OnnxOracle,
    combo_to_hand: &[Vec<usize>; 2],
) {
    // Terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Single action pass-through
    if num_actions == 1 && !node.is_chance() {
        let child = &mut node.play(0);
        solve_recursive_deepstack(result, game, child, player, cfreach, params, oracle, combo_to_hand);
        return;
    }

    // Turn chance node: use oracle instead of recursing into turn+river subtree
    if node.is_chance() && node.turn() == NOT_DEALT {
        oracle_predict_turn_cfv(result, game, node, player, cfreach, oracle, combo_to_hand);
        return;
    }

    // Use MutexLike for parallel access (same pattern as standard solve_recursive)
    let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

    if node.is_chance() {
        // River chance node: parallel recursive handling
        let mut cfreach_updated = Vec::with_capacity(cfreach.len());
        mul_slice_scalar_uninit(
            cfreach_updated.spare_capacity_mut(),
            cfreach,
            1.0 / game.chance_factor(node) as f32,
        );
        unsafe { cfreach_updated.set_len(cfreach.len()) };

        for_each_child(node, |action| {
            solve_recursive_deepstack(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
                oracle,
                combo_to_hand,
            );
        });

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
        // Current player's node: parallel compute + regret update
        for_each_child(node, |action| {
            solve_recursive_deepstack(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
                oracle,
                combo_to_hand,
            );
        });

        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        if game.is_compression_enabled() {
            let scale = node.strategy_scale();
            let decoder = params.gamma_t * scale / u16::MAX as f32;
            let cum_strategy = node.strategy_compressed_mut();
            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x += (*y as f32) * decoder;
            });
            if !locking.is_empty() {
                strategy.iter_mut().zip(locking).for_each(|(d, s)| {
                    if s.is_sign_positive() { *d = 0.0; }
                })
            }
            let new_scale = encode_unsigned_slice(cum_strategy, &strategy);
            node.set_strategy_scale(new_scale);

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
                    if s.is_sign_positive() { *d = 0.0; }
                })
            }
            let new_scale = encode_signed_slice(cum_regret, &cfv_actions);
            node.set_regret_scale(new_scale);
        } else {
            let gamma = params.gamma_t;
            let cum_strategy = node.strategy_mut();
            cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
                *x = *x * gamma + *y;
            });

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
    } else {
        // Opponent's node: parallel compute
        let mut cfreach_actions = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut cfreach_actions, locking);

        let row_size = cfreach.len();
        cfreach_actions.chunks_exact_mut(row_size).for_each(|row| {
            mul_slice(row, cfreach);
        });

        for_each_child(node, |action| {
            solve_recursive_deepstack(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
                oracle,
                combo_to_hand,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

#[cfg(feature = "onnx")]
/// At a turn chance node, use the oracle to predict CFVs for all turn cards.
fn oracle_predict_turn_cfv(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    oracle: &oracle::OnnxOracle,
    _combo_to_hand: &[Vec<usize>; 2],
) {
    let num_actions = node.num_actions();
    let num_hands = result.len();
    let opponent = player ^ 1;

    // Build full 1326-element reach arrays from solver's per-hand arrays
    let mut reach_oop_all = vec![0.0f32; 1326];
    let mut reach_ip_all = vec![0.0f32; 1326];

    // Divide cfreach by chance_factor (same as normal chance handling)
    let chance_div = 1.0 / game.chance_factor(node) as f32;

    // cfreach is opponent's reach; initial_weights[player] is player's reach approximation
    let player_weights = game.initial_weights(player);
    for (hand_idx, &(c1, c2)) in game.private_cards(player).iter().enumerate() {
        let combo_idx = card_pair_to_index(c1, c2);
        if player == 0 {
            reach_oop_all[combo_idx] = player_weights[hand_idx] * chance_div;
        } else {
            reach_ip_all[combo_idx] = player_weights[hand_idx] * chance_div;
        }
    }
    for (hand_idx, &(c1, c2)) in game.private_cards(opponent).iter().enumerate() {
        let combo_idx = card_pair_to_index(c1, c2);
        if opponent == 0 {
            reach_oop_all[combo_idx] = cfreach[hand_idx] * chance_div;
        } else {
            reach_ip_all[combo_idx] = cfreach[hand_idx] * chance_div;
        }
    }

    // Collect turn cards from chance children
    let mut turn_cards = Vec::with_capacity(num_actions);
    for action in 0..num_actions {
        let child = node.play(action);
        if let Action::Chance(card) = child.prev_action() {
            turn_cards.push(card);
        }
    }

    // Extract features and run oracle
    let feat_start = Instant::now();
    let (combo_feat, global_feat) =
        oracle::extract_features(game, &reach_oop_all, &reach_ip_all, &turn_cards);
    oracle.record_feature_time(feat_start.elapsed().as_micros() as u64);

    let oracle_output = oracle
        .predict(&combo_feat, &global_feat)
        .expect("Oracle prediction failed");

    // Map oracle output to per-action CFVs, then sum across turn cards
    // oracle_output layout: [batch * 1326 * 2], where dim 2 = [cfv_oop, cfv_ip]
    let player_cfv_idx = player; // 0 for OOP, 1 for IP
    let mut result_f64 = vec![0.0f64; num_hands];

    // Allocate space for per-action CFVs (needed for isomorphism)
    let mut cfv_actions = vec![0.0f32; num_actions * num_hands];

    for (action, &_turn_card) in turn_cards.iter().enumerate() {
        let batch_offset = action * 1326 * 2;

        for (hand_idx, &(c1, c2)) in game.private_cards(player).iter().enumerate() {
            let combo_idx = card_pair_to_index(c1, c2);
            let cfv = oracle_output[batch_offset + combo_idx * 2 + player_cfv_idx];
            cfv_actions[action * num_hands + hand_idx] = cfv;
            result_f64[hand_idx] += cfv as f64;
        }
    }

    // Handle isomorphic chances (same as normal solver)
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

    // Write final result
    result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
        r.write(v as f32);
    });
}
