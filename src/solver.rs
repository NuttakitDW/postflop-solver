use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;
#[cfg(any(feature = "onnx", feature = "logging"))]
use std::time::Instant;

#[cfg(feature = "onnx")]
use crate::action_tree::Action;
#[cfg(feature = "onnx")]
use crate::card::*;
#[cfg(feature = "onnx")]
use crate::game::*;
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
    #[cfg(feature = "logging")]
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
// Bucketed deepstack solver
// =============================================================================

#[cfg(feature = "onnx")]
use crate::bucketing::{
    compute_board_features, compute_buckets, expand_cfv_from_buckets, project_range_to_buckets,
    BucketMapping, DEFAULT_K,
};
#[cfg(feature = "onnx")]
use crate::net::TurnValueNet;

/// Pre-computed bucket mappings for all possible turn cards on a given flop.
#[cfg(feature = "onnx")]
pub struct BucketCache {
    /// Indexed by card (0..52). `None` for flop cards.
    mappings: Vec<Option<BucketMapping>>,
    k: usize,
}

#[cfg(feature = "onnx")]
impl BucketCache {
    /// Pre-compute bucket mappings for all valid turn cards on the given flop.
    pub fn new(flop: &[Card; 3], k: usize) -> Self {
        let flop_mask: u64 = (1u64 << flop[0]) | (1u64 << flop[1]) | (1u64 << flop[2]);
        let mut mappings: Vec<Option<BucketMapping>> = (0..52).map(|_| None).collect();

        for turn in 0u8..52 {
            if flop_mask & (1u64 << turn) != 0 {
                continue;
            }
            let board = [flop[0], flop[1], flop[2], turn];
            mappings[turn as usize] = Some(compute_buckets(&board, k));
        }

        Self { mappings, k }
    }

    /// Get the bucket mapping for a given turn card.
    pub fn get(&self, turn_card: Card) -> &BucketMapping {
        self.mappings[turn_card as usize]
            .as_ref()
            .expect("No bucket mapping for this turn card (likely a flop card)")
    }
}

#[cfg(feature = "onnx")]
/// Solves a postflop game using the bucketed deepstack approach: at turn chance nodes,
/// the bucketed value network predicts CFVs instead of recursing into turn+river subtrees.
///
/// Returns the final exploitability.
pub fn solve_bucketed(
    game: &mut PostFlopGame,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
    net: &TurnValueNet,
) -> f32 {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    // Pre-compute bucket mappings for all 49 turn cards
    let flop = game.card_config().flop;
    if print_progress {
        print!("Pre-computing bucket mappings...");
        io::stdout().flush().unwrap();
    }
    let bucket_cache = BucketCache::new(&flop, DEFAULT_K);
    if print_progress {
        println!(" done.");
    }

    let mut root = game.root();
    let solve_timer = Instant::now();
    let mut exploitability =
        compute_exploitability_bucketed(game, net, &bucket_cache);
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
                "(exploitability = {current_percent:.2}% | target = {target_percent:.2}%) [{:.1}s] [bucketed]",
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
            solve_recursive_bucketed(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
                net,
                &bucket_cache,
            );
        }

        let check_exploitability = (t + 1) % 50 == 0 || t + 1 == max_num_iterations;
        if check_exploitability {
            exploitability = compute_exploitability_bucketed(game, net, &bucket_cache);
        }

        net.reset_stats();

        if print_progress {
            print!("\riteration: {} / {} ", t + 1, max_num_iterations);
            if starting_pot > 0.0 {
                let current_percent = exploitability / starting_pot * 100.0;
                print!(
                    "(exploitability = {current_percent:.2}% | target = {target_percent:.2}%) [{:.1}s] [bucketed]",
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

    finalize_bucketed(game, net, &bucket_cache);

    exploitability
}

#[cfg(feature = "onnx")]
fn solve_recursive_bucketed(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
    net: &TurnValueNet,
    bucket_cache: &BucketCache,
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
        solve_recursive_bucketed(result, game, child, player, cfreach, params, net, bucket_cache);
        return;
    }

    // Turn chance node: use bucketed net instead of recursing into turn+river subtree
    if node.is_chance() && node.turn() == NOT_DEALT {
        bucketed_predict_turn_cfv(result, game, node, player, cfreach, net, bucket_cache);
        return;
    }

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
            solve_recursive_bucketed(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                params,
                net,
                bucket_cache,
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
            solve_recursive_bucketed(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                params,
                net,
                bucket_cache,
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
            solve_recursive_bucketed(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                row(&cfreach_actions, action, row_size),
                params,
                net,
                bucket_cache,
            );
        });

        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        sum_slices_uninit(result, &cfv_actions);
    }
}

#[cfg(feature = "onnx")]
/// At a turn chance node, use the bucketed value network to predict CFVs.
///
/// For each turn card:
/// 1. Project ranges to buckets using pre-computed BucketMapping
/// 2. Build input vector: board_features(15) + range_oop(K) + range_ip(K)
/// 3. Run network inference
/// 4. Expand bucket CFVs back to per-combo, then map to per-hand
pub(crate) fn bucketed_predict_turn_cfv(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &PostFlopNode,
    player: usize,
    cfreach: &[f32],
    net: &TurnValueNet,
    bucket_cache: &BucketCache,
) {
    let num_actions = node.num_actions();
    let num_hands = result.len();
    let opponent = player ^ 1;

    let pot = game.tree_config().starting_pot as f32;
    let stack = game.tree_config().effective_stack as f32;
    let flop = game.card_config().flop;

    // Build full 1326-element reach arrays
    let mut reach_oop_all = [0.0f32; 1326];
    let mut reach_ip_all = [0.0f32; 1326];

    // Use full-scale reaches (no chance_div scaling).
    // Training data stores reaches at full scale, so inference must match.
    // The averaging across turn cards is handled by dividing by total_turn_cards below.
    let player_weights = game.initial_weights(player);
    for (hand_idx, &(c1, c2)) in game.private_cards(player).iter().enumerate() {
        let combo_idx = card_pair_to_index(c1, c2);
        if player == 0 {
            reach_oop_all[combo_idx] = player_weights[hand_idx];
        } else {
            reach_ip_all[combo_idx] = player_weights[hand_idx];
        }
    }
    for (hand_idx, &(c1, c2)) in game.private_cards(opponent).iter().enumerate() {
        let combo_idx = card_pair_to_index(c1, c2);
        if opponent == 0 {
            reach_oop_all[combo_idx] = cfreach[hand_idx];
        } else {
            reach_ip_all[combo_idx] = cfreach[hand_idx];
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

    // For each turn card: bucket → infer → expand
    let player_cfv_idx = player; // 0 for OOP, 1 for IP
    let mut result_f64 = vec![0.0f64; num_hands];
    let mut cfv_actions = vec![0.0f32; num_actions * num_hands];
    let k = bucket_cache.k;

    for (action, &turn_card) in turn_cards.iter().enumerate() {
        let board = [flop[0], flop[1], flop[2], turn_card];
        let mapping = bucket_cache.get(turn_card);

        // Project reaches to buckets
        let bucket_range_oop = project_range_to_buckets(&reach_oop_all, mapping);
        let bucket_range_ip = project_range_to_buckets(&reach_ip_all, mapping);

        // Build input: board_features(15) + range_oop(K) + range_ip(K)
        let board_features = compute_board_features(&board, pot, stack);
        let mut input = Vec::with_capacity(15 + 2 * k);
        input.extend_from_slice(&board_features);
        input.extend_from_slice(&bucket_range_oop);
        input.extend_from_slice(&bucket_range_ip);

        // Inference
        let output = net.predict(&input).expect("Net prediction failed");

        // Split output into OOP and IP bucket CFVs
        let cfv_oop_buckets = &output[..k];
        let cfv_ip_buckets = &output[k..2 * k];

        // Expand bucket CFVs to per-combo (1326)
        let cfv_all = if player_cfv_idx == 0 {
            expand_cfv_from_buckets(cfv_oop_buckets, mapping)
        } else {
            expand_cfv_from_buckets(cfv_ip_buckets, mapping)
        };

        // Map per-combo CFVs to per-hand (solver's hand indexing)
        for (hand_idx, &(c1, c2)) in game.private_cards(player).iter().enumerate() {
            let combo_idx = card_pair_to_index(c1, c2);
            let cfv = cfv_all[combo_idx] * pot; // denormalize
            cfv_actions[action * num_hands + hand_idx] = cfv;
            result_f64[hand_idx] += cfv as f64;
        }
    }

    // Handle isomorphic chances
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

    // Normalize: divide by total number of turn cards (actions + isomorphic)
    // to get the correct chance-weighted average CFV.
    //
    // The standard solver divides cfreach by chance_factor before recursing into
    // each child, making each child's CFV proportional to 1/N. Summing gives the
    // correct average. Here we sum full-scale per-turn CFVs, so we must divide.
    let total_turn_cards = (num_actions + isomorphic_chances.len()) as f64;

    // Write final result
    result.iter_mut().zip(&result_f64).for_each(|(r, &v)| {
        r.write((v / total_turn_cards) as f32);
    });
}

// =============================================================================
// Tests: Mock CFV injection to prove deepstack integration correctness
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_tree::*;
    use crate::card::NOT_DEALT;
    use crate::range::flop_from_str;
    use crate::CardConfig;
    use crate::game::{PostFlopGame, PostFlopNode};
    use crate::range::Range;
    use crate::BetSizeOptions;
    use std::collections::HashMap;
    use std::mem::MaybeUninit;
    use std::sync::Mutex;

    type CfvMap = HashMap<(usize, usize), Vec<f32>>;

    fn node_key(node: &PostFlopNode) -> usize {
        node as *const PostFlopNode as usize
    }

    /// Identical to `solve_recursive` but specialized for PostFlopGame.
    /// At turn chance nodes, delegates to the original `solve_recursive` for the
    /// full subtree computation, then captures the result into a shared HashMap.
    fn solve_recursive_capture(
        result: &mut [MaybeUninit<f32>],
        game: &PostFlopGame,
        node: &mut PostFlopNode,
        player: usize,
        cfreach: &[f32],
        params: &DiscountParams,
        captured: &Mutex<CfvMap>,
    ) {
        if node.is_terminal() {
            game.evaluate(result, node, player, cfreach);
            return;
        }

        let num_actions = node.num_actions();
        let num_hands = result.len();

        if num_actions == 1 && !node.is_chance() {
            let child = &mut node.play(0);
            solve_recursive_capture(result, game, child, player, cfreach, params, captured);
            return;
        }

        // Turn chance node: use standard solve_recursive, then capture the result
        if node.is_chance() && node.turn() == NOT_DEALT {
            solve_recursive(result, game, node, player, cfreach, params);

            // Capture the computed CFVs
            let key = (node_key(node), player);
            let result_vec: Vec<f32> = result
                .iter()
                .map(|r| unsafe { r.assume_init() })
                .collect();
            captured.lock().unwrap().insert(key, result_vec);
            return;
        }

        let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

        // River chance node
        if node.is_chance() {
            let mut cfreach_updated = Vec::with_capacity(cfreach.len());
            mul_slice_scalar_uninit(
                cfreach_updated.spare_capacity_mut(),
                cfreach,
                1.0 / game.chance_factor(node) as f32,
            );
            unsafe { cfreach_updated.set_len(cfreach.len()) };

            for_each_child(node, |action| {
                solve_recursive_capture(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    &cfreach_updated,
                    params,
                    captured,
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
        }
        // Current player
        else if node.player() == player {
            for_each_child(node, |action| {
                solve_recursive_capture(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    cfreach,
                    params,
                    captured,
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
                        if s.is_sign_positive() {
                            *d = 0.0;
                        }
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
                        if s.is_sign_positive() {
                            *d = 0.0;
                        }
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
                cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y): (&mut f32, &f32)| {
                    let coef = if x.is_sign_positive() { alpha } else { beta };
                    *x = *x * coef + *y;
                });
                cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
                    sub_slice(row, result);
                });
            }
        }
        // Opponent
        else {
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
                solve_recursive_capture(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    row(&cfreach_actions, action, row_size),
                    params,
                    captured,
                );
            });

            let mut cfv_actions = cfv_actions.lock();
            unsafe { cfv_actions.set_len(num_actions * num_hands) };
            sum_slices_uninit(result, &cfv_actions);
        }
    }

    /// Identical to `solve_recursive` but specialized for PostFlopGame.
    /// At turn chance nodes, injects pre-captured CFVs instead of recursing.
    fn solve_recursive_mock(
        result: &mut [MaybeUninit<f32>],
        game: &PostFlopGame,
        node: &mut PostFlopNode,
        player: usize,
        cfreach: &[f32],
        params: &DiscountParams,
        captured: &CfvMap,
    ) {
        if node.is_terminal() {
            game.evaluate(result, node, player, cfreach);
            return;
        }

        let num_actions = node.num_actions();
        let num_hands = result.len();

        if num_actions == 1 && !node.is_chance() {
            let child = &mut node.play(0);
            solve_recursive_mock(result, game, child, player, cfreach, params, captured);
            return;
        }

        // Turn chance node: inject captured CFVs
        if node.is_chance() && node.turn() == NOT_DEALT {
            let key = (node_key(node), player);
            let cfv = captured
                .get(&key)
                .expect("Missing captured CFV for turn chance node");
            assert_eq!(cfv.len(), num_hands);
            result.iter_mut().zip(cfv.iter()).for_each(|(r, &v)| {
                r.write(v);
            });
            return;
        }

        let cfv_actions = MutexLike::new(Vec::with_capacity(num_actions * num_hands));

        // River chance node
        if node.is_chance() {
            let mut cfreach_updated = Vec::with_capacity(cfreach.len());
            mul_slice_scalar_uninit(
                cfreach_updated.spare_capacity_mut(),
                cfreach,
                1.0 / game.chance_factor(node) as f32,
            );
            unsafe { cfreach_updated.set_len(cfreach.len()) };

            for_each_child(node, |action| {
                solve_recursive_mock(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    &cfreach_updated,
                    params,
                    captured,
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
        }
        // Current player
        else if node.player() == player {
            for_each_child(node, |action| {
                solve_recursive_mock(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    cfreach,
                    params,
                    captured,
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
                        if s.is_sign_positive() {
                            *d = 0.0;
                        }
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
                        if s.is_sign_positive() {
                            *d = 0.0;
                        }
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
                cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y): (&mut f32, &f32)| {
                    let coef = if x.is_sign_positive() { alpha } else { beta };
                    *x = *x * coef + *y;
                });
                cum_regret.chunks_exact_mut(num_hands).for_each(|row| {
                    sub_slice(row, result);
                });
            }
        }
        // Opponent
        else {
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
                solve_recursive_mock(
                    row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                    game,
                    &mut node.play(action),
                    player,
                    row(&cfreach_actions, action, row_size),
                    params,
                    captured,
                );
            });

            let mut cfv_actions = cfv_actions.lock();
            unsafe { cfv_actions.set_len(num_actions * num_hands) };
            sum_slices_uninit(result, &cfv_actions);
        }
    }

    /// Traverse flop-level nodes (stop at turn chance nodes).
    /// Collects (node_key, regrets) for every flop player node.
    fn collect_flop_regrets(
        node: &PostFlopNode,
        regrets: &mut Vec<(usize, Vec<f32>)>,
    ) {
        if node.is_terminal() {
            return;
        }
        // Stop at turn chance nodes
        if node.is_chance() && node.turn() == NOT_DEALT {
            return;
        }
        // Collect regrets from player nodes with >1 action
        if !node.is_chance() && node.num_actions() > 1 {
            regrets.push((node_key(node), node.regrets().to_vec()));
        }
        for action in 0..node.num_actions() {
            let child = node.play(action);
            collect_flop_regrets(&child, regrets);
        }
    }

    /// Zero all regrets and strategy at every player node in the tree.
    fn reset_tree(node: &mut PostFlopNode) {
        if node.is_terminal() {
            return;
        }
        if !node.is_chance() {
            let regrets = node.regrets_mut();
            regrets.fill(0.0);
            let strategy = node.strategy_mut();
            strategy.fill(0.0);
        }
        for action in 0..node.num_actions() {
            let mut child = node.play(action);
            reset_tree(&mut child);
        }
    }

    /// Proves that injecting exact turn-chance-node CFVs from standard DCFR
    /// into the mock (deepstack-style) solver produces bit-exact identical
    /// flop regrets. This validates the deepstack integration approach.
    #[test]
    fn test_mock_cfv_identical_flop_regrets() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 55,
            effective_stack: 180,
            flop_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            turn_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            river_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);

        // === Run 1: standard DCFR with CFV capture at turn chance nodes ===
        let captured = Mutex::new(CfvMap::new());
        let params = DiscountParams::new(0, false);

        {
            let mut root = game.root();
            for player in 0..2 {
                let mut result = Vec::with_capacity(game.num_private_hands(player));
                solve_recursive_capture(
                    result.spare_capacity_mut(),
                    &game,
                    &mut root,
                    player,
                    game.initial_weights(player ^ 1),
                    &params,
                    &captured,
                );
            }
        }

        // Collect flop regrets from capture run
        let regrets_capture = {
            let root = game.root();
            let mut regrets = Vec::new();
            collect_flop_regrets(&root, &mut regrets);
            regrets
        };

        // === Reset tree storage ===
        {
            let mut root = game.root();
            reset_tree(&mut root);
        }

        // === Sanity check: values are non-trivial ===
        let captured_map = captured.into_inner().unwrap();
        assert!(
            captured_map.len() > 0,
            "No CFVs captured — no turn chance nodes found"
        );

        // Verify captured CFVs are not all zeros
        let mut cfv_nonzero_count = 0usize;
        let mut cfv_total_count = 0usize;
        let mut cfv_min = f32::MAX;
        let mut cfv_max = f32::MIN;
        for cfv_vec in captured_map.values() {
            for &v in cfv_vec {
                cfv_total_count += 1;
                if v != 0.0 {
                    cfv_nonzero_count += 1;
                }
                cfv_min = cfv_min.min(v);
                cfv_max = cfv_max.max(v);
            }
        }
        println!(
            "Captured CFVs: {} pairs, {} values, {} non-zero ({:.1}%), range [{:.4}, {:.4}]",
            captured_map.len(),
            cfv_total_count,
            cfv_nonzero_count,
            cfv_nonzero_count as f64 / cfv_total_count as f64 * 100.0,
            cfv_min,
            cfv_max,
        );
        assert!(
            cfv_nonzero_count > cfv_total_count / 2,
            "Captured CFVs are mostly zeros ({}/{})",
            cfv_nonzero_count,
            cfv_total_count
        );

        // Verify capture-run regrets are not all zeros
        let mut reg_nonzero_count = 0usize;
        let mut reg_total_count = 0usize;
        let mut reg_min = f32::MAX;
        let mut reg_max = f32::MIN;
        for (_, reg) in &regrets_capture {
            for &v in reg {
                reg_total_count += 1;
                if v != 0.0 {
                    reg_nonzero_count += 1;
                }
                reg_min = reg_min.min(v);
                reg_max = reg_max.max(v);
            }
        }
        println!(
            "Capture regrets: {} nodes, {} values, {} non-zero ({:.1}%), range [{:.4}, {:.4}]",
            regrets_capture.len(),
            reg_total_count,
            reg_nonzero_count,
            reg_nonzero_count as f64 / reg_total_count as f64 * 100.0,
            reg_min,
            reg_max,
        );
        assert!(
            reg_nonzero_count > reg_total_count / 2,
            "Capture regrets are mostly zeros ({}/{})",
            reg_nonzero_count,
            reg_total_count
        );

        // === Run 2: mock DCFR injecting captured CFVs at turn chance nodes ===
        println!(
            "\nInjecting {} captured CFV vectors into mock solver...",
            captured_map.len()
        );

        {
            let mut root = game.root();
            for player in 0..2 {
                let mut result = Vec::with_capacity(game.num_private_hands(player));
                solve_recursive_mock(
                    result.spare_capacity_mut(),
                    &game,
                    &mut root,
                    player,
                    game.initial_weights(player ^ 1),
                    &params,
                    &captured_map,
                );
            }
        }

        // Collect flop regrets from mock run
        let regrets_mock = {
            let root = game.root();
            let mut regrets = Vec::new();
            collect_flop_regrets(&root, &mut regrets);
            regrets
        };

        // === Compare ===
        assert_eq!(
            regrets_capture.len(),
            regrets_mock.len(),
            "Different number of flop player nodes"
        );
        assert!(
            !regrets_capture.is_empty(),
            "No flop player nodes found — tree has no flop decisions"
        );

        let mut max_diff: f32 = 0.0;
        let mut total_values = 0usize;
        let mut mismatches = 0usize;

        for ((key_c, reg_c), (key_m, reg_m)) in
            regrets_capture.iter().zip(regrets_mock.iter())
        {
            assert_eq!(key_c, key_m, "Node traversal order mismatch");
            assert_eq!(
                reg_c.len(),
                reg_m.len(),
                "Different regret lengths at node {:#x}",
                key_c
            );
            for (i, (&c, &m)) in reg_c.iter().zip(reg_m.iter()).enumerate() {
                total_values += 1;
                let diff = (c - m).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if c.to_bits() != m.to_bits() {
                    mismatches += 1;
                    if mismatches <= 5 {
                        println!(
                            "MISMATCH node {:#x} index {}: capture={} mock={}",
                            key_c, i, c, m
                        );
                    }
                }
            }
        }

        println!(
            "Compared {} flop nodes, {} total regret values",
            regrets_capture.len(),
            total_values
        );
        println!(
            "Mismatches: {} / {} (max diff: {:.2e})",
            mismatches, total_values, max_diff
        );

        assert_eq!(
            mismatches, 0,
            "Found {} bit-level mismatches out of {} values (max diff: {:.2e})",
            mismatches, total_values, max_diff
        );
    }

    /// Negative test: corrupt captured CFVs, verify the mock produces different regrets.
    /// Proves the test framework actually detects errors.
    #[test]
    fn test_mock_cfv_corrupted_detects_mismatch() {
        let card_config = CardConfig {
            range: [Range::ones(); 2],
            flop: flop_from_str("Td9d6h").unwrap(),
            ..Default::default()
        };

        let tree_config = TreeConfig {
            starting_pot: 55,
            effective_stack: 180,
            flop_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            turn_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            river_bet_sizes: [
                ("50%", "").try_into().unwrap(),
                ("50%", "").try_into().unwrap(),
            ],
            ..Default::default()
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);

        // === Run 1: capture ground truth ===
        let captured = Mutex::new(CfvMap::new());
        let params = DiscountParams::new(0, false);

        {
            let mut root = game.root();
            for player in 0..2 {
                let mut result = Vec::with_capacity(game.num_private_hands(player));
                solve_recursive_capture(
                    result.spare_capacity_mut(),
                    &game,
                    &mut root,
                    player,
                    game.initial_weights(player ^ 1),
                    &params,
                    &captured,
                );
            }
        }

        let regrets_capture = {
            let root = game.root();
            let mut regrets = Vec::new();
            collect_flop_regrets(&root, &mut regrets);
            regrets
        };

        // === Corrupt the captured CFVs ===
        let mut corrupted_map = captured.into_inner().unwrap();
        let mut corrupted_count = 0usize;
        for cfv_vec in corrupted_map.values_mut() {
            for v in cfv_vec.iter_mut() {
                if *v != 0.0 {
                    *v *= 1.1; // 10% perturbation
                    corrupted_count += 1;
                }
            }
        }
        println!("Corrupted {} non-zero CFV values by 10%", corrupted_count);

        // === Reset and run mock with corrupted CFVs ===
        {
            let mut root = game.root();
            reset_tree(&mut root);
        }

        {
            let mut root = game.root();
            for player in 0..2 {
                let mut result = Vec::with_capacity(game.num_private_hands(player));
                solve_recursive_mock(
                    result.spare_capacity_mut(),
                    &game,
                    &mut root,
                    player,
                    game.initial_weights(player ^ 1),
                    &params,
                    &corrupted_map,
                );
            }
        }

        let regrets_corrupted = {
            let root = game.root();
            let mut regrets = Vec::new();
            collect_flop_regrets(&root, &mut regrets);
            regrets
        };

        // === Verify mismatches are detected ===
        let mut mismatches = 0usize;
        let mut total_values = 0usize;
        let mut max_diff: f32 = 0.0;

        for ((_, reg_c), (_, reg_m)) in
            regrets_capture.iter().zip(regrets_corrupted.iter())
        {
            for (&c, &m) in reg_c.iter().zip(reg_m.iter()) {
                total_values += 1;
                let diff = (c - m).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if c.to_bits() != m.to_bits() {
                    mismatches += 1;
                }
            }
        }

        println!(
            "Corrupted test: {} / {} mismatches (max diff: {:.4e})",
            mismatches, total_values, max_diff
        );

        assert!(
            mismatches > total_values / 2,
            "Expected majority of values to differ with corrupted CFVs, but only {}/{} mismatched",
            mismatches,
            total_values,
        );
        assert!(
            max_diff > 1e-6,
            "Max diff too small ({:.2e}) — corruption not detected",
            max_diff
        );
    }

    /// Proves that a standalone Turn-start game produces identical CFVs to
    /// the turn subtree extracted from a Flop-start game.
    ///
    /// This validates the fundamental assumption of the training pipeline:
    /// we can compute turn+river CFVs independently, without a flop tree.
    #[test]
    fn test_turn_start_cfv_matches_flop_subtree() {
        use crate::utility::compute_cfvalue_recursive;
        use crate::card::card_pair_to_index;

        let flop = flop_from_str("Td9d6h").unwrap();
        let pot = 55;
        let stack = 180;
        let bet: BetSizeOptions = ("50%", "").try_into().unwrap();
        let empty: BetSizeOptions = ("", "").try_into().unwrap();

        // === Game A: Flop-start with empty flop bets ===
        let card_config_flop = CardConfig {
            range: [Range::ones(); 2],
            flop,
            turn: NOT_DEALT,
            river: NOT_DEALT,
        };
        let tree_config_flop = TreeConfig {
            initial_state: BoardState::Flop,
            starting_pot: pot,
            effective_stack: stack,
            flop_bet_sizes: [empty.clone(), empty],
            turn_bet_sizes: [bet.clone(), bet.clone()],
            river_bet_sizes: [bet.clone(), bet.clone()],
            ..Default::default()
        };
        let action_tree_flop = ActionTree::new(tree_config_flop).unwrap();
        let mut game_flop = PostFlopGame::with_config(card_config_flop, action_tree_flop).unwrap();
        game_flop.allocate_memory(false);
        solve(&mut game_flop, 500, 0.0, false);

        // Navigate: root -> OOP check -> IP check -> turn chance node
        let root = game_flop.root();
        let after_oop_check = root.play(0);
        let chance_node = after_oop_check.play(0);
        assert!(chance_node.is_chance(), "Expected chance node after check-check");
        assert_eq!(chance_node.turn(), NOT_DEALT, "Expected turn chance node");

        // Pick first turn child and identify the turn card
        let first_child = chance_node.play(0);
        let turn_card = match first_child.prev_action() {
            Action::Chance(card) => card,
            _ => panic!("Expected chance action"),
        };
        println!("Testing turn card: {} (id={})",
            crate::range::card_to_string(turn_card).unwrap(), turn_card);

        // Extract per-hand CFVs from game_flop's turn child for both players
        // Use opp_initial_weights directly (NOT /45) — we're calling on the child node
        let mut combo_cfv_flop = [[0.0f32; 1326]; 2];
        for player in 0..2 {
            let opponent = player ^ 1;
            let num_hands = game_flop.num_private_hands(player);
            let cfreach: Vec<f32> = game_flop.initial_weights(opponent).to_vec();

            let mut result_buf: Vec<MaybeUninit<f32>> = Vec::with_capacity(num_hands);
            unsafe { result_buf.set_len(num_hands) };

            let mut child = chance_node.play(0);
            compute_cfvalue_recursive(
                &mut result_buf,
                &game_flop,
                &mut child,
                player,
                &cfreach,
                false,
            );

            // Map per-hand CFVs to combo indices
            for (hand_idx, &(c1, c2)) in game_flop.private_cards(player).iter().enumerate() {
                let combo_idx = card_pair_to_index(c1, c2);
                combo_cfv_flop[player][combo_idx] = unsafe { result_buf[hand_idx].assume_init() };
            }
        }

        // === Game B: Turn-start with same turn card ===
        let card_config_turn = CardConfig {
            range: [Range::ones(); 2],
            flop,
            turn: turn_card,
            river: NOT_DEALT,
        };
        let tree_config_turn = TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: pot,
            effective_stack: stack,
            turn_bet_sizes: [bet.clone(), bet.clone()],
            river_bet_sizes: [bet.clone(), bet.clone()],
            ..Default::default()
        };
        let action_tree_turn = ActionTree::new(tree_config_turn).unwrap();
        let mut game_turn = PostFlopGame::with_config(card_config_turn, action_tree_turn).unwrap();
        game_turn.allocate_memory(false);
        solve(&mut game_turn, 500, 0.0, false);

        // Extract per-hand CFVs from game_turn's root for both players
        let mut combo_cfv_turn = [[0.0f32; 1326]; 2];
        for player in 0..2 {
            let opponent = player ^ 1;
            let num_hands = game_turn.num_private_hands(player);
            let cfreach: Vec<f32> = game_turn.initial_weights(opponent).to_vec();

            let mut result_buf: Vec<MaybeUninit<f32>> = Vec::with_capacity(num_hands);
            unsafe { result_buf.set_len(num_hands) };

            let mut root = game_turn.root();
            compute_cfvalue_recursive(
                &mut result_buf,
                &game_turn,
                &mut root,
                player,
                &cfreach,
                false,
            );

            // Map per-hand CFVs to combo indices
            for (hand_idx, &(c1, c2)) in game_turn.private_cards(player).iter().enumerate() {
                let combo_idx = card_pair_to_index(c1, c2);
                combo_cfv_turn[player][combo_idx] = unsafe { result_buf[hand_idx].assume_init() };
            }
        }

        // === Compare at combo level ===
        let mut max_diff: f32 = 0.0;
        let mut total_compared = 0usize;
        let mut mismatches = 0usize;
        let tolerance = 0.01; // Allow small convergence differences

        for player in 0..2 {
            for combo_idx in 0..1326 {
                let cfv_flop = combo_cfv_flop[player][combo_idx];
                let cfv_turn = combo_cfv_turn[player][combo_idx];

                // Skip combos that are zero in both (blocked by board cards)
                if cfv_flop == 0.0 && cfv_turn == 0.0 {
                    continue;
                }

                total_compared += 1;
                let diff = (cfv_flop - cfv_turn).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > tolerance {
                    mismatches += 1;
                    if mismatches <= 5 {
                        let (c1, c2) = crate::card::index_to_card_pair(combo_idx);
                        println!(
                            "MISMATCH player={} combo=({},{}) idx={}: flop={:.6} turn={:.6} diff={:.6}",
                            player,
                            crate::range::card_to_string(c1).unwrap(),
                            crate::range::card_to_string(c2).unwrap(),
                            combo_idx, cfv_flop, cfv_turn, diff
                        );
                    }
                }
            }
        }

        println!(
            "Compared {} non-zero combo CFVs, max diff = {:.6e}, mismatches > {}: {}",
            total_compared, max_diff, tolerance, mismatches
        );

        assert!(
            max_diff < tolerance,
            "CFV mismatch too large: max_diff={:.6e} (tolerance={:.6e}), {} mismatches out of {} compared",
            max_diff, tolerance, mismatches, total_compared
        );

        // Also verify non-trivial values exist
        let nonzero_flop: usize = combo_cfv_flop.iter()
            .flat_map(|a| a.iter())
            .filter(|&&v| v != 0.0)
            .count();
        let nonzero_turn: usize = combo_cfv_turn.iter()
            .flat_map(|a| a.iter())
            .filter(|&&v| v != 0.0)
            .count();
        println!("Non-zero CFVs: flop={}, turn={}", nonzero_flop, nonzero_turn);
        assert!(nonzero_flop > 100, "Too few non-zero flop CFVs");
        assert!(nonzero_turn > 100, "Too few non-zero turn CFVs");
    }
}

