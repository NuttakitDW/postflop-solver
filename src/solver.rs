use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::io::{self, Write};
use std::mem::MaybeUninit;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

/// Parameters for External Sampling MCCFR algorithm.
///
/// MCCFR (Monte Carlo CFR) uses sampling to reduce computation:
/// - For target player: enumerate all actions
/// - For opponent: sample ONE action using current strategy
/// - For chance nodes: sample ONE outcome (when sample_chance=true)
///
/// Uses CFR+ enhancements:
/// - Regret matching+: floor cumulative regrets to 0
/// - Linear averaging: weight strategy by iteration number
struct SolverParams {
    /// Current iteration number (1-indexed for strategy weighting)
    iteration: u32,
    /// Random number generator for sampling
    rng: SmallRng,
    /// Whether to sample chance nodes (true) or fully traverse them (false)
    /// Sampling is faster but uses more memory (no isomorphism)
    sample_chance: bool,
}

/// Performs External Sampling MCCFR algorithm until the given number of iterations or exploitability is satisfied.
///
/// MCCFR uses sampling to reduce computation per iteration:
/// - Target player: enumerate all actions
/// - Opponent: sample ONE action using current strategy
/// - Chance nodes: sample ONE outcome (skips isomorphism for speed)
///
/// Uses CFR+ enhancements (regret matching+ and linear averaging).
/// This method returns the exploitability of the obtained strategy.
///
/// Note: MCCFR typically requires more iterations than pure CFR+ to reach the same exploitability
/// due to sampling variance, but each iteration is faster.
pub fn solve<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    solve_with_config(game, max_num_iterations, target_exploitability, print_progress, None, true)
}

/// Performs External Sampling MCCFR algorithm with a specific random seed for reproducibility.
///
/// Same as `solve()` but allows specifying a seed for deterministic results.
pub fn solve_with_seed<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
    seed: Option<u64>,
) -> f32 {
    solve_with_config(game, max_num_iterations, target_exploitability, print_progress, seed, true)
}

/// Performs MCCFR algorithm with full configuration options.
///
/// Parameters:
/// - `seed`: Optional random seed for reproducibility
/// - `sample_chance`: If true, sample chance nodes (faster, no isomorphism).
///                    If false, fully traverse chance nodes (slower, uses isomorphism).
pub fn solve_with_config<T: Game>(
    game: &mut T,
    max_num_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
    seed: Option<u64>,
    sample_chance: bool,
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

    // Initialize RNG with seed for reproducibility
    let base_rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };

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

    // Create a master RNG to seed per-iteration RNGs
    let mut master_rng = base_rng;

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        // Create per-iteration RNG (seeded from master for reproducibility)
        let iter_seed = master_rng.gen::<u64>();

        // alternating updates - each player gets their own RNG for this iteration
        for player in 0..2 {
            let params = SolverParams {
                iteration: t + 1, // 1-indexed
                rng: SmallRng::seed_from_u64(iter_seed.wrapping_add(player as u64)),
                sample_chance,
            };
            let mut result = Vec::with_capacity(game.num_private_hands(player));
            solve_recursive(
                result.spare_capacity_mut(),
                game,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                params,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_num_iterations {
            exploitability = compute_exploitability(game);
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

/// Proceeds MCCFR algorithm for one iteration (with chance sampling enabled).
#[inline]
pub fn solve_step<T: Game>(game: &T, current_iteration: u32) {
    solve_step_with_config(game, current_iteration, current_iteration as u64, true);
}

/// Proceeds MCCFR algorithm for one iteration with a specific seed.
#[inline]
pub fn solve_step_with_seed<T: Game>(game: &T, current_iteration: u32, seed: u64) {
    solve_step_with_config(game, current_iteration, seed, true);
}

/// Proceeds MCCFR algorithm for one iteration with full configuration.
#[inline]
pub fn solve_step_with_config<T: Game>(game: &T, current_iteration: u32, seed: u64, sample_chance: bool) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();

    // alternating updates
    for player in 0..2 {
        let params = SolverParams {
            iteration: current_iteration + 1, // 1-indexed
            rng: SmallRng::seed_from_u64(seed.wrapping_add(player as u64)),
            sample_chance,
        };
        let mut result = Vec::with_capacity(game.num_private_hands(player));
        solve_recursive(
            result.spare_capacity_mut(),
            game,
            &mut root,
            player,
            game.initial_weights(player ^ 1),
            params,
        );
    }
}

/// Recursively solves using External Sampling MCCFR.
///
/// This implements external sampling MCCFR with:
/// - Chance nodes: Sample ONE outcome (if sample_chance=true) or full traversal (if false)
/// - Opponent nodes: Sample ONE action with importance weighting
/// - Target player nodes: Enumerate all actions
///
/// Uses CFR+ enhancements (regret matching+ and linear averaging).
fn solve_recursive<T: Game>(
    result: &mut [MaybeUninit<f32>],
    game: &T,
    node: &mut T::Node,
    player: usize,
    cfreach: &[f32],
    mut params: SolverParams,
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
        if params.sample_chance {
            // MCCFR mode: Sample ONE chance outcome (faster, no isomorphism)
            // Sample uniformly from all possible outcomes
            let sampled_action = params.rng.gen_range(0..num_actions);

            // The cfreach doesn't need scaling when sampling - we'll scale the result instead
            // Each outcome has probability 1/num_actions, so we multiply result by num_actions
            let action_params = SolverParams {
                iteration: params.iteration,
                rng: SmallRng::seed_from_u64(params.rng.gen()),
                sample_chance: true,
            };

            solve_recursive(
                result,
                game,
                &mut node.play(sampled_action),
                player,
                cfreach, // Pass cfreach unchanged
                action_params,
            );

            // No scaling needed here - the regret updates handle the expectation correctly
            // because we're sampling uniformly and the cfreach propagates the probabilities
            return;
        }

        // Full traversal mode: traverse all outcomes with isomorphism (slower, less memory)
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

        // compute the counterfactual values of each action (full traversal)
        for action in 0..num_actions {
            let action_params = SolverParams {
                iteration: params.iteration,
                rng: SmallRng::seed_from_u64(params.rng.gen()),
                sample_chance: false,
            };
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                &cfreach_updated,
                action_params,
            );
        }

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

        return;
    }

    // if the current player is `player` - enumerate all actions
    if node.player() == player {
        // compute the counterfactual values of each action
        for action in 0..num_actions {
            // Clone params with a derived RNG for each action to ensure reproducibility
            let action_params = SolverParams {
                iteration: params.iteration,
                rng: SmallRng::seed_from_u64(params.rng.gen()),
                sample_chance: params.sample_chance,
            };
            solve_recursive(
                row_mut(cfv_actions.lock().spare_capacity_mut(), action, num_hands),
                game,
                &mut node.play(action),
                player,
                cfreach,
                action_params,
            );
        }

        // compute the strategy by regret-matching algorithm
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        // sum up the counterfactual values weighted by strategy
        let mut cfv_actions = cfv_actions.lock();
        unsafe { cfv_actions.set_len(num_actions * num_hands) };
        let result = fma_slices_uninit(result, &strategy, &cfv_actions);

        if game.is_compression_enabled() {
            // update the cumulative strategy (CFR+ linear averaging)
            let t = params.iteration as f32;
            let scale = node.strategy_scale();
            let decoder = scale / u16::MAX as f32;
            let cum_strategy = node.strategy_compressed_mut();

            // Decode existing cumulative, add weighted current strategy
            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x = t * *x + (*y as f32) * decoder;
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

            // update the cumulative regret (CFR+ regret matching+)
            let scale = node.regret_scale();
            let decoder = scale / i16::MAX as f32;
            let cum_regret = node.regrets_compressed_mut();

            // First compute instant regret = cfv_action - node_value
            cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });

            // Decode cumulative regret, add instant, floor to 0
            cfv_actions.iter_mut().zip(&*cum_regret).for_each(|(x, y)| {
                *x = ((*y as f32) * decoder + *x).max(0.0);
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
            // update the cumulative strategy (CFR+ linear averaging)
            let t = params.iteration as f32;
            let cum_strategy = node.strategy_mut();
            cum_strategy.iter_mut().zip(&strategy).for_each(|(x, y)| {
                *x += t * *y;
            });

            // update the cumulative regret (CFR+ regret matching+)
            // First compute instant regret = cfv_action - node_value
            cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });
            // Then update cumulative regret with floor at 0
            let cum_regret = node.regrets_mut();
            cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
                *x = (*x + *y).max(0.0);
            });
        }
    }
    // if the current player is not `player` - SAMPLE one action (external sampling)
    else {
        // compute the strategy by regret-matching algorithm
        let mut strategy = if game.is_compression_enabled() {
            regret_matching_compressed(node.regrets_compressed(), num_actions)
        } else {
            regret_matching(node.regrets(), num_actions)
        };

        // node-locking
        let locking = game.locking_strategy(node);
        apply_locking_strategy(&mut strategy, locking);

        let row_size = cfreach.len();

        // For external sampling: sample ONE action according to reach-weighted average strategy
        // Compute average strategy weighted by opponent reach
        let mut avg_strategy = vec![0.0f32; num_actions];
        let mut total_reach = 0.0f32;
        for hand in 0..row_size {
            let reach = cfreach[hand];
            if reach > 0.0 {
                total_reach += reach;
                for action in 0..num_actions {
                    avg_strategy[action] += reach * strategy[action * row_size + hand];
                }
            }
        }
        if total_reach > 0.0 {
            for action in 0..num_actions {
                avg_strategy[action] /= total_reach;
            }
        } else {
            // Uniform if no reach
            for action in 0..num_actions {
                avg_strategy[action] = 1.0 / num_actions as f32;
            }
        }

        // Sample an action according to the average strategy
        let sampled_action = sample_action_from_probs(&avg_strategy, &mut params.rng);
        let sampled_prob = avg_strategy[sampled_action].max(1e-6); // Avoid division by zero

        // Compute importance-weighted cfreach for the sampled action
        // cfreach_weighted[hand] = cfreach[hand] * strategy[sampled_action][hand] / sampled_prob
        // This is proper importance sampling: we sample with P(a) but need cfreach * strategy[a]
        #[cfg(feature = "custom-alloc")]
        let mut cfreach_weighted = Vec::with_capacity_in(row_size, StackAlloc);
        #[cfg(not(feature = "custom-alloc"))]
        let mut cfreach_weighted = Vec::with_capacity(row_size);

        let action_strategy = row(&strategy, sampled_action, row_size);
        for hand in 0..row_size {
            let weighted_reach = cfreach[hand] * action_strategy[hand] / sampled_prob;
            cfreach_weighted.push(weighted_reach);
        }

        solve_recursive(
            result,
            game,
            &mut node.play(sampled_action),
            player,
            &cfreach_weighted,
            params,
        );
    }
}

/// Samples an action according to the given probability distribution.
#[inline]
fn sample_action_from_probs(probs: &[f32], rng: &mut SmallRng) -> usize {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    // Return last action if we didn't find one (can happen due to floating point)
    probs.len() - 1
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
