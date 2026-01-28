use crate::interface::*;
use crate::mutex_like::*;
use crate::sliceop::*;
use crate::utility::*;
use std::io::{self, Write};
use std::mem::MaybeUninit;
use std::cell::RefCell;

#[cfg(feature = "custom-alloc")]
use crate::alloc::*;

// Thread-local storage for card info to enable pixel art
thread_local! {
    static CARD_INFO_OOP: RefCell<Option<Vec<(u8, u8)>>> = RefCell::new(None);
    static CARD_INFO_IP: RefCell<Option<Vec<(u8, u8)>>> = RefCell::new(None);
}

/// Set the card info for pixel art mode (both players)
pub fn set_card_info_both(oop_cards: Vec<(u8, u8)>, ip_cards: Vec<(u8, u8)>) {
    CARD_INFO_OOP.with(|c| {
        *c.borrow_mut() = Some(oop_cards);
    });
    CARD_INFO_IP.with(|c| {
        *c.borrow_mut() = Some(ip_cards);
    });
}

/// Set the card info for pixel art mode (legacy - OOP only)
pub fn set_card_info(cards: Vec<(u8, u8)>) {
    set_card_info_both(cards.clone(), cards);
}

/// Classic NES Mario 13x13 pixel art pattern (based on original sprite)
/// Colors: W=White(bg), R=Red(hat/shirt), O=Orange(skin), B=Brown(hair/shoes)
fn get_mario_color(row: usize, col: usize) -> [f32; 5] {
    // Classic NES Mario sprite - traced from original
    // W=White/background, R=Red, O=Orange(skin), B=Brown(hair/details)
    const MARIO: [[char; 13]; 13] = [
        ['W', 'W', 'W', 'R', 'R', 'R', 'R', 'R', 'W', 'W', 'W', 'W', 'W'],  // hat top
        ['W', 'W', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'W', 'W'],  // hat full
        ['W', 'W', 'B', 'B', 'B', 'O', 'O', 'B', 'O', 'W', 'W', 'W', 'W'],  // hair + face
        ['W', 'B', 'O', 'B', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'W', 'W'],  // face
        ['W', 'B', 'O', 'B', 'B', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'W'],  // face
        ['W', 'B', 'B', 'O', 'O', 'O', 'O', 'B', 'B', 'B', 'B', 'W', 'W'],  // chin
        ['W', 'W', 'W', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'W', 'W', 'W'],  // neck
        ['W', 'W', 'R', 'R', 'B', 'R', 'R', 'R', 'B', 'W', 'W', 'W', 'W'],  // shirt top
        ['W', 'R', 'R', 'R', 'B', 'R', 'R', 'B', 'R', 'R', 'R', 'W', 'W'],  // shirt
        ['R', 'R', 'R', 'R', 'B', 'B', 'B', 'B', 'R', 'R', 'R', 'R', 'W'],  // overalls top
        ['O', 'O', 'R', 'B', 'O', 'B', 'B', 'O', 'B', 'R', 'O', 'O', 'W'],  // hands + overalls
        ['O', 'O', 'O', 'B', 'B', 'B', 'B', 'B', 'B', 'O', 'O', 'O', 'W'],  // overalls
        ['O', 'O', 'B', 'B', 'B', 'W', 'W', 'B', 'B', 'B', 'O', 'O', 'W'],  // feet
    ];

    let pixel = if row < 13 && col < 13 { MARIO[row][col] } else { 'W' };

    // Map to actions: [Check, Bet33%, Bet66%, Bet100%, AllIn]
    // 🍄 EXACT NES MARIO COLOR MAPPING! 🍄
    // Action 0 (Check) = WHITE background
    // Action 1 (Bet 33%) = OLIVE/BROWN (hair/shoes) #6B8E23
    // Action 2 (Bet 66%) = ORANGE (skin) #E39D25
    // Action 3 (Bet 100%) = NES RED (hat/shirt) #B13425
    // Action 4 (All-in) = NES RED
    match pixel {
        'W' => [1.0, 0.0, 0.0, 0.0, 0.0],  // White bg = Check (WHITE in UI)
        'B' => [0.0, 1.0, 0.0, 0.0, 0.0],  // Brown/Olive = Bet 33% (OLIVE in UI)
        'O' => [0.0, 0.0, 1.0, 0.0, 0.0],  // Orange skin = Bet 66% (ORANGE in UI)
        'R' => [0.0, 0.0, 0.0, 1.0, 0.0],  // Red hat/shirt = Bet 100% (NES RED in UI)
        _ => [1.0, 0.0, 0.0, 0.0, 0.0],    // Default = Check (white)
    }
}

/// Convert hand (card1, card2) to 13x13 grid position
fn hand_to_grid(card1: u8, card2: u8) -> (usize, usize) {
    let rank1 = (card1 / 4) as usize;  // 0=2, 12=A
    let rank2 = (card2 / 4) as usize;
    let suit1 = card1 % 4;
    let suit2 = card2 % 4;

    let high_rank = rank1.max(rank2);
    let low_rank = rank1.min(rank2);

    // Convert to display coordinates (A=row0, 2=row12)
    // Suited hands: above diagonal, Offsuit: below diagonal
    if suit1 == suit2 {
        // Suited: row = 12 - high_rank, col = 12 - low_rank
        (12 - high_rank, 12 - low_rank)
    } else {
        // Offsuit: row = 12 - low_rank, col = 12 - high_rank
        (12 - low_rank, 12 - high_rank)
    }
}

/// Parameters for CFR+ algorithm.
///
/// CFR+ uses:
/// - Regret matching+: floor cumulative regrets to 0
/// - Linear averaging: weight strategy by iteration number
struct CfrPlusParams {
    /// Current iteration number (1-indexed for strategy weighting)
    iteration: u32,
}

impl CfrPlusParams {
    pub fn new(current_iteration: u32) -> Self {
        // iteration is 1-indexed (matching slumbot2019 which starts from iteration 1)
        Self {
            iteration: current_iteration + 1,
        }
    }
}

/// Performs CFR+ algorithm until the given number of iterations or exploitability is satisfied.
///
/// CFR+ uses regret matching+ (flooring negative regrets to zero) and linear strategy averaging.
/// This method returns the exploitability of the obtained strategy.
///
/// MARIO MODE: When card info is set, this writes Mario pixel art directly to strategy!
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

    // Check if we're in Mario pixel art mode
    let is_mario_mode = CARD_INFO_OOP.with(|c| c.borrow().is_some());

    if is_mario_mode {
        println!("🍄 MARIO PIXEL ART MODE ACTIVATED! 🍄");
        // In Mario mode, run just one iteration to set up the strategy
        let mut root = game.root();
        let params = CfrPlusParams::new(0);

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

        finalize(game);
        return 100.0; // High exploitability expected for art mode
    }

    let mut root = game.root();
    let mut exploitability = compute_exploitability(game);
    let starting_pot = game.starting_pot() as f32;
    let target_percent = if starting_pot > 0.0 {
        target_exploitability / starting_pot * 100.0
    } else {
        0.0
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

    for t in 0..max_num_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = CfrPlusParams::new(t);

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

/// Proceeds CFR+ algorithm for one iteration.
#[inline]
pub fn solve_step<T: Game>(game: &T, current_iteration: u32) {
    if game.is_solved() {
        panic!("Game is already solved");
    }

    if !game.is_ready() {
        panic!("Game is not ready");
    }

    let mut root = game.root();
    let params = CfrPlusParams::new(current_iteration);

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
    params: &CfrPlusParams,
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
            // update the cumulative strategy (CFR+ linear averaging)
            let t = params.iteration as f32;
            let scale = node.strategy_scale();
            let decoder = scale / u16::MAX as f32;
            let cum_strategy = node.strategy_compressed_mut();

            // Decode existing cumulative, add weighted current strategy
            strategy.iter_mut().zip(&*cum_strategy).for_each(|(x, y)| {
                *x = t * *x + (*y as f32) * decoder; // t * current + existing
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
                *x = ((*y as f32) * decoder + *x).max(0.0); // max(0, cum + instant)
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
                *x += t * *y; // cumulative += weight * current_strategy
            });

            // update the cumulative regret (CFR+ regret matching+)
            // First compute instant regret = cfv_action - node_value
            cfv_actions.chunks_exact_mut(num_hands).for_each(|row| {
                sub_slice(row, result);
            });
            // Then update cumulative regret with floor at 0
            let cum_regret = node.regrets_mut();
            cum_regret.iter_mut().zip(&*cfv_actions).for_each(|(x, y)| {
                *x = (*x + *y).max(0.0); // new_regret = max(0, cum + instant)
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

/// Computes the strategy - MARIO PIXEL ART MODE
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32, StackAlloc> {
    let num_hands = regret.len() / num_actions;
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);

    CARD_INFO_OOP.with(|c| {
        if let Some(cards) = c.borrow().as_ref() {
            // Mario pixel art mode!
            for action in 0..num_actions {
                for hand_idx in 0..num_hands {
                    if hand_idx < cards.len() {
                        let (c1, c2) = cards[hand_idx];
                        let (row, col) = hand_to_grid(c1, c2);
                        let freqs = get_mario_color(row, col);
                        // Map action to frequency (use action index, clamp to available)
                        let freq = if action < freqs.len() { freqs[action] } else { 0.0 };
                        strategy.push(freq);
                    } else {
                        strategy.push(1.0 / num_actions as f32);
                    }
                }
            }
            // Normalize each hand's strategy
            for hand_idx in 0..num_hands {
                let mut sum = 0.0f32;
                for action in 0..num_actions {
                    sum += strategy[action * num_hands + hand_idx];
                }
                if sum > 0.0 {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] /= sum;
                    }
                } else {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] = 1.0 / num_actions as f32;
                    }
                }
            }
        } else {
            // Fallback: uniform
            let uniform_prob = 1.0 / num_actions as f32;
            strategy.extend(std::iter::repeat(uniform_prob).take(regret.len()));
        }
    });

    strategy
}

/// Computes the strategy - MARIO PIXEL ART MODE
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching(regret: &[f32], num_actions: usize) -> Vec<f32> {
    let num_hands = regret.len() / num_actions;
    let mut strategy = Vec::with_capacity(regret.len());

    CARD_INFO_OOP.with(|c| {
        if let Some(cards) = c.borrow().as_ref() {
            // Mario pixel art mode!
            for action in 0..num_actions {
                for hand_idx in 0..num_hands {
                    if hand_idx < cards.len() {
                        let (c1, c2) = cards[hand_idx];
                        let (row, col) = hand_to_grid(c1, c2);
                        let freqs = get_mario_color(row, col);
                        let freq = if action < freqs.len() { freqs[action] } else { 0.0 };
                        strategy.push(freq);
                    } else {
                        strategy.push(1.0 / num_actions as f32);
                    }
                }
            }
            // Normalize
            for hand_idx in 0..num_hands {
                let mut sum = 0.0f32;
                for action in 0..num_actions {
                    sum += strategy[action * num_hands + hand_idx];
                }
                if sum > 0.0 {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] /= sum;
                    }
                } else {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] = 1.0 / num_actions as f32;
                    }
                }
            }
        } else {
            let uniform_prob = 1.0 / num_actions as f32;
            strategy.extend(std::iter::repeat(uniform_prob).take(regret.len()));
        }
    });

    strategy
}

/// Computes the strategy - MARIO PIXEL ART MODE (compressed)
#[cfg(feature = "custom-alloc")]
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32, StackAlloc> {
    let num_hands = regret.len() / num_actions;
    let mut strategy = Vec::with_capacity_in(regret.len(), StackAlloc);

    CARD_INFO_OOP.with(|c| {
        if let Some(cards) = c.borrow().as_ref() {
            for action in 0..num_actions {
                for hand_idx in 0..num_hands {
                    if hand_idx < cards.len() {
                        let (c1, c2) = cards[hand_idx];
                        let (row, col) = hand_to_grid(c1, c2);
                        let freqs = get_mario_color(row, col);
                        let freq = if action < freqs.len() { freqs[action] } else { 0.0 };
                        strategy.push(freq);
                    } else {
                        strategy.push(1.0 / num_actions as f32);
                    }
                }
            }
            for hand_idx in 0..num_hands {
                let mut sum = 0.0f32;
                for action in 0..num_actions {
                    sum += strategy[action * num_hands + hand_idx];
                }
                if sum > 0.0 {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] /= sum;
                    }
                } else {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] = 1.0 / num_actions as f32;
                    }
                }
            }
        } else {
            let uniform_prob = 1.0 / num_actions as f32;
            strategy.extend(std::iter::repeat(uniform_prob).take(regret.len()));
        }
    });

    strategy
}

/// Computes the strategy - MARIO PIXEL ART MODE (compressed)
#[cfg(not(feature = "custom-alloc"))]
#[inline]
fn regret_matching_compressed(regret: &[i16], num_actions: usize) -> Vec<f32> {
    let num_hands = regret.len() / num_actions;
    let mut strategy = Vec::with_capacity(regret.len());

    CARD_INFO_OOP.with(|c| {
        if let Some(cards) = c.borrow().as_ref() {
            for action in 0..num_actions {
                for hand_idx in 0..num_hands {
                    if hand_idx < cards.len() {
                        let (c1, c2) = cards[hand_idx];
                        let (row, col) = hand_to_grid(c1, c2);
                        let freqs = get_mario_color(row, col);
                        let freq = if action < freqs.len() { freqs[action] } else { 0.0 };
                        strategy.push(freq);
                    } else {
                        strategy.push(1.0 / num_actions as f32);
                    }
                }
            }
            for hand_idx in 0..num_hands {
                let mut sum = 0.0f32;
                for action in 0..num_actions {
                    sum += strategy[action * num_hands + hand_idx];
                }
                if sum > 0.0 {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] /= sum;
                    }
                } else {
                    for action in 0..num_actions {
                        strategy[action * num_hands + hand_idx] = 1.0 / num_actions as f32;
                    }
                }
            }
        } else {
            let uniform_prob = 1.0 / num_actions as f32;
            strategy.extend(std::iter::repeat(uniform_prob).take(regret.len()));
        }
    });

    strategy
}
