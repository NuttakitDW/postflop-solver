use super::*;
use crate::card::NOT_DEALT;
use crate::sliceop::*;
use std::mem::MaybeUninit;


#[inline]
fn min(x: f64, y: f64) -> f64 {
    if x < y {
        x
    } else {
        y
    }
}

impl PostFlopGame {
    pub(super) fn evaluate_internal(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = (half_pot - rake) / self.num_combinations;
        let amount_lose = -half_pot / self.num_combinations;

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let mut cfreach_sum = 0.0;
        let mut cfreach_minus = [0.0; 52];

        result.iter_mut().for_each(|v| {
            v.write(0.0);
        });

        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        // someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            let valid_indices = if node.river != NOT_DEALT {
                &self.valid_indices_river[card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.valid_indices_turn[node.turn as usize]
            } else {
                &self.valid_indices_flop
            };

            let opponent_indices = &valid_indices[player ^ 1];
            for &i in opponent_indices {
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(i as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(i as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let player_indices = &valid_indices[player];
            let same_hand_index = &self.same_hand_index[player];
            for &i in player_indices {
                unsafe {
                    let (c1, c2) = *player_cards.get_unchecked(i as usize);
                    let same_i = *same_hand_index.get_unchecked(i as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };
                    // inclusion-exclusion principle
                    let cfreach = cfreach_sum + cfreach_same
                        - *cfreach_minus.get_unchecked(c1 as usize)
                        - *cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(i as usize) = (payoff * cfreach) as f32;
                }
            }
        }
        // equity terminal (flop-only mode: non-fold terminal at flop level)
        // Uses equity-based evaluation by averaging showdown results over all runouts
        // Note: EV map lookup was disabled due to extraction issues and theoretical
        // limitations (static EVs don't help CFR converge - see PDCFR+ paper)
        else if node.turn == NOT_DEALT && self.is_flop_only_mode() {
            self.evaluate_flop_equity(result, node, player, cfreach);
        }
        // showdown (optimized for no rake; 2-pass)
        else if rake == 0.0 {
            let pair_index = card_pair_to_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let mut i = 1;

            for &StrengthItem { strength, index } in valid_player_strength {
                unsafe {
                    while opponent_strength.get_unchecked(i).strength < strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i += 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) = (amount_win * cfreach) as f32;
                }
            }

            cfreach_sum = 0.0;
            cfreach_minus.fill(0.0);
            i = opponent_strength.len() - 2;

            for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
                unsafe {
                    while opponent_strength.get_unchecked(i).strength > strength {
                        let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                        let cfreach_i = *cfreach.get_unchecked(opponent_index);
                        if cfreach_i != 0.0 {
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i_f64 = cfreach_i as f64;
                            cfreach_sum += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                            *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                        }
                        i -= 1;
                    }
                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    *result.get_unchecked_mut(index as usize) += (amount_lose * cfreach) as f32;
                }
            }
        }
        // showdown (raked; 3-pass)
        else {
            let amount_tie = -0.5 * rake / self.num_combinations;
            let same_hand_index = &self.same_hand_index[player];

            let pair_index = card_pair_to_index(node.turn, node.river);
            let hand_strength = &self.hand_strength[pair_index];
            let player_strength = &hand_strength[player];
            let opponent_strength = &hand_strength[player ^ 1];

            let valid_player_strength = &player_strength[1..player_strength.len() - 1];
            let valid_opponent_strength = &opponent_strength[1..opponent_strength.len() - 1];

            for &StrengthItem { index, .. } in valid_opponent_strength {
                unsafe {
                    let cfreach_i = *cfreach.get_unchecked(index as usize);
                    if cfreach_i != 0.0 {
                        let (c1, c2) = *opponent_cards.get_unchecked(index as usize);
                        let cfreach_i_f64 = cfreach_i as f64;
                        cfreach_sum += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c1 as usize) += cfreach_i_f64;
                        *cfreach_minus.get_unchecked_mut(c2 as usize) += cfreach_i_f64;
                    }
                }
            }

            if cfreach_sum == 0.0 {
                return;
            }

            let mut cfreach_sum_win = 0.0;
            let mut cfreach_sum_tie = 0.0;
            let mut cfreach_minus_win = [0.0; 52];
            let mut cfreach_minus_tie = [0.0; 52];

            let mut i = 1;
            let mut j = 1;
            let mut prev_strength = 0; // strength is always > 0

            for &StrengthItem { strength, index } in valid_player_strength {
                unsafe {
                    if strength > prev_strength {
                        prev_strength = strength;

                        if i < j {
                            cfreach_sum_win = cfreach_sum_tie;
                            cfreach_minus_win = cfreach_minus_tie;
                            i = j;
                        }

                        while opponent_strength.get_unchecked(i).strength < strength {
                            let opponent_index = opponent_strength.get_unchecked(i).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_i = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_win += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c1 as usize) += cfreach_i;
                            *cfreach_minus_win.get_unchecked_mut(c2 as usize) += cfreach_i;
                            i += 1;
                        }

                        if j < i {
                            cfreach_sum_tie = cfreach_sum_win;
                            cfreach_minus_tie = cfreach_minus_win;
                            j = i;
                        }

                        while opponent_strength.get_unchecked(j).strength == strength {
                            let opponent_index = opponent_strength.get_unchecked(j).index as usize;
                            let (c1, c2) = *opponent_cards.get_unchecked(opponent_index);
                            let cfreach_j = *cfreach.get_unchecked(opponent_index) as f64;
                            cfreach_sum_tie += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c1 as usize) += cfreach_j;
                            *cfreach_minus_tie.get_unchecked_mut(c2 as usize) += cfreach_j;
                            j += 1;
                        }
                    }

                    let (c1, c2) = *player_cards.get_unchecked(index as usize);
                    let cfreach_total = cfreach_sum
                        - cfreach_minus.get_unchecked(c1 as usize)
                        - cfreach_minus.get_unchecked(c2 as usize);
                    let cfreach_win = cfreach_sum_win
                        - cfreach_minus_win.get_unchecked(c1 as usize)
                        - cfreach_minus_win.get_unchecked(c2 as usize);
                    let cfreach_tie = cfreach_sum_tie
                        - cfreach_minus_tie.get_unchecked(c1 as usize)
                        - cfreach_minus_tie.get_unchecked(c2 as usize);
                    let same_i = *same_hand_index.get_unchecked(index as usize);
                    let cfreach_same = if same_i == u16::MAX {
                        0.0
                    } else {
                        *cfreach.get_unchecked(same_i as usize) as f64
                    };

                    let cfvalue = amount_win * cfreach_win
                        + amount_tie * (cfreach_tie - cfreach_win + cfreach_same)
                        + amount_lose * (cfreach_total - cfreach_tie);
                    *result.get_unchecked_mut(index as usize) = cfvalue as f32;
                }
            }
        }
    }

    pub(super) fn evaluate_internal_bunching(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = ((half_pot - rake) / self.bunching_num_combinations) as f32;
        let amount_lose = (-half_pot / self.bunching_num_combinations) as f32;
        let amount_tie = (-0.5 * rake / self.bunching_num_combinations) as f32;
        let opponent_len = self.private_cards[player ^ 1].len();

        // someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            let indices = if node.river != NOT_DEALT {
                &self.bunching_num_river[player][card_pair_to_index(node.turn, node.river)]
            } else if node.turn != NOT_DEALT {
                &self.bunching_num_turn[player][node.turn as usize]
            } else {
                &self.bunching_num_flop[player]
            };

            result.iter_mut().zip(indices).for_each(|(r, &index)| {
                if index != 0 {
                    let slice = &self.bunching_arena[index..index + opponent_len];
                    r.write(payoff * inner_product(cfreach, slice));
                } else {
                    r.write(0.0);
                }
            });
        }
        // showdown
        else {
            let pair_index = card_pair_to_index(node.turn, node.river);
            let indices = &self.bunching_num_river[player][pair_index];
            let player_strength = &self.bunching_strength[pair_index][player];
            let opponent_strength = &self.bunching_strength[pair_index][player ^ 1];

            result
                .iter_mut()
                .zip(indices)
                .zip(player_strength)
                .for_each(|((r, &index), &strength)| {
                    if index != 0 {
                        r.write(inner_product_cond(
                            cfreach,
                            &self.bunching_arena[index..index + opponent_len],
                            opponent_strength,
                            strength,
                            amount_win,
                            amount_lose,
                            amount_tie,
                        ));
                    } else {
                        r.write(0.0);
                    }
                });
        }
    }

    /// Evaluates equity at flop by averaging showdown results over all turn/river runouts.
    /// This is used for flop-only solving mode.
    fn evaluate_flop_equity(
        &self,
        result: &mut [f32],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);
        let amount_win = (half_pot - rake) / self.num_combinations;
        let amount_lose = -half_pot / self.num_combinations;

        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        // Initialize result to zero
        result.iter_mut().for_each(|v| {
            *v = 0.0;
        });

        // Count total runouts for normalization
        let mut total_runouts = 0u32;

        // Iterate over all possible turn cards
        for turn in 0u8..52 {
            let turn_mask = 1u64 << turn;
            if turn_mask & flop_mask != 0 {
                continue;
            }

            // Iterate over all possible river cards
            for river in 0u8..52 {
                let river_mask = 1u64 << river;
                if river_mask & (flop_mask | turn_mask) != 0 {
                    continue;
                }

                total_runouts += 1;
                let pair_index = card_pair_to_index(turn, river);
                let board_mask = flop_mask | turn_mask | river_mask;

                // Get hand strength for this runout
                let hand_strength = &self.hand_strength[pair_index];
                let player_strength = &hand_strength[player];
                let opponent_strength = &hand_strength[player ^ 1];

                // Skip if no valid hands for this runout
                if player_strength.is_empty() || opponent_strength.is_empty() {
                    continue;
                }

                let valid_player_strength = &player_strength[1..player_strength.len() - 1];
                let valid_opponent_strength = &opponent_strength[1..opponent_strength.len() - 1];

                // Build opponent reach sums for this runout
                let mut cfreach_sum_runout = 0.0f64;
                let mut cfreach_minus_runout = [0.0f64; 52];

                for &StrengthItem { index, .. } in valid_opponent_strength {
                    let (c1, c2) = opponent_cards[index as usize];
                    let hand_mask = (1u64 << c1) | (1u64 << c2);
                    if hand_mask & board_mask != 0 {
                        continue;
                    }
                    let cfreach_i = cfreach[index as usize] as f64;
                    if cfreach_i != 0.0 {
                        cfreach_sum_runout += cfreach_i;
                        cfreach_minus_runout[c1 as usize] += cfreach_i;
                        cfreach_minus_runout[c2 as usize] += cfreach_i;
                    }
                }

                if cfreach_sum_runout == 0.0 {
                    continue;
                }

                // First pass: count wins (opponents with lower strength)
                let mut cfreach_sum_win = 0.0f64;
                let mut cfreach_minus_win = [0.0f64; 52];
                let mut opp_idx = 1usize;

                for &StrengthItem { strength, index } in valid_player_strength {
                    let (c1, c2) = player_cards[index as usize];
                    let hand_mask = (1u64 << c1) | (1u64 << c2);
                    if hand_mask & board_mask != 0 {
                        continue;
                    }

                    while opp_idx < opponent_strength.len() - 1
                        && opponent_strength[opp_idx].strength < strength
                    {
                        let opp_item = &opponent_strength[opp_idx];
                        let (oc1, oc2) = opponent_cards[opp_item.index as usize];
                        let opp_mask = (1u64 << oc1) | (1u64 << oc2);
                        if opp_mask & board_mask == 0 {
                            let cf = cfreach[opp_item.index as usize] as f64;
                            if cf != 0.0 {
                                cfreach_sum_win += cf;
                                cfreach_minus_win[oc1 as usize] += cf;
                                cfreach_minus_win[oc2 as usize] += cf;
                            }
                        }
                        opp_idx += 1;
                    }

                    let cfreach_win = cfreach_sum_win
                        - cfreach_minus_win[c1 as usize]
                        - cfreach_minus_win[c2 as usize];

                    result[index as usize] += (amount_win * cfreach_win) as f32;
                }

                // Second pass: count losses (opponents with higher strength)
                let mut cfreach_sum_lose = 0.0f64;
                let mut cfreach_minus_lose = [0.0f64; 52];
                opp_idx = opponent_strength.len() - 2;

                for &StrengthItem { strength, index } in valid_player_strength.iter().rev() {
                    let (c1, c2) = player_cards[index as usize];
                    let hand_mask = (1u64 << c1) | (1u64 << c2);
                    if hand_mask & board_mask != 0 {
                        continue;
                    }

                    while opp_idx > 0 && opponent_strength[opp_idx].strength > strength {
                        let opp_item = &opponent_strength[opp_idx];
                        let (oc1, oc2) = opponent_cards[opp_item.index as usize];
                        let opp_mask = (1u64 << oc1) | (1u64 << oc2);
                        if opp_mask & board_mask == 0 {
                            let cf = cfreach[opp_item.index as usize] as f64;
                            if cf != 0.0 {
                                cfreach_sum_lose += cf;
                                cfreach_minus_lose[oc1 as usize] += cf;
                                cfreach_minus_lose[oc2 as usize] += cf;
                            }
                        }
                        opp_idx -= 1;
                    }

                    let cfreach_lose = cfreach_sum_lose
                        - cfreach_minus_lose[c1 as usize]
                        - cfreach_minus_lose[c2 as usize];

                    result[index as usize] += (amount_lose * cfreach_lose) as f32;
                }
            }
        }

        // Normalize by number of runouts
        if total_runouts > 0 {
            let scale = 1.0 / total_runouts as f32;
            for r in result.iter_mut() {
                *r *= scale;
            }
        }
    }

    /// Evaluates flop terminal using precomputed EVs from the EV map.
    /// This is the "Perfect Leaf Evaluator" for flop-only solving.
    ///
    /// Uses Range-Agnostic transfer via Equity-Ratio Scaling:
    ///   EV_final = Precomputed_EV × (Current_Equity / Precomputed_Equity)
    #[cfg(feature = "bincode")]
    fn evaluate_with_ev_map(
        &self,
        result: &mut [f32],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let ev_map = match &self.ev_map {
            Some(map) => map,
            None => {
                // Fallback to equity-based evaluation
                self.evaluate_flop_equity(result, node, player, cfreach);
                return;
            }
        };

        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;

        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        // Compute current equity for each hand against opponent's reach-weighted range
        let current_equities = self.compute_current_equities(player, cfreach);

        // Compute path hash for this terminal using compound key (pot_size, prev_action)
        let path_hash = self.compute_path_hash(node);

        // Initialize result to zero
        result.iter_mut().for_each(|v| *v = 0.0);

        // Valid opponent indices for card blocking
        let valid_indices = &self.valid_indices_flop[player ^ 1];

        // Compute opponent reach sum and per-card reach for blocking
        let mut cfreach_sum = 0.0f64;
        let mut cfreach_minus = [0.0f64; 52];

        for &i in valid_indices {
            let cfreach_i = cfreach[i as usize] as f64;
            if cfreach_i != 0.0 {
                let (c1, c2) = opponent_cards[i as usize];
                cfreach_sum += cfreach_i;
                cfreach_minus[c1 as usize] += cfreach_i;
                cfreach_minus[c2 as usize] += cfreach_i;
            }
        }

        if cfreach_sum == 0.0 {
            return;
        }

        let player_indices = &self.valid_indices_flop[player];

        for &i in player_indices {
            let (c1, c2) = player_cards[i as usize];
            let hand_mask = (1u64 << c1) | (1u64 << c2);

            // Skip if hand conflicts with board
            if hand_mask & flop_mask != 0 {
                continue;
            }

            let hand = if c1 < c2 { (c1, c2) } else { (c2, c1) };
            let current_equity = current_equities[i as usize];

            // Look up EV from the map with equity-ratio scaling
            // DEBUG: Track lookup stats (only log once per solve)
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            static LOGGED: AtomicBool = AtomicBool::new(false);
            static HITS: AtomicUsize = AtomicUsize::new(0);
            static MISSES: AtomicUsize = AtomicUsize::new(0);

            if let Some(ev) = ev_map.get_ev_with_transfer(path_hash, player, hand, current_equity) {
                HITS.fetch_add(1, Ordering::Relaxed);
                // Apply card removal / blocking adjustment
                // Compute the opponent weight after removing our cards
                let opponent_weight = cfreach_sum
                    - cfreach_minus[c1 as usize]
                    - cfreach_minus[c2 as usize];

                // Scale EV by opponent reach (for CFR weighting)
                result[i as usize] = (ev as f64 * opponent_weight / self.num_combinations) as f32;
            } else {
                MISSES.fetch_add(1, Ordering::Relaxed);
                // Log first few misses to debug
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!("[EV_MAP DEBUG] First miss: path_hash={}, player={}, hand=({},{}), equity={}",
                        path_hash, player, hand.0, hand.1, current_equity);
                    eprintln!("[EV_MAP DEBUG] Available path_hashes: {:?}",
                        ev_map.terminals.keys().collect::<Vec<_>>());
                    // Show sample hands in the map for this terminal
                    if let Some(terminal) = ev_map.terminals.get(&path_hash) {
                        let data = if player == 0 { &terminal.oop_data } else { &terminal.ip_data };

                        // Check specifically for our hand and nearby hands
                        eprintln!("[EV_MAP DEBUG] Checking for hand (0,1): {:?}", data.get(&(0u8, 1u8)));
                        eprintln!("[EV_MAP DEBUG] Checking for hand (1,0): {:?}", data.get(&(1u8, 0u8)));
                        eprintln!("[EV_MAP DEBUG] Checking for hand (0,2): {:?}", data.get(&(0u8, 2u8)));
                        eprintln!("[EV_MAP DEBUG] Checking for hand (0,3): {:?}", data.get(&(0u8, 3u8)));

                        // Show first 10 hands sorted
                        let mut all_hands: Vec<_> = data.keys().cloned().collect();
                        all_hands.sort();
                        eprintln!("[EV_MAP DEBUG] First 10 hands (sorted): {:?}", all_hands.iter().take(10).collect::<Vec<_>>());
                        eprintln!("[EV_MAP DEBUG] Total hands in map: {}", data.len());

                        // Check hand range - min and max cards
                        let min_c1 = all_hands.iter().map(|h| h.0).min();
                        let max_c1 = all_hands.iter().map(|h| h.0).max();
                        eprintln!("[EV_MAP DEBUG] Card1 range: {:?} to {:?}", min_c1, max_c1);
                    }
                }

                // Fallback: use simple equity-based value
                let opponent_weight = cfreach_sum
                    - cfreach_minus[c1 as usize]
                    - cfreach_minus[c2 as usize];

                let amount_win = (half_pot) / self.num_combinations;
                let amount_lose = -half_pot / self.num_combinations;

                // Simple equity-based fallback
                let ev = current_equity as f64 * amount_win
                    + (1.0 - current_equity as f64) * amount_lose;
                result[i as usize] = (ev * opponent_weight) as f32;
            }

            // Log stats periodically
            let total = HITS.load(Ordering::Relaxed) + MISSES.load(Ordering::Relaxed);
            if total > 0 && total % 1000000 == 0 {
                eprintln!("[EV_MAP STATS] hits={}, misses={}, hit_rate={:.1}%",
                    HITS.load(Ordering::Relaxed),
                    MISSES.load(Ordering::Relaxed),
                    100.0 * HITS.load(Ordering::Relaxed) as f64 / total as f64);
            }
        }
    }

    /// Compute current equity for each hand against opponent's reach-weighted range.
    #[cfg(feature = "bincode")]
    fn compute_current_equities(&self, player: usize, cfreach: &[f32]) -> Vec<f32> {
        let player_cards = &self.private_cards[player];
        let opponent_cards = &self.private_cards[player ^ 1];

        let flop = self.card_config.flop;
        let flop_mask: u64 = (1 << flop[0]) | (1 << flop[1]) | (1 << flop[2]);

        let mut equities = vec![0.0f32; player_cards.len()];

        // For each player hand, compute equity against opponent's reach-weighted range
        for (hand_idx, &(c1, c2)) in player_cards.iter().enumerate() {
            let hand_mask = (1u64 << c1) | (1u64 << c2);

            // Skip if hand conflicts with board
            if hand_mask & flop_mask != 0 {
                continue;
            }

            let mut total_weight = 0.0f64;
            let mut win_weight = 0.0f64;

            // Compare against each opponent hand
            for (opp_idx, &(oc1, oc2)) in opponent_cards.iter().enumerate() {
                let opp_mask = (1u64 << oc1) | (1u64 << oc2);

                // Skip if opponent hand conflicts with our hand or board
                if opp_mask & (hand_mask | flop_mask) != 0 {
                    continue;
                }

                let weight = cfreach[opp_idx] as f64;
                if weight <= 0.0 {
                    continue;
                }

                total_weight += weight;

                // Simple hand strength comparison on flop
                let our_strength = self.simple_hand_strength_flop(c1, c2);
                let opp_strength = self.simple_hand_strength_flop(oc1, oc2);

                if our_strength > opp_strength {
                    win_weight += weight;
                } else if our_strength == opp_strength {
                    win_weight += weight * 0.5;
                }
            }

            equities[hand_idx] = if total_weight > 0.0 {
                (win_weight / total_weight) as f32
            } else {
                0.5
            };
        }

        equities
    }

    /// Simple hand strength on flop (approximation for equity calculation).
    #[cfg(feature = "bincode")]
    fn simple_hand_strength_flop(&self, c1: Card, c2: Card) -> u32 {
        let flop = &self.card_config.flop;
        let flop_ranks: [u8; 3] = [flop[0] >> 2, flop[1] >> 2, flop[2] >> 2];
        let flop_suits: [u8; 3] = [flop[0] & 3, flop[1] & 3, flop[2] & 3];

        let rank1 = c1 >> 2;
        let rank2 = c2 >> 2;
        let suit1 = c1 & 3;
        let suit2 = c2 & 3;

        let high = rank1.max(rank2);
        let low = rank1.min(rank2);
        let paired = rank1 == rank2;

        // Check for trips
        let matching_flop = flop_ranks.iter().filter(|&&r| r == rank1 || r == rank2).count();
        if matching_flop >= 2 && paired {
            return 7000 + high as u32; // Full house potential
        }
        if matching_flop >= 2 {
            return 6000 + high as u32; // Trips
        }

        // Check for two pair
        let board_pairs: Vec<u8> = flop_ranks.iter()
            .filter(|&&r| r == rank1 || r == rank2)
            .cloned()
            .collect();
        if board_pairs.len() == 2 && !paired {
            return 5000 + high as u32 * 15 + low as u32; // Two pair
        }

        // Check for pair
        if paired {
            let overpair = flop_ranks.iter().all(|&r| rank1 > r);
            if overpair {
                return 4500 + high as u32 * 15; // Overpair
            }
            return 4000 + high as u32 * 15; // Pocket pair
        }

        if board_pairs.len() == 1 {
            // One pair with board
            let pair_rank = board_pairs[0];
            let kicker = if pair_rank == rank1 { rank2 } else { rank1 };
            return 3000 + pair_rank as u32 * 15 + kicker as u32; // Top/middle/bottom pair
        }

        // Flush draw
        let flush_suit = if suit1 == suit2 { Some(suit1) } else { None };
        let flush_draw = flush_suit.map_or(false, |s| {
            flop_suits.iter().filter(|&&fs| fs == s).count() >= 2
        });

        // High card with draws
        let base = high as u32 * 15 + low as u32;
        if flush_draw {
            return 2000 + base; // Flush draw
        }

        // Pure high card
        base
    }

    /// Compute path hash for the current terminal.
    /// Uses compound key (pot_size, prev_action) to uniquely identify chance nodes.
    #[cfg(feature = "bincode")]
    fn compute_path_hash(&self, node: &PostFlopNode) -> u64 {
        use crate::ev_map::EvMap;
        let pot = self.tree_config.starting_pot + 2 * node.amount;
        EvMap::compute_compound_key(pot, node.prev_action)
    }
}
