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

/// Abstracted evaluation using precomputed bucket equity tables.
///
/// This function computes CFV for each bucket instead of each hand,
/// using O(k²) operations instead of O(n²) where k is number of buckets.
///
/// Key insight: cfreach already incorporates bucket weights through the
/// effective_initial_weights mechanism. We don't need to multiply by
/// weights again here.
impl PostFlopGame {
    pub(super) fn evaluate_internal_abstracted(
        &self,
        result: &mut [MaybeUninit<f32>],
        node: &PostFlopNode,
        player: usize,
        cfreach: &[f32],
    ) {
        let abstraction_data = self.abstraction_data.as_ref().unwrap();
        let num_player_buckets = abstraction_data.num_buckets(player);
        let num_opponent_buckets = abstraction_data.num_buckets(player ^ 1);

        let pot = (self.tree_config.starting_pot + 2 * node.amount) as f64;
        let half_pot = 0.5 * pot;
        let rake = min(pot * self.tree_config.rake_rate, self.tree_config.rake_cap);

        // Normalize by num_combinations like the original evaluation
        let amount_win = ((half_pot - rake) / self.num_combinations) as f32;
        let amount_lose = (-half_pot / self.num_combinations) as f32;

        // Initialize result to zeros
        result.iter_mut().for_each(|v| {
            v.write(0.0);
        });
        let result = unsafe { &mut *(result as *mut _ as *mut [f32]) };

        // Someone folded
        if node.player & PLAYER_FOLD_FLAG == PLAYER_FOLD_FLAG {
            let folded_player = node.player & PLAYER_MASK;
            let payoff = if folded_player as usize != player {
                amount_win
            } else {
                amount_lose
            };

            // Get bucket matchups to account for card blocking
            // For fold evaluation, card blocking depends only on private cards, not the runout.
            // If the specific runout is unavailable (e.g., river=NOT_DEALT), use the first runout's matchups.
            let bucket_matchups = if node.river != NOT_DEALT {
                abstraction_data.get_bucket_matchups(node.turn, node.river)
            } else {
                // Use first runout's matchups (blocking is same for all runouts)
                abstraction_data.bucket_matchups.first()
            };

            if let Some(matchups_matrix) = bucket_matchups {
                // For each player bucket, compute CFV accounting for card blocking
                // CFV[b] = payoff * sum_o(cfreach[o] * non_blocking_fraction[b][o])
                // where non_blocking_fraction = matchups[b][o] / (|bucket_b| * |bucket_o|)
                //
                // NOTE: matchups_matrix is indexed as [oop_bucket][ip_bucket]
                // We must access it correctly based on which player we're evaluating

                for player_bucket in 0..num_player_buckets {
                    let mut cfv = 0.0f32;
                    let player_bucket_size = abstraction_data.bucket_to_hands[player][player_bucket].len() as f32;

                    for opp_bucket in 0..num_opponent_buckets {
                        let opp_reach = cfreach[opp_bucket];
                        if opp_reach > 0.0 {
                            // matchups_matrix is [oop][ip], so index correctly based on player
                            let matchup_count = if player == 0 {
                                // Player is OOP: matchups[oop_bucket][ip_bucket]
                                matchups_matrix[player_bucket][opp_bucket] as f32
                            } else {
                                // Player is IP: matchups[oop_bucket][ip_bucket] where oop=opp, ip=player
                                matchups_matrix[opp_bucket][player_bucket] as f32
                            };
                            let opp_bucket_size = abstraction_data.bucket_to_hands[player ^ 1][opp_bucket].len() as f32;

                            if opp_bucket_size > 0.0 && player_bucket_size > 0.0 {
                                // non_blocking_fraction approximates the fraction of hand pairs
                                // between these buckets that don't share cards
                                let non_blocking_fraction = matchup_count / (player_bucket_size * opp_bucket_size);
                                cfv += opp_reach * non_blocking_fraction;
                            }
                        }
                    }

                    result[player_bucket] = payoff * cfv;
                }
            } else {
                // Fallback: no matchups data, use simple sum (shouldn't happen)
                let cfreach_sum: f32 = cfreach[..num_opponent_buckets].iter().sum();
                for b in 0..num_player_buckets {
                    result[b] = payoff * cfreach_sum;
                }
            }
        }
        // Showdown
        else {
            // Get bucket equity for this runout
            let bucket_equity = abstraction_data.get_bucket_equity(node.turn, node.river);

            if let Some(equity_matrix) = bucket_equity {
                // For each player bucket, compute weighted CFV
                // NOTE: equity_matrix is indexed as [oop_bucket][ip_bucket] and stores OOP's equity
                for player_bucket in 0..num_player_buckets {
                    let mut cfv = 0.0f32;

                    for opp_bucket in 0..num_opponent_buckets {
                        let opp_reach = cfreach[opp_bucket];
                        if opp_reach > 0.0 {
                            // equity_matrix[oop][ip] = OOP's equity
                            // For OOP (player=0): use equity directly
                            // For IP (player=1): use 1 - equity (opponent is OOP, so we flip)
                            let equity = if player == 0 {
                                equity_matrix[player_bucket][opp_bucket]
                            } else {
                                1.0 - equity_matrix[opp_bucket][player_bucket]
                            };
                            // CFV = reach * (equity * win + (1-equity) * lose)
                            let value = equity * amount_win + (1.0 - equity) * amount_lose;
                            cfv += opp_reach * value;
                        }
                    }

                    result[player_bucket] = cfv;
                }
            } else {
                // No equity data for this runout (shouldn't happen normally)
                for b in 0..num_player_buckets {
                    result[b] = 0.0;
                }
            }
        }
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
}
