//! Oracle-based DCFR solver: only updates flop nodes, uses matrices extracted
//! from the full solved tree at the turn boundary for CFV evaluation.
//!
//! This is the DeepStack concept: at the flop→turn chance node, instead of
//! recursing into the turn/river subtree, query a pre-extracted matrix that
//! returns CFVs via matrix multiply: CFV = M × cfreach.
//!
//! Option 1 approach: extract matrices from the full solved tree (not from
//! independently solved turn games). This guarantees exact match with the
//! full tree's turn boundary CFVs (same chance_factor, isomorphism, strategies).
//!
//! No library code is modified. Everything uses the public API.

#![allow(dead_code)]

use postflop_solver::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{self, Read as _, Write};
use std::mem::MaybeUninit;

// =============================================================================
// Tree-extracted Oracle
// =============================================================================

/// Oracle that stores CFV matrices extracted directly from a fully-solved tree.
/// Matrices are in flop hand indexing (no remapping needed).
/// Each matrix captures the full turn/river subtree behavior including
/// chance_factor scaling, isomorphism, and Nash strategies.
pub struct TreeOracle {
    /// matrices[amount][player] = flat row-major matrix [num_player_hands × num_opponent_hands]
    /// CFV_player[i] = Σ_j matrix[i * num_opp + j] * cfreach_opp[j]
    matrices: HashMap<i32, [Vec<f32>; 2]>,
    num_hands: [usize; 2],
}

impl TreeOracle {
    /// Create oracle from pre-built matrices.
    pub fn from_matrices(matrices: HashMap<i32, [Vec<f32>; 2]>, num_hands: [usize; 2]) -> Self {
        Self { matrices, num_hands }
    }

    /// Build oracle by probing the fully solved tree at turn boundary nodes.
    /// For each boundary amount, extracts a matrix by calling compute_cfvalue_recursive
    /// with basis vectors at the turn chance node.
    pub fn build(game: &PostFlopGame, print_progress: bool) -> Self {
        let num_hands = [game.num_private_hands(0), game.num_private_hands(1)];
        let mut matrices: HashMap<i32, [Vec<f32>; 2]> = HashMap::new();

        let mut root = game.root();
        Self::extract_recursive(game, &mut root, &mut matrices, num_hands, print_progress);

        if print_progress {
            println!("  Extracted {} boundary amounts", matrices.len());
            for (&amount, _mats) in &matrices {
                println!("    amount={}: OOP matrix {}x{}, IP matrix {}x{}",
                    amount,
                    num_hands[0], num_hands[1],
                    num_hands[1], num_hands[0],
                );
            }
        }

        Self { matrices, num_hands }
    }

    fn extract_recursive(
        game: &PostFlopGame,
        node: &mut PostFlopNode,
        matrices: &mut HashMap<i32, [Vec<f32>; 2]>,
        num_hands: [usize; 2],
        print_progress: bool,
    ) {
        if node.is_terminal() {
            return;
        }

        // Found a turn chance node (flop→turn boundary)
        if node.is_chance() && node.turn() == NOT_DEALT {
            let amount = node.amount();
            if !matrices.contains_key(&amount) {
                if print_progress {
                    print!("  Extracting matrix for amount={}...", amount);
                    io::stdout().flush().unwrap();
                }

                let mut player_matrices: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
                for player in 0..2 {
                    let pname = if player == 0 { "OOP" } else { "IP" };
                    let n_player = num_hands[player];
                    let n_opp = num_hands[player ^ 1];

                    let probe_start = std::time::Instant::now();

                    // Parallel basis vector probing with rayon.
                    // compute_cfvalue_recursive takes &mut node but is read-only on tree data.
                    // Wrap in MutexLike for parallel access (same pattern as library solver).
                    let node_wrapper = MutexLike::new(node as *mut PostFlopNode as usize);
                    let columns: Vec<Vec<f32>> = (0..n_opp)
                        .into_par_iter()
                        .map(|j| {
                            let mut basis = vec![0.0f32; n_opp];
                            basis[j] = 1.0;
                            let mut result = vec![MaybeUninit::<f32>::uninit(); n_player];
                            let node_ptr = *node_wrapper.lock() as *mut PostFlopNode;
                            compute_cfvalue_recursive(
                                &mut result, game, unsafe { &mut *node_ptr }, player, &basis, false,
                            );
                            result.iter().map(|v| unsafe { v.assume_init() }).collect()
                        })
                        .collect();

                    // Assemble into row-major matrix
                    let mut matrix = vec![0.0f32; n_player * n_opp];
                    for j in 0..n_opp {
                        for i in 0..n_player {
                            matrix[i * n_opp + j] = columns[j][i];
                        }
                    }

                    if print_progress {
                        let elapsed = probe_start.elapsed().as_secs_f64();
                        print!("\r  amount={} {} probing: {}/{} ({:.1}s)              ",
                            amount, pname, n_opp, n_opp, elapsed);
                        io::stdout().flush().unwrap();
                    }

                    player_matrices[player] = matrix;
                }
                matrices.insert(amount, player_matrices);

                if print_progress {
                    println!("\r  amount={}: done                                              ", amount);
                }
            }
            return; // Don't recurse into turn/river
        }

        // Continue walking the flop tree
        let num_actions = node.num_actions();
        for action in 0..num_actions {
            let mut child = node.play(action);
            Self::extract_recursive(game, &mut child, matrices, num_hands, print_progress);
        }
    }

    /// Evaluate turn boundary: result = matrix × cfreach (simple matrix-vector multiply)
    pub fn evaluate_turn_boundary(
        &self,
        result: &mut [MaybeUninit<f32>],
        amount: i32,
        player: usize,
        cfreach: &[f32],
    ) {
        let matrix = match self.matrices.get(&amount) {
            Some(m) => &m[player],
            None => panic!("TreeOracle: no matrix for amount={}", amount),
        };
        let num_player = self.num_hands[player];
        let num_opp = self.num_hands[player ^ 1];

        for i in 0..num_player {
            let mut sum = 0.0f32;
            let row_start = i * num_opp;
            for j in 0..num_opp {
                sum += matrix[row_start + j] * cfreach[j];
            }
            result[i].write(sum);
        }
    }

    pub fn num_amounts(&self) -> usize {
        self.matrices.len()
    }

    pub fn num_hands(&self) -> [usize; 2] {
        self.num_hands
    }

    /// Save oracle to binary file.
    /// Format: magic(8) + num_oop(4) + num_ip(4) + num_amounts(4)
    ///         + [amount(4) + oop_matrix(f32s) + ip_matrix(f32s)] × num_amounts
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut f = io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(b"TORACLE\0")?;
        f.write_all(&(self.num_hands[0] as u32).to_le_bytes())?;
        f.write_all(&(self.num_hands[1] as u32).to_le_bytes())?;
        f.write_all(&(self.matrices.len() as u32).to_le_bytes())?;

        let mut amounts: Vec<_> = self.matrices.keys().collect();
        amounts.sort();
        for &amount in &amounts {
            let mats = &self.matrices[amount];
            f.write_all(&amount.to_le_bytes())?;
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
        f.flush()?;
        Ok(())
    }

    /// Load oracle from binary file.
    pub fn load(path: &str) -> io::Result<Self> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"TORACLE\0" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let num_oop = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_ip = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_amounts = u32::from_le_bytes(buf4) as usize;

        let num_hands = [num_oop, num_ip];
        let mut matrices = HashMap::with_capacity(num_amounts);

        for _ in 0..num_amounts {
            f.read_exact(&mut buf4)?;
            let amount = i32::from_le_bytes(buf4);

            let mut player_matrices: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
            for player in 0..2 {
                let n_player = num_hands[player];
                let n_opp = num_hands[player ^ 1];
                let mut data = vec![0.0f32; n_player * n_opp];
                let bytes: &mut [u8] = unsafe {
                    std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
                };
                f.read_exact(bytes)?;
                player_matrices[player] = data;
            }
            matrices.insert(amount, player_matrices);
        }

        Ok(Self { matrices, num_hands })
    }
}

// =============================================================================
// DCFR Discount Parameters (reimplemented from solver.rs)
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
// Reset flop storage (tree traversal via public API)
// =============================================================================

/// Zeros out regrets and strategy for flop action nodes only.
/// Turn/river node strategies are preserved (they exist but won't be used by oracle).
/// Returns the number of flop action nodes reset.
pub fn reset_flop_storage(game: &PostFlopGame) -> usize {
    let mut root = game.root();
    reset_flop_node(&mut root)
}

fn reset_flop_node(node: &mut PostFlopNode) -> usize {
    if node.turn() != NOT_DEALT || node.is_terminal() {
        return 0;
    }

    let mut count = 0;

    if !node.is_chance() {
        for v in node.regrets_mut() {
            *v = 0.0;
        }
        for v in node.strategy_mut() {
            *v = 0.0;
        }
        count = 1;
    }

    let num_actions = node.num_actions();
    for action in 0..num_actions {
        let mut child = node.play(action);
        count += reset_flop_node(&mut child);
    }
    count
}

// =============================================================================
// Oracle DCFR solve loop
// =============================================================================

/// Runs DCFR on flop nodes only. Turn/river values come from tree-extracted oracle.
pub fn solve_flop_with_oracle(
    game: &PostFlopGame,
    oracle: &TreeOracle,
    max_iterations: u32,
    target_exploitability: f32,
    print_progress: bool,
) -> f32 {
    let starting_pot = game.starting_pot() as f32;
    let mut exploitability = compute_exploitability(game);

    if print_progress {
        let pct = exploitability / starting_pot * 100.0;
        let target_pct = target_exploitability / starting_pot * 100.0;
        print!("  oracle iter: 0 / {} (exploitability = {:.2}% | target = {:.2}%)",
            max_iterations, pct, target_pct);
        io::stdout().flush().unwrap();
    }

    for t in 0..max_iterations {
        if exploitability <= target_exploitability {
            break;
        }

        let params = DiscountParams::new(t);

        for player in 0..2 {
            let num_hands = game.num_private_hands(player);
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            let mut root = game.root();
            solve_recursive_with_oracle(
                &mut result,
                game,
                oracle,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            exploitability = compute_exploitability(game);
        }

        if print_progress {
            let pct = exploitability / starting_pot * 100.0;
            let target_pct = target_exploitability / starting_pot * 100.0;
            print!("\r  oracle iter: {} / {} (exploitability = {:.2}% | target = {:.2}%)",
                t + 1, max_iterations, pct, target_pct);
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
    }

    exploitability
}

/// Runs DCFR on flop nodes only with fixed iteration count (no exploitability check).
/// Use this when turn/river strategies don't exist (standalone oracle solve).
pub fn solve_flop_fixed_iterations(
    game: &PostFlopGame,
    oracle: &TreeOracle,
    max_iterations: u32,
    print_progress: bool,
) {
    let solve_start = std::time::Instant::now();

    if print_progress {
        print!("iteration: 0 / {} ", max_iterations);
        io::stdout().flush().unwrap();
    }

    for t in 0..max_iterations {
        let params = DiscountParams::new(t);

        for player in 0..2 {
            let num_hands = game.num_private_hands(player);
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            let mut root = game.root();
            solve_recursive_with_oracle(
                &mut result,
                game,
                oracle,
                &mut root,
                player,
                game.initial_weights(player ^ 1),
                &params,
            );
        }

        if print_progress {
            let elapsed = solve_start.elapsed().as_secs_f64();
            let per_iter = elapsed / (t + 1) as f64;
            print!("\riteration: {} / {} ({:.2}s, {:.4}s/iter)",
                t + 1, max_iterations, elapsed, per_iter);
            io::stdout().flush().unwrap();
        }
    }

    if print_progress {
        println!();
        io::stdout().flush().unwrap();
    }
}

// =============================================================================
// Oracle recursive DCFR (the core) — parallelized with rayon
// =============================================================================

fn solve_recursive_with_oracle(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    oracle: &TreeOracle,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
) {
    // Terminal node
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    // Turn chance node → oracle lookup instead of subtree traversal
    if node.is_chance() && node.turn() == NOT_DEALT {
        oracle.evaluate_turn_boundary(result, node.amount(), player, cfreach);
        return;
    }

    // Passthrough (single action, not chance)
    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        solve_recursive_with_oracle(result, game, oracle, &mut child, player, cfreach, params);
        return;
    }

    // Allocate CFV storage — use MutexLike for rayon parallel access
    let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);

    if node.player() == player {
        // --- Player node: recurse in parallel, then update strategy + regrets ---

        (0..num_actions).into_par_iter().for_each(|action| {
            let mut action_result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            {
                let mut child = node.play(action);
                solve_recursive_with_oracle(
                    &mut action_result, game, oracle, &mut child, player, cfreach, params,
                );
            }
            // Write to distinct row — no data race
            let cfv = cfv_actions.lock();
            let offset = action * num_hands;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut f32).add(offset),
                    num_hands,
                )
            };
            for (i, v) in action_result.iter().enumerate() {
                dst[i] = unsafe { v.assume_init() };
            }
        });

        let cfv_actions = cfv_actions.lock();

        // Regret matching: compute current strategy from regrets
        let strategy = regret_matching(node.regrets(), num_actions, num_hands);

        // Compute weighted CFV: result[h] = Σ_a strategy[a,h] * cfv[a,h]
        for h in 0..num_hands {
            let mut weighted = 0.0f32;
            for a in 0..num_actions {
                weighted += strategy[a * num_hands + h] * cfv_actions[a * num_hands + h];
            }
            result[h].write(weighted);
        }
        let result_f32 = unsafe { &*(result as *const _ as *const [f32]) };

        // Update cumulative strategy
        let gamma = params.gamma_t;
        let cum_strategy = node.strategy_mut();
        for (i, s) in cum_strategy.iter_mut().enumerate() {
            *s = *s * gamma + strategy[i];
        }

        // Update cumulative regrets
        let (alpha, beta) = (params.alpha_t, params.beta_t);
        let cum_regret = node.regrets_mut();
        for a in 0..num_actions {
            for h in 0..num_hands {
                let idx = a * num_hands + h;
                let instant_regret = cfv_actions[idx] - result_f32[h];
                let coef = if cum_regret[idx].is_sign_positive() { alpha } else { beta };
                cum_regret[idx] = cum_regret[idx] * coef + instant_regret;
            }
        }
    } else {
        // --- Opponent node: update cfreach by opponent's strategy, then recurse ---

        let opp_strategy = regret_matching(node.regrets(), num_actions, cfreach.len());

        let row_size = cfreach.len();
        let mut cfreach_actions = opp_strategy;
        for a in 0..num_actions {
            for h in 0..row_size {
                cfreach_actions[a * row_size + h] *= cfreach[h];
            }
        }

        (0..num_actions).into_par_iter().for_each(|action| {
            let mut action_result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            {
                let mut child = node.play(action);
                solve_recursive_with_oracle(
                    &mut action_result, game, oracle, &mut child, player,
                    &cfreach_actions[action * row_size..(action + 1) * row_size],
                    params,
                );
            }
            // Write to distinct row — no data race
            let cfv = cfv_actions.lock();
            let offset = action * num_hands;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut f32).add(offset),
                    num_hands,
                )
            };
            for (i, v) in action_result.iter().enumerate() {
                dst[i] = unsafe { v.assume_init() };
            }
        });

        let cfv_actions = cfv_actions.lock();

        // Sum CFVs across opponent's actions
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
// Read-only CFV computation with oracle (for validation)
// =============================================================================

/// Compute CFVs using the oracle at turn boundaries (read-only, no regret updates).
/// Uses normalized_strategy (average strategy) at flop nodes — same as compute_cfvalue_recursive.
pub fn compute_cfvalue_with_oracle(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    oracle: &TreeOracle,
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

    // Turn chance node → oracle
    if node.is_chance() && node.turn() == NOT_DEALT {
        oracle.evaluate_turn_boundary(result, node.amount(), player, cfreach);
        return;
    }

    // Single action passthrough
    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        compute_cfvalue_with_oracle(result, game, oracle, &mut child, player, cfreach);
        return;
    }

    let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);

    if node.player() == player {
        (0..num_actions).into_par_iter().for_each(|action| {
            let mut action_result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            {
                let mut child = node.play(action);
                compute_cfvalue_with_oracle(
                    &mut action_result, game, oracle, &mut child, player, cfreach,
                );
            }
            let cfv = cfv_actions.lock();
            let offset = action * num_hands;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut f32).add(offset),
                    num_hands,
                )
            };
            for (i, v) in action_result.iter().enumerate() {
                dst[i] = unsafe { v.assume_init() };
            }
        });

        let cfv_actions = cfv_actions.lock();

        // Use normalized strategy (average) — same as compute_cfvalue_recursive
        let strategy = normalized_strategy(node.strategy(), num_actions, num_hands);

        for h in 0..num_hands {
            let mut weighted = 0.0f32;
            for a in 0..num_actions {
                weighted += strategy[a * num_hands + h] * cfv_actions[a * num_hands + h];
            }
            result[h].write(weighted);
        }
    } else {
        if num_actions == 1 {
            let mut child = node.play(0);
            compute_cfvalue_with_oracle(result, game, oracle, &mut child, player, cfreach);
            return;
        }

        // Use normalized strategy (average) — same as compute_cfvalue_recursive
        let opp_strategy = normalized_strategy(node.strategy(), num_actions, cfreach.len());

        let row_size = cfreach.len();
        let mut cfreach_actions = opp_strategy;
        for a in 0..num_actions {
            for h in 0..row_size {
                cfreach_actions[a * row_size + h] *= cfreach[h];
            }
        }

        (0..num_actions).into_par_iter().for_each(|action| {
            let mut action_result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            {
                let mut child = node.play(action);
                compute_cfvalue_with_oracle(
                    &mut action_result, game, oracle, &mut child, player,
                    &cfreach_actions[action * row_size..(action + 1) * row_size],
                );
            }
            let cfv = cfv_actions.lock();
            let offset = action * num_hands;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (cfv.as_ptr() as *mut f32).add(offset),
                    num_hands,
                )
            };
            for (i, v) in action_result.iter().enumerate() {
                dst[i] = unsafe { v.assume_init() };
            }
        });

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
// Regret matching (reimplemented from solver.rs)
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

// =============================================================================
// Normalized strategy (reimplemented from utility.rs)
// =============================================================================

fn normalized_strategy(strategy: &[f32], num_actions: usize, num_hands: usize) -> Vec<f32> {
    let mut normalized = vec![0.0f32; num_actions * num_hands];

    for (n, &s) in normalized.iter_mut().zip(strategy.iter()) {
        *n = s;
    }

    for h in 0..num_hands {
        let mut denom = 0.0f32;
        for a in 0..num_actions {
            denom += normalized[a * num_hands + h];
        }
        if denom > 0.0 {
            for a in 0..num_actions {
                normalized[a * num_hands + h] /= denom;
            }
        } else {
            let uniform = 1.0 / num_actions as f32;
            for a in 0..num_actions {
                normalized[a * num_hands + h] = uniform;
            }
        }
    }

    normalized
}
