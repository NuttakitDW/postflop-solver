//! Diagnostic: verify boundary CFVs from .dpairs2 match fresh computation.
//!
//! Usage:
//!   cargo run --example debug_pairs_v2 --release --features "bincode rayon" -- config/9s6d6c.json

#[path = "common/mod.rs"]
mod common;

use common::*;
use postflop_solver::*;
use std::env;
use std::io::{self, Read as _};
use std::mem::MaybeUninit;
use std::path::Path;

// =============================================================================
// Same boundary CFV collection as build_pairs_v2
// =============================================================================

fn apply_swap(slice: &mut [f32], swaps: &[(u16, u16)]) {
    for &(i, j) in swaps {
        slice.swap(i as usize, j as usize);
    }
}

fn regret_matching_readonly(regrets: &[f32], num_actions: usize, num_hands: usize) -> Vec<f32> {
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

fn collect_boundary_cfvs_recursive(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    boundaries: &mut Vec<Vec<f32>>,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        collect_boundary_cfvs_recursive(result, game, &mut child, player, cfreach, boundaries);
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
            collect_boundary_cfvs_recursive(
                action_result, game, &mut child, player, &cfreach_scaled, boundaries,
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

        if node.turn() == NOT_DEALT {
            let cfv: Vec<f32> = result.iter()
                .map(|v| unsafe { v.assume_init() })
                .collect();
            boundaries.push(cfv);
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
            collect_boundary_cfvs_recursive(
                action_result, game, &mut child, player, cfreach, boundaries,
            );
        }

        let cfv_actions = cfv_actions.lock();
        let strategy = regret_matching_readonly(node.regrets(), num_actions, num_hands);
        for h in 0..num_hands {
            let mut weighted = 0.0f32;
            for a in 0..num_actions {
                weighted += strategy[a * num_hands + h] * cfv_actions[a * num_hands + h];
            }
            result[h].write(weighted);
        }
    } else {
        let opp_strategy = regret_matching_readonly(node.regrets(), num_actions, cfreach.len());
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
            collect_boundary_cfvs_recursive(
                action_result, game, &mut child, player,
                &cfreach_actions[action * row_size..(action + 1) * row_size],
                boundaries,
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

fn collect_boundary_cfvs(game: &PostFlopGame, player: usize) -> Vec<Vec<f32>> {
    let num_hands = game.num_private_hands(player);
    let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
    let mut root = game.root();
    let mut boundaries = Vec::new();
    collect_boundary_cfvs_recursive(
        &mut result, game, &mut root, player,
        game.initial_weights(player ^ 1),
        &mut boundaries,
    );
    boundaries
}

// =============================================================================
// Same replay DCFR as solve_with_pairs_v2
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

        let alpha_t = (pow_alpha / (pow_alpha + 1.0)) as f32;
        let beta_t = 0.5;
        let gamma_t = pow_gamma as f32;

        Self { alpha_t, beta_t, gamma_t }
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

// Load .dpairs2
struct BoundaryPairsV2 {
    num_hands: [usize; 2],
    num_boundaries: usize,
    _starting_pot: f32,
    iterations: Vec<IterationData>,
}

struct IterationData {
    _iteration: u32,
    _exploitability: f32,
    boundary_cfvs: Vec<Vec<f32>>,
}

impl BoundaryPairsV2 {
    fn load(path: &str) -> io::Result<Self> {
        let mut f = io::BufReader::new(std::fs::File::open(path)?);
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"DPAIRS2\0" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let _version = u32::from_le_bytes(buf4);
        f.read_exact(&mut buf4)?;
        let num_oop = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_ip = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_boundaries = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let num_iterations = u32::from_le_bytes(buf4) as usize;
        f.read_exact(&mut buf4)?;
        let starting_pot = f32::from_le_bytes(buf4);
        let num_hands = [num_oop, num_ip];
        let mut iterations = Vec::with_capacity(num_iterations);
        for _ in 0..num_iterations {
            f.read_exact(&mut buf4)?;
            let iteration = u32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let exploitability = f32::from_le_bytes(buf4);
            f.read_exact(&mut buf4)?;
            let _reserved = u32::from_le_bytes(buf4);
            let mut boundary_cfvs = Vec::with_capacity(num_boundaries * 2);
            for _b in 0..num_boundaries {
                for player in 0..2 {
                    let n_player = num_hands[player];
                    let mut cfv = vec![0.0f32; n_player];
                    let cfv_bytes: &mut [u8] = unsafe {
                        std::slice::from_raw_parts_mut(cfv.as_mut_ptr() as *mut u8, n_player * 4)
                    };
                    f.read_exact(cfv_bytes)?;
                    boundary_cfvs.push(cfv);
                }
            }
            iterations.push(IterationData {
                _iteration: iteration,
                _exploitability: exploitability,
                boundary_cfvs,
            });
        }
        Ok(Self { num_hands, num_boundaries, _starting_pot: starting_pot, iterations })
    }

    fn get_cfv(&self, iteration: usize, boundary_idx: usize, player: usize) -> &[f32] {
        &self.iterations[iteration].boundary_cfvs[boundary_idx * 2 + player]
    }
}

fn solve_flop_with_pairs(
    result: &mut [MaybeUninit<f32>],
    game: &PostFlopGame,
    pairs: &BoundaryPairsV2,
    iteration: usize,
    node: &mut PostFlopNode,
    player: usize,
    cfreach: &[f32],
    params: &DiscountParams,
    boundary_counter: &mut usize,
) {
    if node.is_terminal() {
        game.evaluate(result, node, player, cfreach);
        return;
    }

    let num_actions = node.num_actions();
    let num_hands = result.len();

    if node.is_chance() && node.turn() == NOT_DEALT {
        let idx = *boundary_counter;
        *boundary_counter += 1;
        let cfv = pairs.get_cfv(iteration, idx, player);
        for (r, &v) in result.iter_mut().zip(cfv) {
            r.write(v);
        }
        return;
    }

    if num_actions == 1 && !node.is_chance() {
        let mut child = node.play(0);
        solve_flop_with_pairs(
            result, game, pairs, iteration, &mut child, player, cfreach, params, boundary_counter,
        );
        return;
    }

    let cfv_actions = MutexLike::new(vec![0.0f32; num_actions * num_hands]);

    if node.player() == player {
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
            solve_flop_with_pairs(
                action_result, game, pairs, iteration, &mut child, player, cfreach, params,
                boundary_counter,
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
            solve_flop_with_pairs(
                action_result, game, pairs, iteration, &mut child, player,
                &cfreach_actions[action * row_size..(action + 1) * row_size],
                params, boundary_counter,
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
// Helpers
// =============================================================================

fn compare_cfvs(
    pairs: &BoundaryPairsV2, t: usize, player: usize, computed: &[Vec<f32>],
) -> (f32, f64, usize, usize) {
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut count = 0usize;
    let mut worst_b = 0;
    let mut worst_h = 0;
    for b in 0..pairs.num_boundaries {
        let recorded = pairs.get_cfv(t, b, player);
        let comp = &computed[b];
        for h in 0..recorded.len() {
            let d = (recorded[h] - comp[h]).abs();
            sum_diff += d as f64;
            count += 1;
            if d > max_diff { max_diff = d; worst_b = b; worst_h = h; }
        }
    }
    (max_diff, sum_diff / count.max(1) as f64, worst_b, worst_h)
}

fn compare_root_regrets(
    game_build: &PostFlopGame, game_replay: &PostFlopGame,
) -> (f32, f64) {
    let rb = game_build.root();
    let rr = game_replay.root();
    let regb = rb.regrets();
    let regr = rr.regrets();
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    for (a, b) in regb.iter().zip(regr.iter()) {
        let d = (a - b).abs();
        sum_diff += d as f64;
        if d > max_diff { max_diff = d; }
    }
    (max_diff, sum_diff / regb.len().max(1) as f64)
}

// =============================================================================
// Main diagnostic
// =============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config.json>", args[0]);
        std::process::exit(1);
    }
    let config_path = &args[1];
    let config = load_config(config_path);
    let (card_config, tree_config) = parse_configs(&config).unwrap();

    let config_filename = Path::new(config_path).file_stem().unwrap().to_str().unwrap();
    let pairs_path = format!("data/oracles/{}.dpairs2", config_filename);

    println!("=== Diagnostic: Verify boundary CFVs ===");
    println!("Config: {}", config_path);
    println!();

    // Load pairs
    println!("Loading pairs: {}", pairs_path);
    let pairs = BoundaryPairsV2::load(&pairs_path).expect("Failed to load pairs");
    println!("  {} boundaries, {} iterations", pairs.num_boundaries, pairs.iterations.len());
    println!();

    // Build game
    println!("Building game...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    game.allocate_memory(false);
    println!("  OOP={}, IP={}", game.num_private_hands(0), game.num_private_hands(1));
    println!();

    // Also build a second game for the replay path
    let action_tree2 = ActionTree::new(tree_config.clone()).unwrap();
    let mut game_replay = PostFlopGame::with_config(card_config.clone(), action_tree2).unwrap();
    game_replay.allocate_memory(false);

    let max_iters = pairs.iterations.len().min(5); // Check first 5 iterations
    println!("Checking first {} iterations...", max_iters);
    println!("(Matching build flow: P0 CFVs → P0 solve → P1 CFVs → P1 solve)");
    println!();

    for t in 0..max_iters {
        let params = DiscountParams::new(t as u32);

        // === Player 0 ===
        // Step 1: Check P0 boundary CFVs (build game is in correct pre-P0 state)
        {
            let computed_cfvs = collect_boundary_cfvs(&game, 0);
            let (max_diff, avg_diff, worst_b, worst_h) =
                compare_cfvs(&pairs, t, 0, &computed_cfvs);
            if max_diff > 0.001 {
                println!("  iter={} P0: MISMATCH! max={:.6} avg={:.6} (b={} h={})",
                    t, max_diff, avg_diff, worst_b, worst_h);
            } else {
                println!("  iter={} P0: boundary CFVs OK (max={:.8})", t, max_diff);
            }
        }

        // Step 2: Run replay for P0 (update replay P0 flop regrets)
        {
            let num_hands = game_replay.num_private_hands(0);
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            let mut root = game_replay.root();
            let mut bc = 0usize;
            solve_flop_with_pairs(
                &mut result, &game_replay, &pairs, t, &mut root, 0,
                game_replay.initial_weights(1), &params, &mut bc,
            );
        }

        // Step 3: Run library solve for P0 on build game (update all P0 regrets)
        solve_step_for_player(&game, t as u32, 0);

        // Compare root regrets after P0 updates
        {
            let (max_diff, _) = compare_root_regrets(&game, &game_replay);
            if max_diff > 0.0001 {
                println!("    ROOT REGRET diff after P0 solve: max={:.6}", max_diff);
            } else {
                println!("    ROOT REGRET after P0 solve: OK (max={:.8})", max_diff);
            }
        }

        // === Player 1 ===
        // Step 4: Check P1 boundary CFVs (build game now has P0 updates!)
        {
            let computed_cfvs = collect_boundary_cfvs(&game, 1);
            let (max_diff, avg_diff, worst_b, worst_h) =
                compare_cfvs(&pairs, t, 1, &computed_cfvs);
            if max_diff > 0.001 {
                println!("  iter={} P1: MISMATCH! max={:.6} avg={:.6} (b={} h={})",
                    t, max_diff, avg_diff, worst_b, worst_h);
                let recorded = pairs.get_cfv(t, worst_b, 1);
                let computed = &computed_cfvs[worst_b];
                let s = worst_h;
                let e = (worst_h + 5).min(recorded.len());
                println!("    recorded[{}..{}]: {:?}", s, e, &recorded[s..e]);
                println!("    computed[{}..{}]: {:?}", s, e, &computed[s..e]);
            } else {
                println!("  iter={} P1: boundary CFVs OK (max={:.8})", t, max_diff);
            }
        }

        // Step 5: Run replay for P1 (update replay P1 flop regrets)
        {
            let num_hands = game_replay.num_private_hands(1);
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            let mut root = game_replay.root();
            let mut bc = 0usize;
            solve_flop_with_pairs(
                &mut result, &game_replay, &pairs, t, &mut root, 1,
                game_replay.initial_weights(0), &params, &mut bc,
            );
        }

        // Step 6: Run library solve for P1 on build game
        solve_step_for_player(&game, t as u32, 1);

        // Compare root regrets after both players
        {
            let (max_diff, _) = compare_root_regrets(&game, &game_replay);
            if max_diff > 0.0001 {
                println!("    ROOT REGRET diff after P1 solve: max={:.6}", max_diff);
                let rb = game.root();
                let rr = game_replay.root();
                let regb = rb.regrets();
                let regr = rr.regrets();
                println!("    build  [0..5]: {:?}", &regb[..5.min(regb.len())]);
                println!("    replay [0..5]: {:?}", &regr[..5.min(regr.len())]);
            } else {
                println!("    ROOT REGRET after full iter: OK (max={:.8})", max_diff);
            }
        }
        println!();
    }

    // After N iterations, compare root strategies
    println!();
    println!("=== Root strategy comparison after {} iterations ===", max_iters);
    {
        let root_build = game.root();
        let root_replay = game_replay.root();
        let strat_build = root_build.strategy();
        let strat_replay = root_replay.strategy();
        let na = root_build.num_actions();
        let nh = game.num_private_hands(0);

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (a, b) in strat_build.iter().zip(strat_replay.iter()) {
            let d = (a - b).abs();
            sum_diff += d as f64;
            if d > max_diff { max_diff = d; }
        }
        println!("  Root cumulative strategy: max_diff={:.6} avg_diff={:.6} (na={} nh={})",
            max_diff, sum_diff / strat_build.len() as f64, na, nh);
    }
}
