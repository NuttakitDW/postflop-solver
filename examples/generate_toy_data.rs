//! Toy game data generator for the Turn value network.
//!
//! Generates training data for a SINGLE flop (AdKc2d) from config/20bb.json.
//! Enumerates all 49 turn cards, generating multiple samples per turn card
//! with different random reach distributions and pot/stack (DeepStack-style).
//!
//! Game parameters:
//!   Flop: AdKc2d (fixed) | Pot/Stack: randomized (SPR capped at 5)
//!   Bet sizes: pot-bet + all-in (DeepStack style)
//!
//! Output files:
//!   meta.npy    — [N, 6]    (board[0..3] as card indices, pot, stack)
//!   ranges.npy  — [N, 2652] (reach_oop[1326] ++ reach_ip[1326])
//!   values.npy  — [N, 2652] (cfv_oop[1326] ++ cfv_ip[1326], pot-normalized)
//!
//! Usage:
//!   cargo run --example generate_toy_data --release --features "rayon" -- \
//!     --output-dir ./data/toy_20bb \
//!     --samples-per-turn 100 \
//!     --target-exploit 0.5 \
//!     --seed 42

use ndarray::{concatenate, Array1, Array2, ArrayView2, Axis};
use postflop_solver::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const NUM_COMBOS: usize = 1326;
const META_DIM: usize = 6;
const RANGE_DIM: usize = 2 * NUM_COMBOS; // 2652
const VALUE_DIM: usize = 2 * NUM_COMBOS; // 2652
const MAX_ITERATIONS: u32 = 1000;

// Fixed game parameters from config/20bb.json
// Card encoding: rank * 4 + suit (rank: 2=0..A=12, suit: c=0,d=1,h=2,s=3)
const FLOP: [Card; 3] = [
    1,  // 2d (rank=0, suit=1)
    44, // Kc (rank=11, suit=0)
    49, // Ad (rank=12, suit=1)
];
const MAX_SPR: f32 = 5.0;

const RANK_CHARS: [char; 13] = [
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];
const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

fn card_to_str(c: Card) -> String {
    format!(
        "{}{}",
        RANK_CHARS[(c >> 2) as usize],
        SUIT_CHARS[(c & 3) as usize]
    )
}

fn board_to_str(cards: &[Card]) -> String {
    cards
        .iter()
        .map(|&c| card_to_str(c))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// NPY helpers
// ---------------------------------------------------------------------------

fn write_npy_2d(path: &str, array: &Array2<f32>) -> std::io::Result<()> {
    let shape = array.shape();
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        shape[0], shape[1]
    );
    write_npy_raw(path, &header, array.as_slice().unwrap())
}

fn write_npy_raw(path: &str, header: &str, data: &[f32]) -> std::io::Result<()> {
    let mut writer = BufWriter::new(fs::File::create(path)?);

    writer.write_all(&[0x93])?;
    writer.write_all(b"NUMPY")?;
    writer.write_all(&[1, 0])?;

    let header_bytes = header.as_bytes();
    let padding_needed = 64 - ((10 + header_bytes.len() + 1) % 64);
    let header_len = (header_bytes.len() + padding_needed + 1) as u16;

    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(header_bytes)?;
    for _ in 0..padding_needed {
        writer.write_all(b" ")?;
    }
    writer.write_all(b"\n")?;

    let byte_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    writer.write_all(byte_slice)?;
    writer.flush()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Bet / donk sizes (DeepStack: pot-bet + all-in)
// ---------------------------------------------------------------------------

fn build_bet_sizes() -> ([BetSizeOptions; 2], [BetSizeOptions; 2]) {
    let oop_turn: BetSizeOptions = ("100%,a", "a").try_into().unwrap();
    let ip_turn: BetSizeOptions = ("100%,a", "a").try_into().unwrap();
    let oop_river: BetSizeOptions = ("100%,a", "a").try_into().unwrap();
    let ip_river: BetSizeOptions = ("100%,a", "a").try_into().unwrap();
    ([oop_turn, ip_turn], [oop_river, ip_river])
}

// ---------------------------------------------------------------------------
// Enumerate all 49 turn cards for the fixed flop
// ---------------------------------------------------------------------------

fn get_turn_cards() -> Vec<Card> {
    let flop_set: std::collections::HashSet<Card> = FLOP.iter().copied().collect();
    (0u8..52).filter(|c| !flop_set.contains(c)).collect()
}

// ---------------------------------------------------------------------------
// DeepStack recursive range generation
// ---------------------------------------------------------------------------

fn generate_deepstack_range(
    board: &[Card; 4],
    equity: &[f32; 1326],
    rng: &mut impl Rng,
) -> [f32; NUM_COMBOS] {
    let board_mask: u64 = board.iter().fold(0u64, |acc, &c| acc | (1u64 << c));

    let mut valid: Vec<(usize, f32)> = Vec::with_capacity(1128);
    for combo_idx in 0..NUM_COMBOS {
        let (c1, c2) = index_to_card_pair(combo_idx);
        let hand_mask: u64 = (1u64 << c1) | (1u64 << c2);
        if hand_mask & board_mask != 0 {
            continue;
        }
        valid.push((combo_idx, equity[combo_idx]));
    }
    valid.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut reach = [0.0f32; NUM_COMBOS];
    recursive_split(&valid, 1.0, &mut reach, rng);
    reach
}

fn recursive_split(
    hands: &[(usize, f32)],
    mass: f32,
    reach: &mut [f32; NUM_COMBOS],
    rng: &mut impl Rng,
) {
    if hands.len() <= 1 {
        if let Some(&(idx, _)) = hands.first() {
            reach[idx] = mass;
        }
        return;
    }
    let mid = hands.len() / 2;
    let split: f32 = rng.gen();
    recursive_split(&hands[..mid], mass * split, reach, rng);
    recursive_split(&hands[mid..], mass * (1.0 - split), reach, rng);
}

// ---------------------------------------------------------------------------
// Per-sample processing
// ---------------------------------------------------------------------------

struct RawSampleResult {
    meta: Array1<f32>,   // (6,): board[0..4], pot, stack
    ranges: Array1<f32>, // (2652,): reach_oop ++ reach_ip
    values: Array1<f32>, // (2652,): cfv_oop ++ cfv_ip (pot-normalized)
}

fn process_sample(
    sample_idx: usize,
    num_samples: usize,
    turn_card: Card,
    reach_oop: &[f32; NUM_COMBOS],
    reach_ip: &[f32; NUM_COMBOS],
    oop_range: &Range,
    ip_range: &Range,
    pot: i32,
    stack: i32,
    target_exploit: f32,
    turn_bet_sizes: &[BetSizeOptions; 2],
    river_bet_sizes: &[BetSizeOptions; 2],
) -> Option<RawSampleResult> {
    let sample_start = Instant::now();
    let board = [FLOP[0], FLOP[1], FLOP[2], turn_card];

    let oop_active = reach_oop.iter().filter(|&&r| r > 0.0).count();
    let ip_active = reach_ip.iter().filter(|&&r| r > 0.0).count();

    eprintln!(
        "[{}/{}] board {} | pot={} stack={} SPR={:.1} | OOP {} combos, IP {} combos",
        sample_idx + 1,
        num_samples,
        board_to_str(&board),
        pot,
        stack,
        stack as f64 / pot as f64,
        oop_active,
        ip_active,
    );

    let card_config = CardConfig {
        range: [oop_range.clone(), ip_range.clone()],
        flop: FLOP,
        turn: turn_card,
        river: NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot,
        effective_stack: stack,
        turn_bet_sizes: turn_bet_sizes.clone(),
        river_bet_sizes: river_bet_sizes.clone(),
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 6.0,
        force_allin_threshold: 0.5,
        merging_threshold: 0.1,
        ..Default::default()
    };

    let action_tree = match ActionTree::new(tree_config) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  SKIP (ActionTree): {}", e);
            return None;
        }
    };

    let mut game = match PostFlopGame::with_config(card_config, action_tree) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("  SKIP (PostFlopGame): {}", e);
            return None;
        }
    };

    game.allocate_memory(false);
    let target = target_exploit / 100.0 * pot as f32;
    let exploitability = solve(&mut game, MAX_ITERATIONS, target, false);
    let pot_f = pot as f32;

    // Extract 1326-combo CFVs (pot-normalized)
    let mut cfv_oop = [0.0f32; NUM_COMBOS];
    let mut cfv_ip = [0.0f32; NUM_COMBOS];

    for player in 0..2 {
        let num_hands = game.num_private_hands(player);
        let cfreach = game.initial_weights(player ^ 1).to_vec();
        let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
        {
            let mut root = game.root();
            compute_cfvalue_recursive(&mut result, &game, &mut root, player, &cfreach, false);
        }

        let cfv_arr = if player == 0 {
            &mut cfv_oop
        } else {
            &mut cfv_ip
        };
        for (hand_idx, &(c1, c2)) in game.private_cards(player).iter().enumerate() {
            let combo_idx = card_pair_to_index(c1, c2);
            cfv_arr[combo_idx] = unsafe { result[hand_idx].assume_init() } / pot_f;
        }
    }

    // Assemble raw output
    let mut meta = Array1::<f32>::zeros(META_DIM);
    meta[0] = board[0] as f32;
    meta[1] = board[1] as f32;
    meta[2] = board[2] as f32;
    meta[3] = board[3] as f32;
    meta[4] = pot as f32;
    meta[5] = stack as f32;

    let mut ranges = Array1::<f32>::zeros(RANGE_DIM);
    ranges.as_slice_mut().unwrap()[..NUM_COMBOS].copy_from_slice(reach_oop);
    ranges.as_slice_mut().unwrap()[NUM_COMBOS..].copy_from_slice(reach_ip);

    let mut values = Array1::<f32>::zeros(VALUE_DIM);
    values.as_slice_mut().unwrap()[..NUM_COMBOS].copy_from_slice(&cfv_oop);
    values.as_slice_mut().unwrap()[NUM_COMBOS..].copy_from_slice(&cfv_ip);

    eprintln!(
        "  solved {:.2}s | exploit={:.4}",
        sample_start.elapsed().as_secs_f64(),
        exploitability / pot_f,
    );

    Some(RawSampleResult {
        meta,
        ranges,
        values,
    })
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> (String, usize, f32, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut output_dir = String::from("./data/toy_20bb");
    let mut samples_per_turn: usize = 100;
    let mut target_exploit: f32 = 0.5;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output-dir" => {
                output_dir = args[i + 1].clone();
                i += 2;
            }
            "--samples-per-turn" => {
                samples_per_turn = args[i + 1].parse().expect("Invalid --samples-per-turn");
                i += 2;
            }
            "--target-exploit" => {
                target_exploit = args[i + 1].parse().expect("Invalid --target-exploit");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("Invalid --seed");
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!("Usage: generate_toy_data [OPTIONS]");
                eprintln!(
                    "  --output-dir <DIR>           Output directory (default: ./data/toy_20bb)"
                );
                eprintln!(
                    "  --samples-per-turn <N>       Samples per turn card (default: 100)"
                );
                eprintln!(
                    "  --target-exploit <PCT>       Target exploitability in %% of pot (default: 0.5)"
                );
                eprintln!("  --seed <N>                   Random seed (default: 42)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    (output_dir, samples_per_turn, target_exploit, seed)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let (output_dir, samples_per_turn, target_exploit, seed) = parse_args();

    let turn_cards = get_turn_cards();
    let num_turns = turn_cards.len();
    let num_samples = num_turns * samples_per_turn;

    eprintln!("=== Toy Game Data Generator (20bb) ===");
    eprintln!("Flop:             {}", board_to_str(&FLOP));
    eprintln!("Pot/Stack:        randomized (SPR capped at {:.1})", MAX_SPR);
    eprintln!("Turn cards:       {} (52 - 3 flop)", num_turns);
    eprintln!("Samples/turn:     {}", samples_per_turn);
    eprintln!("Total samples:    {}", num_samples);
    eprintln!("Target exploit:   {}% of pot", target_exploit);
    eprintln!("Seed:             {}", seed);
    eprintln!("Output dir:       {}", output_dir);
    eprintln!("Output format:    meta[N,6] + ranges[N,2652] + values[N,2652]");
    eprintln!();

    let (turn_bet_sizes, river_bet_sizes) = build_bet_sizes();

    // Build work items: (global_index, turn_card, sample_within_turn)
    let work_items: Vec<(usize, Card, usize)> = turn_cards
        .iter()
        .enumerate()
        .flat_map(|(turn_idx, &tc)| {
            (0..samples_per_turn).map(move |s| (turn_idx * samples_per_turn + s, tc, s))
        })
        .collect();

    let start = Instant::now();
    let completed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);

    let results: Vec<RawSampleResult> = work_items
        .into_par_iter()
        .filter_map(|(global_idx, turn_card, _sample_idx)| {
            // Unique seed per (turn_card, sample) pair
            let mut rng = StdRng::seed_from_u64(seed + global_idx as u64);

            let board = [FLOP[0], FLOP[1], FLOP[2], turn_card];
            let equity = compute_equity(&board);
            let reach_oop = generate_deepstack_range(&board, &equity, &mut rng);
            let reach_ip = generate_deepstack_range(&board, &equity, &mut rng);

            let oop_range = Range::from_raw_data(&reach_oop).unwrap();
            let ip_range = Range::from_raw_data(&reach_ip).unwrap();

            // Randomize pot/stack (same logic as generate_raw_data.rs)
            let bb = 100;
            let pot = rng.gen_range(5i32..=60) * bb;
            let max_stack = (pot as f32 * MAX_SPR) as i32 / bb;
            let max_stack = max_stack.clamp(10, 100);
            let stack = rng.gen_range(10i32..=max_stack) * bb;

            let result = process_sample(
                global_idx,
                num_samples,
                turn_card,
                &reach_oop,
                &reach_ip,
                &oop_range,
                &ip_range,
                pot,
                stack,
                target_exploit,
                &turn_bet_sizes,
                &river_bet_sizes,
            );

            if result.is_some() {
                completed.fetch_add(1, Ordering::Relaxed);
            } else {
                skipped.fetch_add(1, Ordering::Relaxed);
            }

            let done = completed.load(Ordering::Relaxed) + skipped.load(Ordering::Relaxed);
            if done % 100 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!(
                    "--- progress: {}/{} ({} ok, {} skipped) | {:.1}s | {:.1} samples/s ---",
                    done,
                    num_samples,
                    completed.load(Ordering::Relaxed),
                    skipped.load(Ordering::Relaxed),
                    elapsed,
                    done as f64 / elapsed,
                );
            }

            result
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64();
    let ok_count = results.len();
    let skip_count = skipped.load(Ordering::Relaxed);
    eprintln!();
    eprintln!(
        "Solving complete: {} samples in {:.1}s ({:.1} samples/s), {} skipped",
        ok_count, elapsed, ok_count as f64 / elapsed, skip_count
    );

    if results.is_empty() {
        eprintln!("No samples generated. Exiting.");
        return;
    }

    // Stack into 2-D arrays
    eprintln!("Concatenating arrays...");

    let meta_views: Vec<ArrayView2<f32>> = results
        .iter()
        .map(|r| r.meta.view().insert_axis(Axis(0)))
        .collect();
    let range_views: Vec<ArrayView2<f32>> = results
        .iter()
        .map(|r| r.ranges.view().insert_axis(Axis(0)))
        .collect();
    let value_views: Vec<ArrayView2<f32>> = results
        .iter()
        .map(|r| r.values.view().insert_axis(Axis(0)))
        .collect();

    let meta_all = concatenate(Axis(0), &meta_views).unwrap();
    let ranges_all = concatenate(Axis(0), &range_views).unwrap();
    let values_all = concatenate(Axis(0), &value_views).unwrap();

    eprintln!(
        "Shapes: meta={:?}, ranges={:?}, values={:?}",
        meta_all.shape(),
        ranges_all.shape(),
        values_all.shape()
    );

    // Write NPY files
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    let write_start = Instant::now();

    let meta_path = format!("{}/meta.npy", output_dir);
    let ranges_path = format!("{}/ranges.npy", output_dir);
    let values_path = format!("{}/values.npy", output_dir);

    write_npy_2d(&meta_path, &meta_all).expect("Failed to write meta.npy");
    eprintln!("Wrote {}", meta_path);

    write_npy_2d(&ranges_path, &ranges_all).expect("Failed to write ranges.npy");
    eprintln!("Wrote {}", ranges_path);

    write_npy_2d(&values_path, &values_all).expect("Failed to write values.npy");
    eprintln!("Wrote {}", values_path);

    eprintln!(
        "Write complete in {:.1}s",
        write_start.elapsed().as_secs_f64()
    );
    eprintln!("Total: {} data points across {} turn cards ({} per turn)",
        ok_count, num_turns, samples_per_turn);
    eprintln!("Done!");
}
