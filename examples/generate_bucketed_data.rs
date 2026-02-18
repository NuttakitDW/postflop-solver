//! Bucketed training data generator for the DeepStack-style value network.
//!
//! Each sample = one random 4-card turn board + random ranges + random pot/stack.
//! Ranges are generated via DeepStack's recursive equity-based splitting on the
//! *actual* board being solved, so ranges always match the board.
//!
//! Usage:
//!   cargo run --example generate_bucketed_data --release --features "rayon" -- \
//!     --output-dir ./bucketed_training_data \
//!     --num-samples 10000 \
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
const K: usize = DEFAULT_K; // 1000
const BOARD_FEATURES: usize = 15;
const INPUT_DIM: usize = BOARD_FEATURES + 2 * K; // 2015
const OUTPUT_DIM: usize = 2 * K; // 2000
const MAX_ITERATIONS: u32 = 1000;

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
// Bet / donk sizes (identical to generate_training_data.rs)
// ---------------------------------------------------------------------------

fn build_bet_sizes() -> ([BetSizeOptions; 2], [BetSizeOptions; 2]) {
    let oop_turn: BetSizeOptions =
        ("22%,33%,55%,66%,75%,85%,125%,150%,200%", "33%,50%,75%,100%")
            .try_into()
            .unwrap();
    let ip_turn: BetSizeOptions =
        ("22%,33%,55%,66%,75%,85%,125%,150%,200%", "33%,50%,75%,100%")
            .try_into()
            .unwrap();
    let oop_river: BetSizeOptions =
        ("22%,33%,55%,66%,75%,85%,125%,150%,200%", "33%,50%,75%,100%")
            .try_into()
            .unwrap();
    let ip_river: BetSizeOptions =
        ("22%,33%,55%,66%,75%,85%,125%,150%,200%", "33%,50%,75%,100%")
            .try_into()
            .unwrap();
    ([oop_turn, ip_turn], [oop_river, ip_river])
}

fn build_donk_sizes() -> (DonkSizeOptions, DonkSizeOptions) {
    let turn_donk: DonkSizeOptions = "15%,33%,55%,75%,100%".try_into().unwrap();
    let river_donk: DonkSizeOptions = "15%,33%,55%,75%,100%".try_into().unwrap();
    (turn_donk, river_donk)
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Deal 4 random cards: 3 flop + 1 turn, returned sorted.
fn sample_board(rng: &mut impl Rng) -> ([Card; 3], Card) {
    let mut cards: Vec<u8> = (0..52).collect();
    cards.shuffle(rng);
    let mut flop = [cards[0], cards[1], cards[2]];
    flop.sort();
    (flop, cards[3])
}

// ---------------------------------------------------------------------------
// DeepStack recursive range generation (Page 26)
// ---------------------------------------------------------------------------

/// Generate a random range using DeepStack's recursive equity-based splitting.
///
/// 1. Sort valid (non-blocked) hands by equity on the actual board.
/// 2. Recursively divide hands into halves and randomly split probability mass.
/// 3. Blocked hands stay at 0.0.
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
// Per-sample processing: one board = one data point
// ---------------------------------------------------------------------------

struct SampleResult {
    input: Array1<f32>,  // (2015,)
    target: Array1<f32>, // (2000,)
}

fn process_sample(
    sample_idx: usize,
    num_samples: usize,
    flop: [Card; 3],
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
    turn_donk: &DonkSizeOptions,
    river_donk: &DonkSizeOptions,
) -> Option<SampleResult> {
    let sample_start = Instant::now();
    let board = [flop[0], flop[1], flop[2], turn_card];

    // Range stats
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

    // Bucket mapping for this board
    let bucket_mapping = compute_buckets(&board, K);

    // Build turn-start game
    let card_config = CardConfig {
        range: [oop_range.clone(), ip_range.clone()],
        flop,
        turn: turn_card,
        river: NOT_DEALT,
    };

    let tree_config = TreeConfig {
        initial_state: BoardState::Turn,
        starting_pot: pot,
        effective_stack: stack,
        turn_bet_sizes: turn_bet_sizes.clone(),
        river_bet_sizes: river_bet_sizes.clone(),
        turn_donk_sizes: Some(turn_donk.clone()),
        river_donk_sizes: Some(river_donk.clone()),
        add_allin_threshold: 6.0,
        force_allin_threshold: 0.50,
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

    // Project to K buckets
    let range_oop_bucketed = project_range_to_buckets(reach_oop, &bucket_mapping);
    let range_ip_bucketed = project_range_to_buckets(reach_ip, &bucket_mapping);
    let cfv_oop_bucketed = project_cfv_to_buckets(&cfv_oop, reach_oop, &bucket_mapping);
    let cfv_ip_bucketed = project_cfv_to_buckets(&cfv_ip, reach_ip, &bucket_mapping);

    // Board features
    let board_features = compute_board_features(&board, pot_f, stack as f32);

    // Assemble input [board(15) | range_oop(K) | range_ip(K)]
    let mut input = Array1::<f32>::zeros(INPUT_DIM);
    input.as_slice_mut().unwrap()[..BOARD_FEATURES].copy_from_slice(&board_features);
    input.as_slice_mut().unwrap()[BOARD_FEATURES..BOARD_FEATURES + K]
        .copy_from_slice(&range_oop_bucketed);
    input.as_slice_mut().unwrap()[BOARD_FEATURES + K..].copy_from_slice(&range_ip_bucketed);

    // Assemble target [cfv_oop(K) | cfv_ip(K)]
    let mut target_arr = Array1::<f32>::zeros(OUTPUT_DIM);
    target_arr.as_slice_mut().unwrap()[..K].copy_from_slice(&cfv_oop_bucketed);
    target_arr.as_slice_mut().unwrap()[K..].copy_from_slice(&cfv_ip_bucketed);

    // Bucket stats for logging
    let oop_nonzero = range_oop_bucketed.iter().filter(|&&x| x > 0.0).count();
    let ip_nonzero = range_ip_bucketed.iter().filter(|&&x| x > 0.0).count();
    let cfv_oop_mean: f64 =
        cfv_oop_bucketed.iter().map(|&x| x as f64).sum::<f64>() / K as f64;
    let cfv_ip_mean: f64 =
        cfv_ip_bucketed.iter().map(|&x| x as f64).sum::<f64>() / K as f64;

    eprintln!(
        "  solved {:.2}s | exploit={:.4} | buckets OOP={} IP={} | mean_cfv OOP={:.4} IP={:.4}",
        sample_start.elapsed().as_secs_f64(),
        exploitability / pot_f,
        oop_nonzero,
        ip_nonzero,
        cfv_oop_mean,
        cfv_ip_mean,
    );

    Some(SampleResult {
        input,
        target: target_arr,
    })
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> (String, usize, f32, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut output_dir = String::from("./bucketed_training_data");
    let mut num_samples: usize = 10000;
    let mut target_exploit: f32 = 0.5;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output-dir" => {
                output_dir = args[i + 1].clone();
                i += 2;
            }
            "--num-samples" => {
                num_samples = args[i + 1].parse().expect("Invalid --num-samples");
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
                eprintln!("Usage: generate_bucketed_data [OPTIONS]");
                eprintln!(
                    "  --output-dir <DIR>       Output directory (default: ./bucketed_training_data)"
                );
                eprintln!(
                    "  --num-samples <N>        Number of samples (1 board = 1 sample) (default: 10000)"
                );
                eprintln!(
                    "  --target-exploit <PCT>   Target exploitability in % of pot (default: 0.5)"
                );
                eprintln!("  --seed <N>               Random seed (default: 42)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    (output_dir, num_samples, target_exploit, seed)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let (output_dir, num_samples, target_exploit, seed) = parse_args();

    eprintln!("=== Bucketed Training Data Generator ===");
    eprintln!("Output dir:       {}", output_dir);
    eprintln!("Num samples:      {}", num_samples);
    eprintln!("K (buckets):      {}", K);
    eprintln!("Target exploit:   {}% of pot", target_exploit);
    eprintln!("Seed:             {}", seed);
    eprintln!(
        "Input dim:        {} (board {} + ranges {}x2)",
        INPUT_DIM, BOARD_FEATURES, K
    );
    eprintln!("Output dim:       {} (CFVs {}x2)", OUTPUT_DIM, K);
    eprintln!();

    let (turn_bet_sizes, river_bet_sizes) = build_bet_sizes();
    let (turn_donk, river_donk) = build_donk_sizes();

    let start = Instant::now();
    let completed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);

    let results: Vec<SampleResult> = (0..num_samples)
        .into_par_iter()
        .filter_map(|i| {
            let mut rng = StdRng::seed_from_u64(seed + i as u64);

            // Deal 4 random cards (the actual board being solved)
            let (flop, turn_card) = sample_board(&mut rng);
            let board = [flop[0], flop[1], flop[2], turn_card];

            // Equity on THIS board → ranges match the board
            let equity = compute_equity(&board);
            let reach_oop = generate_deepstack_range(&board, &equity, &mut rng);
            let reach_ip = generate_deepstack_range(&board, &equity, &mut rng);

            let oop_range = Range::from_raw_data(&reach_oop).unwrap();
            let ip_range = Range::from_raw_data(&reach_ip).unwrap();

            // Random pot/stack
            let bb = 100;
            let pot = rng.gen_range(5i32..=60) * bb;
            let stack = rng.gen_range(10i32..=100) * bb;

            let result = process_sample(
                i,
                num_samples,
                flop,
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
                &turn_donk,
                &river_donk,
            );

            if result.is_some() {
                completed.fetch_add(1, Ordering::Relaxed);
            } else {
                skipped.fetch_add(1, Ordering::Relaxed);
            }

            // Periodic summary
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

    // Stack 1-D rows into 2-D arrays
    eprintln!("Concatenating arrays...");

    let input_views: Vec<ArrayView2<f32>> = results
        .iter()
        .map(|r| r.input.view().insert_axis(Axis(0)))
        .collect();
    let target_views: Vec<ArrayView2<f32>> = results
        .iter()
        .map(|r| r.target.view().insert_axis(Axis(0)))
        .collect();

    let inputs_all = concatenate(Axis(0), &input_views).unwrap();
    let targets_all = concatenate(Axis(0), &target_views).unwrap();

    eprintln!(
        "Shapes: inputs={:?}, targets={:?}",
        inputs_all.shape(),
        targets_all.shape()
    );

    // Write NPY files
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    let write_start = Instant::now();

    let inputs_path = format!("{}/inputs.npy", output_dir);
    let targets_path = format!("{}/targets.npy", output_dir);

    write_npy_2d(&inputs_path, &inputs_all).expect("Failed to write inputs.npy");
    eprintln!("Wrote {}", inputs_path);

    write_npy_2d(&targets_path, &targets_all).expect("Failed to write targets.npy");
    eprintln!("Wrote {}", targets_path);

    eprintln!(
        "Write complete in {:.1}s",
        write_start.elapsed().as_secs_f64()
    );
    eprintln!("Total: {} data points", ok_count);
    eprintln!("Done!");
}
