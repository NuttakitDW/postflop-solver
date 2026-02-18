//! Training data generator for the ONNX oracle model.
//!
//! Generates (features, CFV labels) by solving random turn-start poker scenarios.
//!
//! Usage:
//!   cargo run --example generate_training_data --release --features "rayon onnx" -- \
//!     --output-dir ./training_data \
//!     --num-samples 1000 \
//!     --target-exploit 0.5 \
//!     --seed 42

use ndarray::{concatenate, Array2, Array3, ArrayView2, ArrayView3, Axis};
use postflop_solver::oracle::extract_features;
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

struct FlopResult {
    combo_features: Array3<f32>,  // (49, 1326, 19)
    global_features: Array2<f32>, // (49, 20)
    cfv_labels: Array3<f32>,      // (49, 1326, 2)
}

/// Write an ndarray to NPY format (v1.0).
/// NPY is: magic + version + header_len + header_str + raw data.
fn write_npy_3d(path: &str, array: &Array3<f32>) -> std::io::Result<()> {
    let shape = array.shape();
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}), }}",
        shape[0], shape[1], shape[2]
    );
    write_npy_raw(path, &header, array.as_slice().unwrap())
}

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

    // Magic number + version 1.0
    writer.write_all(&[0x93])?;
    writer.write_all(b"NUMPY")?;
    writer.write_all(&[1, 0])?; // version 1.0

    // Header must be padded to multiple of 64 bytes (including magic+version+header_len = 10 bytes)
    let header_bytes = header.as_bytes();
    let padding_needed = 64 - ((10 + header_bytes.len() + 1) % 64); // +1 for trailing newline
    let header_len = (header_bytes.len() + padding_needed + 1) as u16;

    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(header_bytes)?;
    for _ in 0..padding_needed {
        writer.write_all(b" ")?;
    }
    writer.write_all(b"\n")?;

    // Raw f32 data in little-endian (native on x86/ARM)
    let byte_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    writer.write_all(byte_slice)?;
    writer.flush()?;

    Ok(())
}

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

fn sample_flop(rng: &mut impl Rng) -> [Card; 3] {
    let mut cards: Vec<u8> = (0..52).collect();
    cards.shuffle(rng);
    let mut flop = [cards[0], cards[1], cards[2]];
    flop.sort();
    flop
}

/// Sparse random range: zero out 30-70% of combos, rest ~ U[0.01, 1.0].
fn sample_sparse_range(rng: &mut impl Rng) -> Range {
    let mut data = vec![0.0f32; NUM_COMBOS];
    let fold_frac: f64 = rng.gen_range(0.3..=0.7);

    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            if rng.gen::<f64>() >= fold_frac {
                let idx = card_pair_to_index(c1, c2);
                data[idx] = rng.gen_range(0.01f32..=1.0);
            }
        }
    }

    Range::from_raw_data(&data).unwrap()
}

const MAX_ITERATIONS: u32 = 1000;

fn process_flop(
    flop: [Card; 3],
    oop_range: &Range,
    ip_range: &Range,
    pot: i32,
    stack: i32,
    target_exploit: f32,
    turn_bet_sizes: &[BetSizeOptions; 2],
    river_bet_sizes: &[BetSizeOptions; 2],
    turn_donk: &DonkSizeOptions,
    river_donk: &DonkSizeOptions,
) -> FlopResult {
    let turn_cards: Vec<Card> = (0..52u8).filter(|c| !flop.contains(c)).collect();
    let num_turns = turn_cards.len(); // always 49

    let mut cfv_all = Array3::<f32>::zeros((num_turns, NUM_COMBOS, 2));
    let mut last_game: Option<PostFlopGame> = None;

    for (t_idx, &turn_card) in turn_cards.iter().enumerate() {
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
                eprintln!(
                    "Warning: ActionTree failed for flop {:?} turn {}: {}",
                    flop, turn_card, e
                );
                continue;
            }
        };

        let mut game = match PostFlopGame::with_config(card_config, action_tree) {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "Warning: PostFlopGame failed for flop {:?} turn {}: {}",
                    flop, turn_card, e
                );
                continue;
            }
        };

        game.allocate_memory(false);
        let target = target_exploit / 100.0 * pot as f32;
        solve(&mut game, MAX_ITERATIONS, target, false);

        // Extract CFVs for both players
        for player in 0..2 {
            let num_hands = game.num_private_hands(player);
            let cfreach = game.initial_weights(player ^ 1).to_vec();
            let mut result = vec![MaybeUninit::<f32>::uninit(); num_hands];
            {
                let mut root = game.root();
                compute_cfvalue_recursive(
                    &mut result,
                    &game,
                    &mut root,
                    player,
                    &cfreach,
                    false,
                );
            }

            // Map per-hand CFVs to 1326-element combo array, pot-normalized
            let pot_f = pot as f32;
            for (hand_idx, &(c1, c2)) in game.private_cards(player).iter().enumerate() {
                let combo_idx = card_pair_to_index(c1, c2);
                let cfv = unsafe { result[hand_idx].assume_init() };
                cfv_all[[t_idx, combo_idx, player]] = cfv / pot_f;
            }
        }

        last_game = Some(game);
    }

    // Build 1326-element reach arrays from raw Range weights
    let mut reach_oop = vec![0.0f32; NUM_COMBOS];
    let mut reach_ip = vec![0.0f32; NUM_COMBOS];
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            let idx = card_pair_to_index(c1, c2);
            reach_oop[idx] = oop_range.get_weight_by_cards(c1, c2);
            reach_ip[idx] = ip_range.get_weight_by_cards(c1, c2);
        }
    }

    // Extract features using the last solved game (for flop/pot/stack reference)
    let game_ref = last_game.as_ref().expect("No games were solved");
    let (combo_features, global_features) =
        extract_features(game_ref, &reach_oop, &reach_ip, &turn_cards);

    FlopResult {
        combo_features,
        global_features,
        cfv_labels: cfv_all,
    }
}

fn parse_args() -> (String, usize, f32, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut output_dir = String::from("./training_data");
    let mut num_samples: usize = 100;
    let mut target_exploit: f32 = 0.5; // % of pot
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
                eprintln!("Usage: generate_training_data [OPTIONS]");
                eprintln!(
                    "  --output-dir <DIR>       Output directory (default: ./training_data)"
                );
                eprintln!("  --num-samples <N>          Number of scenarios (each = 1 flop × 49 turn cards) (default: 100)");
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

fn main() {
    let (output_dir, num_samples, target_exploit, seed) = parse_args();

    eprintln!("=== Training Data Generator ===");
    eprintln!("Output dir:       {}", output_dir);
    eprintln!("Num samples:        {}", num_samples);
    eprintln!("Target exploit:   {}% of pot", target_exploit);
    eprintln!("Seed:             {}", seed);
    eprintln!();

    // Build bet sizes (shared across all flops)
    let (turn_bet_sizes, river_bet_sizes) = build_bet_sizes();
    let (turn_donk, river_donk) = build_donk_sizes();

    let start = Instant::now();
    let completed = AtomicUsize::new(0);

    // Process flops in parallel
    let results: Vec<FlopResult> = (0..num_samples)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed + i as u64);
            let flop = sample_flop(&mut rng);
            let oop_range = sample_sparse_range(&mut rng);
            let ip_range = sample_sparse_range(&mut rng);
            // Pot: 5-60bb, Stack: 10-100bb (BB=100 for integer precision)
            let bb = 100;
            let pot = rng.gen_range(5i32..=60) * bb;
            let stack = rng.gen_range(10i32..=100) * bb;

            let result = process_flop(
                flop,
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

            let n = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 10 == 0 || n == num_samples {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!(
                    "[{}/{}] {:.1}s elapsed, {:.2} samples/s",
                    n,
                    num_samples,
                    elapsed,
                    n as f64 / elapsed
                );
            }

            result
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!(
        "Solving complete: {} flops in {:.1}s ({:.2} samples/s)",
        num_samples,
        elapsed,
        num_samples as f64 / elapsed
    );

    // Concatenate results
    eprintln!("Concatenating arrays...");

    let combo_views: Vec<ArrayView3<f32>> =
        results.iter().map(|r| r.combo_features.view()).collect();
    let global_views: Vec<ArrayView2<f32>> =
        results.iter().map(|r| r.global_features.view()).collect();
    let cfv_views: Vec<ArrayView3<f32>> = results.iter().map(|r| r.cfv_labels.view()).collect();

    let combo_all = concatenate(Axis(0), &combo_views).unwrap();
    let global_all = concatenate(Axis(0), &global_views).unwrap();
    let cfv_all = concatenate(Axis(0), &cfv_views).unwrap();

    eprintln!(
        "Shapes: combo_features={:?}, global_features={:?}, cfv_labels={:?}",
        combo_all.shape(),
        global_all.shape(),
        cfv_all.shape()
    );

    // Write NPY files
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");

    let write_start = Instant::now();

    let combo_path = format!("{}/combo_features.npy", output_dir);
    let global_path = format!("{}/global_features.npy", output_dir);
    let cfv_path = format!("{}/cfv_labels.npy", output_dir);

    write_npy_3d(&combo_path, &combo_all).expect("Failed to write combo_features.npy");
    eprintln!("Wrote {}", combo_path);

    write_npy_2d(&global_path, &global_all).expect("Failed to write global_features.npy");
    eprintln!("Wrote {}", global_path);

    write_npy_3d(&cfv_path, &cfv_all).expect("Failed to write cfv_labels.npy");
    eprintln!("Wrote {}", cfv_path);

    eprintln!(
        "Write complete in {:.1}s",
        write_start.elapsed().as_secs_f64()
    );
    eprintln!("Done!");
}
