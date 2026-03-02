//! Phase 2 v2: Solve a flop game using a bt2 ONNX model with universal 1326-dim hand space.
//!
//! Uses the full 1326 card-pair index space (C(52,2)) so that position i
//! always maps to the same two-card combo regardless of player.
//! Hands not in range are 0.
//!
//! Model input:  [pot_norm, stack_norm, player, cfreach(1326)]
//! Model output: [cfv(1326)]
//!
//! At each boundary:
//!   1. Expand solver-indexed cfreach → 1326-dim (using opponent's card_pair_indices)
//!   2. Run model → 1326-dim CFV
//!   3. Extract player's CFVs back to solver indexing (using player's card_pair_indices)
//!
//! Usage:
//!   ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
//!   cargo run --example solve_with_model_v2 --release --features "bincode rayon" -- \
//!     config/KcQh7s.json models/bt2_KcQh7s

#[path = "../common/mod.rs"]
mod common;

use common::*;
use ort::session::Session;
use postflop_solver::*;
use std::env;
use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

const FULL_HANDS: usize = 1326; // C(52, 2)

// =============================================================================
// Model metadata
// =============================================================================

#[derive(Debug)]
struct ModelMeta {
    total_iterations: usize,
    y_scale: f32,
    in_dim: usize,
    out_dim: usize,
    max_pot: f32,
    num_oop: usize,
    num_ip: usize,
}

impl ModelMeta {
    fn load(path: &str) -> Self {
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("Failed to read meta"))
                .expect("Failed to parse meta JSON");

        Self {
            total_iterations: data["total_iterations"].as_u64().unwrap() as usize,
            y_scale: data["y_scale"].as_f64().unwrap() as f32,
            in_dim: data["in_dim"].as_u64().unwrap() as usize,
            out_dim: data["out_dim"].as_u64().unwrap() as usize,
            max_pot: data["max_pot"].as_f64().unwrap() as f32,
            num_oop: data["num_oop"].as_u64().unwrap() as usize,
            num_ip: data["num_ip"].as_u64().unwrap() as usize,
        }
    }
}

// =============================================================================
// Card pair index (same formula as solver's card_pair_to_index)
// =============================================================================

/// Map two card IDs to the universal 0..1325 index.
fn card_pair_to_index(card1: u8, card2: u8) -> usize {
    let (c1, c2) = if card1 <= card2 {
        (card1 as usize, card2 as usize)
    } else {
        (card2 as usize, card1 as usize)
    };
    c1 * (101 - c1) / 2 + c2 - 1
}

/// Compute solver_hand_idx → card_pair_index mapping for a player.
fn compute_pair_indices(game: &PostFlopGame, player: usize) -> Vec<usize> {
    let cards = game.private_cards(player);
    cards
        .iter()
        .map(|&(c1, c2)| card_pair_to_index(c1, c2))
        .collect()
}

// =============================================================================
// Collect (pot, stack) at each boundary in DFS order
// =============================================================================

fn collect_boundary_pot_stack(
    game: &PostFlopGame,
    starting_pot: i32,
    effective_stack: i32,
) -> Vec<(f32, f32)> {
    fn walk(
        node: &mut PostFlopNode,
        results: &mut Vec<(f32, f32)>,
        starting_pot: i32,
        effective_stack: i32,
    ) {
        if node.is_terminal() {
            return;
        }
        if node.is_chance() && node.turn() == NOT_DEALT {
            let amount = node.amount();
            results.push((
                (starting_pot + 2 * amount) as f32,
                (effective_stack - amount) as f32,
            ));
            return;
        }
        let num_actions = node.num_actions();
        for a in 0..num_actions {
            walk(&mut node.play(a), results, starting_pot, effective_stack);
        }
    }
    let mut root = game.root();
    let mut results = Vec::new();
    walk(&mut root, &mut results, starting_pot, effective_stack);
    results
}

// =============================================================================
// ONNX model wrapper with 1326-dim expansion
// =============================================================================

struct OnnxModel {
    session: Session,
    meta: ModelMeta,
}

impl OnnxModel {
    fn load(model_dir: &str) -> Self {
        let model_path = format!("{}/model.onnx", model_dir);
        let meta_path = format!("{}/meta.json", model_dir);

        let meta = ModelMeta::load(&meta_path);
        assert_eq!(
            meta.in_dim,
            3 + FULL_HANDS,
            "Model in_dim {} != expected {}",
            meta.in_dim,
            3 + FULL_HANDS
        );
        assert_eq!(
            meta.out_dim, FULL_HANDS,
            "Model out_dim {} != expected {}",
            meta.out_dim, FULL_HANDS
        );

        let session = Session::builder()
            .expect("Failed to create session builder")
            .with_intra_threads(1)
            .expect("Failed to set threads")
            .commit_from_file(&model_path)
            .expect("Failed to load ONNX model");

        Self { session, meta }
    }

    /// Predict boundary CFVs for all boundaries at once.
    ///
    /// Takes solver-indexed cfreaches, expands to 1326-dim, runs model,
    /// extracts back to solver indexing.
    fn predict_boundary_cfvs(
        &mut self,
        cfreaches: &[Vec<f32>],
        boundary_pot_stack: &[(f32, f32)],
        player: usize,
        opponent_pair_indices: &[usize],
        player_pair_indices: &[usize],
    ) -> Vec<Vec<f32>> {
        let in_dim = self.meta.in_dim;
        let max_pot = self.meta.max_pot;
        let batch_size = cfreaches.len();
        assert_eq!(batch_size, boundary_pot_stack.len());

        let num_player_hands = player_pair_indices.len();

        // Build batch input: [pot_norm, stack_norm, player, cfreach_1326]
        let mut input_data = vec![0.0f32; batch_size * in_dim];

        for (b, cfreach_solver) in cfreaches.iter().enumerate() {
            let offset = b * in_dim;
            let (pot, stack) = boundary_pot_stack[b];

            input_data[offset] = pot / max_pot;
            input_data[offset + 1] = stack / max_pot;
            input_data[offset + 2] = player as f32;

            // Expand cfreach from solver indexing to 1326-dim
            let cfreach_offset = offset + 3;
            for (h, &reach) in cfreach_solver.iter().enumerate() {
                let pair_idx = opponent_pair_indices[h];
                input_data[cfreach_offset + pair_idx] = reach;
            }
        }

        // Run inference
        let input_array =
            ndarray::Array2::from_shape_vec((batch_size, in_dim), input_data)
                .expect("Failed to create input array");

        let input_value =
            ort::value::Tensor::from_array(input_array).expect("Failed to create input tensor");

        let outputs = self
            .session
            .run(ort::inputs![input_value])
            .expect("Model inference failed");

        let (_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract output tensor");

        // Extract player's CFVs from 1326-dim output back to solver indexing
        let mut boundary_cfvs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let row_start = b * FULL_HANDS;
            let cfv: Vec<f32> = (0..num_player_hands)
                .map(|h| {
                    let pair_idx = player_pair_indices[h];
                    output_data[row_start + pair_idx] * self.meta.y_scale
                })
                .collect();
            boundary_cfvs.push(cfv);
        }

        boundary_cfvs
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <config.json> <model_dir>", args[0]);
        eprintln!(
            "Example: {} config/KcQh7s.json models/bt2_KcQh7s",
            args[0]
        );
        std::process::exit(1);
    }
    let config_path = &args[1];
    let model_dir = &args[2];

    if !Path::new(config_path).exists() {
        eprintln!("Error: Config file not found: {}", config_path);
        std::process::exit(1);
    }

    let config = load_config(config_path);
    let (card_config, tree_config) = parse_configs(&config).unwrap();

    let config_filename = Path::new(config_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();

    let total_start = Instant::now();

    println!("=== Phase 2 v2: Solve with bt2 ONNX Model (1326-dim) ===");
    println!("Config: {}", config_path);
    println!("Model: {}", model_dir);
    println!();

    // Load ONNX model
    println!("Loading model...");
    let load_start = Instant::now();
    let mut model = OnnxModel::load(model_dir);
    let load_time = load_start.elapsed().as_secs_f64();
    println!("  Input dim: {} (3 + {})", model.meta.in_dim, FULL_HANDS);
    println!("  Output dim: {} ({})", model.meta.out_dim, FULL_HANDS);
    println!("  max_pot: {}", model.meta.max_pot);
    println!("  y_scale: {}", model.meta.y_scale);
    println!("  Load time: {:.3}s", load_time);
    println!();

    // Build flop game tree
    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    let (mem_usage, _) = game.memory_usage();
    let memory_mb = mem_usage as f64 / 1024.0 / 1024.0;
    println!(
        "  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0),
        game.num_private_hands(1)
    );
    println!("  Memory: {:.2} MB", memory_mb);
    game.allocate_memory(false);

    // Compute solver_hand_idx → card_pair_index mappings
    let oop_pair_indices = compute_pair_indices(&game, 0);
    let ip_pair_indices = compute_pair_indices(&game, 1);
    println!(
        "  OOP: {} hands → 1326-dim (card_pair_to_index)",
        oop_pair_indices.len()
    );
    println!(
        "  IP:  {} hands → 1326-dim (card_pair_to_index)",
        ip_pair_indices.len()
    );

    // Verify hand counts match model expectations
    assert_eq!(
        game.num_private_hands(0),
        model.meta.num_oop,
        "OOP hands mismatch: game={} model={}",
        game.num_private_hands(0),
        model.meta.num_oop
    );
    assert_eq!(
        game.num_private_hands(1),
        model.meta.num_ip,
        "IP hands mismatch: game={} model={}",
        game.num_private_hands(1),
        model.meta.num_ip
    );

    // Collect boundary (pot, stack) for this tree
    let boundary_pot_stack = collect_boundary_pot_stack(
        &game,
        tree_config.starting_pot,
        tree_config.effective_stack,
    );
    let num_boundaries = boundary_pot_stack.len();
    println!("  Boundaries: {}", num_boundaries);
    for &(pot, stack) in &boundary_pot_stack {
        print!("    pot={}, stack={}  ", pot as i32, stack as i32);
    }
    println!();
    println!();

    // Solve
    let max_iterations = config.solver.max_iterations as usize;
    println!(
        "Solving ({} iterations with bt2 model, 1326-dim, trained on {} iters)...",
        max_iterations, model.meta.total_iterations
    );
    let solve_start = Instant::now();

    for t in 0..max_iterations {
        for player in 0..2 {
            // Determine pair index mappings for this player
            let (opponent_pair_indices, player_pair_indices) = if player == 0 {
                (&ip_pair_indices, &oop_pair_indices)
            } else {
                (&oop_pair_indices, &ip_pair_indices)
            };

            // 1. Collect individual cfreach at each boundary
            let cfreaches = collect_boundary_cfreaches(&game, player);

            // 2. Run model (handles expand/extract internally)
            let boundary_cfvs = model.predict_boundary_cfvs(
                &cfreaches,
                &boundary_pot_stack,
                player,
                opponent_pair_indices,
                player_pair_indices,
            );

            // 3. Standard replay with predicted CFVs
            solve_step_for_player_replay(&game, t as u32, player, &boundary_cfvs);
        }

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            let elapsed = solve_start.elapsed().as_secs_f64();
            let per_iter = elapsed / (t + 1) as f64;
            print!(
                "\r  iteration: {} / {} ({:.2}s, {:.4}s/iter)",
                t + 1,
                max_iterations,
                elapsed,
                per_iter,
            );
            io::stdout().flush().unwrap();
        }
    }
    println!();
    let solve_time = solve_start.elapsed().as_secs_f64();

    // Finalize
    println!("Finalizing...");
    let finalize_start = Instant::now();
    finalize(&mut game);
    let finalize_time = finalize_start.elapsed().as_secs_f64();
    println!("  Finalize: {:.2}s", finalize_time);

    // Save
    let output_path = format!("data/out/{}-model-v2.flop", config_filename);
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let save_start = Instant::now();
    save_data_to_file(&game, "model-v2", &output_path, None)
        .expect("Failed to save .flop file");
    let save_time = save_start.elapsed().as_secs_f64();
    println!("  Save: {:.2}s ({})", save_time, output_path);

    let total_time = total_start.elapsed().as_secs_f64();

    println!();
    println!("=== Summary ===");
    println!("Output: {}", output_path);
    println!("Model load: {:.3}s", load_time);
    println!(
        "Solve time: {:.2}s ({} iterations, {:.4}s/iter)",
        solve_time,
        max_iterations,
        solve_time / max_iterations as f64
    );
    println!("Finalize: {:.2}s", finalize_time);
    println!("Save: {:.2}s", save_time);
    println!("Total time: {:.2}s", total_time);
    println!("Memory: {:.2} MB", memory_mb);
    println!();
    println!("Compare against baseline:");
    println!(
        "  cargo run --example compare_flop_files --release --features \"bincode rayon\" -- \
data/out/dpair2/{}-standard.flop {}",
        config_filename, output_path
    );
}
