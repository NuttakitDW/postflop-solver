//! Phase 2 v1: Solve a flop game using an ONNX neural network model.
//!
//! Instead of looking up pre-recorded boundary CFVs from a .dpairs2 file,
//! this solver uses an ONNX model to predict CFVs from the current cfreach
//! at each boundary.
//!
//! Flow per iteration per player:
//!   1. Traverse flop tree to collect cfreach at each boundary
//!   2. Pad cfreach to max_hands, build model input (solver indexing)
//!   3. Run ONNX inference (batched across all boundaries)
//!   4. Extract first num_player_hands values as CFV
//!   5. Pass predicted CFVs to solve_step_for_player_replay
//!
//! Usage:
//!   ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
//!   cargo run --example solve_with_model_v1 --release --features "bincode rayon" -- \
//!     config/KcQh7s.json models/bt1_KcQh7s

#[path = "../common/mod.rs"]
mod common;

use common::*;
use ort::session::Session;
use postflop_solver::*;
use std::env;
use std::io::{self, Write as _};
use std::path::Path;
use std::time::Instant;

// =============================================================================
// Model metadata
// =============================================================================

#[derive(Debug)]
struct ModelMeta {
    num_boundaries: usize,
    num_iterations: usize,
    y_scale: f32,
    in_dim: usize,
    out_dim: usize,
    max_hands: usize,
    num_oop: usize,
    num_ip: usize,
}

impl ModelMeta {
    fn load(path: &str) -> Self {
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("Failed to read meta"))
                .expect("Failed to parse meta JSON");

        Self {
            num_boundaries: data["num_boundaries"].as_u64().unwrap() as usize,
            num_iterations: data["num_iterations"].as_u64().unwrap() as usize,
            y_scale: data["y_scale"].as_f64().unwrap() as f32,
            in_dim: data["in_dim"].as_u64().unwrap() as usize,
            out_dim: data["out_dim"].as_u64().unwrap() as usize,
            max_hands: data["max_hands"].as_u64().unwrap() as usize,
            num_oop: data["num_oop"].as_u64().unwrap() as usize,
            num_ip: data["num_ip"].as_u64().unwrap() as usize,
        }
    }

    fn num_player_hands(&self, player: usize) -> usize {
        if player == 0 { self.num_oop } else { self.num_ip }
    }
}

// =============================================================================
// ONNX model wrapper
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
    /// Takes cfreaches in solver indexing, returns CFVs in solver indexing.
    fn predict_boundary_cfvs(
        &mut self,
        cfreaches: &[Vec<f32>],
        player: usize,
    ) -> Vec<Vec<f32>> {
        let num_boundaries = self.meta.num_boundaries;
        let in_dim = self.meta.in_dim;
        let out_dim = self.meta.out_dim;
        let batch_size = cfreaches.len();
        assert_eq!(batch_size, num_boundaries);

        let num_player_hands = self.meta.num_player_hands(player);

        // Build batch input: [boundary_onehot + player + cfreach_padded]
        let mut input_data = vec![0.0f32; batch_size * in_dim];

        for (b, cfreach_solver) in cfreaches.iter().enumerate() {
            let offset = b * in_dim;

            // Boundary one-hot
            input_data[offset + b] = 1.0;

            // Player flag
            input_data[offset + num_boundaries] = player as f32;

            // cfreach in solver indexing (padded to max_hands with zeros)
            let cfreach_offset = offset + num_boundaries + 1;
            for (i, &val) in cfreach_solver.iter().enumerate() {
                input_data[cfreach_offset + i] = val;
            }
        }

        // Run inference
        let input_array = ndarray::Array2::from_shape_vec(
            (batch_size, in_dim),
            input_data,
        ).expect("Failed to create input array");

        let input_value = ort::value::Tensor::from_array(input_array)
            .expect("Failed to create input tensor");

        let outputs = self.session.run(
            ort::inputs![input_value]
        ).expect("Model inference failed");

        let (_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("Failed to extract output tensor");

        // Extract first num_player_hands values per boundary + unscale
        let mut boundary_cfvs = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let row_start = b * out_dim;
            let cfv: Vec<f32> = (0..num_player_hands)
                .map(|i| output_data[row_start + i] * self.meta.y_scale)
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
        eprintln!("Example: {} config/KcQh7s.json models/bt1_KcQh7s", args[0]);
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
        .file_stem().unwrap().to_str().unwrap();

    let total_start = Instant::now();

    println!("=== Phase 2 v1: Solve with ONNX Model ===");
    println!("Config: {}", config_path);
    println!("Model: {}", model_dir);
    println!();

    // Load ONNX model
    println!("Loading model...");
    let load_start = Instant::now();
    let mut model = OnnxModel::load(model_dir);
    let load_time = load_start.elapsed().as_secs_f64();
    println!("  Boundaries: {}", model.meta.num_boundaries);
    println!("  Input dim: {}", model.meta.in_dim);
    println!("  Output dim: {}", model.meta.out_dim);
    println!("  max_hands: {}", model.meta.max_hands);
    println!("  y_scale: {}", model.meta.y_scale);
    println!("  Load time: {:.3}s", load_time);
    println!();

    // Build flop game tree
    println!("Building game tree...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    let (mem_usage, _) = game.memory_usage();
    let memory_mb = mem_usage as f64 / 1024.0 / 1024.0;
    println!("  OOP hands: {}, IP hands: {}",
        game.num_private_hands(0), game.num_private_hands(1));
    println!("  Memory: {:.2} MB", memory_mb);
    game.allocate_memory(false);
    println!();

    // Solve — use iterations from config (can exceed training data iterations)
    let max_iterations = config.solver.max_iterations as usize;
    println!("Solving ({} iterations with NN model, trained on {} iters)...",
        max_iterations, model.meta.num_iterations);
    let solve_start = Instant::now();

    for t in 0..max_iterations {
        for player in 0..2 {
            // 1. Collect cfreach at each boundary using current strategies
            let cfreaches = collect_boundary_cfreaches(&game, player);

            // 2. Run model to predict CFVs
            let boundary_cfvs = model.predict_boundary_cfvs(&cfreaches, player);

            // 3. Standard replay with predicted CFVs
            solve_step_for_player_replay(
                &game,
                t as u32,
                player,
                &boundary_cfvs,
            );
        }

        if (t + 1) % 10 == 0 || t + 1 == max_iterations {
            let elapsed = solve_start.elapsed().as_secs_f64();
            let per_iter = elapsed / (t + 1) as f64;
            print!(
                "\r  iteration: {} / {} ({:.2}s, {:.4}s/iter)",
                t + 1, max_iterations, elapsed, per_iter,
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
    let output_path = format!("data/out/{}-model-v1.flop", config_filename);
    if let Some(parent) = Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let save_start = Instant::now();
    save_data_to_file(&game, "model-v1", &output_path, None)
        .expect("Failed to save .flop file");
    let save_time = save_start.elapsed().as_secs_f64();
    println!("  Save: {:.2}s ({})", save_time, output_path);

    let total_time = total_start.elapsed().as_secs_f64();

    println!();
    println!("=== Summary ===");
    println!("Output: {}", output_path);
    println!("Model load: {:.3}s", load_time);
    println!("Solve time: {:.2}s ({} iterations, {:.4}s/iter)",
        solve_time, max_iterations, solve_time / max_iterations as f64);
    println!("Finalize: {:.2}s", finalize_time);
    println!("Save: {:.2}s", save_time);
    println!("Total time: {:.2}s", total_time);
    println!("Memory: {:.2} MB", memory_mb);
    println!();
    println!("Compare against baseline:");
    println!("  cargo run --example compare_flop_files --release --features \"bincode rayon\" -- \
data/out/dpair2/{}-standard.flop {}", config_filename, output_path);
}
