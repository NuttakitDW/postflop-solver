//! Benchmark ONNX models (10k/25k/50k/100k) on a seen board using locked-flop deepstack.
//!
//! Everything is hardcoded — no CLI args needed.
//!
//! Usage:
//!   cargo run --example benchmark_models --release --features "bincode rayon zstd jemalloc onnx-coreml"

#[cfg(feature = "jemalloc")]
use tikv_jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use postflop_solver::*;
use std::fs;
use std::io::Write;
use std::time::Instant;

// ── Hardcoded settings ───────────────────────────────────────────────────

// Board: 5dJdQc — most-seen flop in training data (14 samples in 100k)
const FLOP: &str = "5dJdQc";

// Ranges: standard opening ranges
const RANGE_OOP: &str = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo";
const RANGE_IP: &str = "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo";

// Tree config (matches template.json)
const STARTING_POT: i32 = 55;
const EFFECTIVE_STACK: i32 = 180;

// Solver
const FLOP_ITERS: u32 = 100;
const TURNRIVER_ITERS: u32 = 100;
const TARGET_EXPLOIT_PCT: f32 = 0.5;

// Device
const DEVICE: &str = "coreml";

// Models to benchmark
const MODELS: &[(&str, u64)] = &[
    ("models/benchmark_onnx/model_10k.onnx", 10_000),
    ("models/benchmark_onnx/model_25k.onnx", 25_000),
    ("models/benchmark_onnx/model_50k.onnx", 50_000),
    ("models/benchmark_onnx/model_100k.onnx", 100_000),
];

// Output
const OUTPUT_CSV: &str = "models/benchmark_onnx/benchmark_results.csv";

// ── Helpers ──────────────────────────────────────────────────────────────

fn normalize_bet_sizes(s: &str) -> String {
    s.split(',')
        .map(|part| {
            let trimmed = part.trim();
            if trimmed.is_empty() { return trimmed.to_string(); }
            let lower = trimmed.to_lowercase();
            if lower == "a" || lower.ends_with('%') || lower.ends_with('x')
                || lower.contains('c') || lower.contains('e') {
                trimmed.to_string()
            } else if trimmed.parse::<f64>().is_ok() {
                format!("{}%", trimmed)
            } else {
                trimmed.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn build_tree_config() -> Result<TreeConfig, String> {
    let oop_flop = BetSizeOptions::try_from((
        normalize_bet_sizes("33, a").as_str(),
        normalize_bet_sizes("33, 55, a").as_str(),
    )).map_err(|e| format!("{}", e))?;
    let oop_turn = BetSizeOptions::try_from((
        normalize_bet_sizes("20, 33, 55, 83, 125, 200, a").as_str(),
        normalize_bet_sizes("33, 55, a").as_str(),
    )).map_err(|e| format!("{}", e))?;
    let oop_river = BetSizeOptions::try_from((
        normalize_bet_sizes("11, 35, 60, 85, 149, a").as_str(),
        normalize_bet_sizes("33, 55, a").as_str(),
    )).map_err(|e| format!("{}", e))?;
    let ip_flop = BetSizeOptions::try_from((
        normalize_bet_sizes("20, 33, 55, 83, 125, a").as_str(),
        normalize_bet_sizes("33, 55, a").as_str(),
    )).map_err(|e| format!("{}", e))?;
    let ip_turn = BetSizeOptions::try_from((
        normalize_bet_sizes("20, 33, 55, 83, 125, 200, a").as_str(),
        normalize_bet_sizes("33, 55, a").as_str(),
    )).map_err(|e| format!("{}", e))?;
    let ip_river = BetSizeOptions::try_from((
        normalize_bet_sizes("11, 35, 60, 85, 149, a").as_str(),
        normalize_bet_sizes("33, 55, a").as_str(),
    )).map_err(|e| format!("{}", e))?;

    Ok(TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: STARTING_POT,
        effective_stack: EFFECTIVE_STACK,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_flop, ip_flop],
        turn_bet_sizes: [oop_turn, ip_turn],
        river_bet_sizes: [oop_river, ip_river],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 1.5,
        force_allin_threshold: 0.2,
        merging_threshold: 0.1,
        max_raises_per_street: 0,
    })
}

fn build_game() -> Result<PostFlopGame, String> {
    let oop: Range = RANGE_OOP.parse().map_err(|e| format!("{}", e))?;
    let ip: Range = RANGE_IP.parse().map_err(|e| format!("{}", e))?;
    let flop = flop_from_str(FLOP).map_err(|e| format!("{}", e))?;

    let card_config = CardConfig {
        range: [oop, ip],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    let tree_config = build_tree_config()?;
    let action_tree = ActionTree::new(tree_config).map_err(|e| format!("{}", e))?;
    PostFlopGame::with_config(card_config, action_tree).map_err(|e| format!("{}", e))
}

struct BenchmarkResult {
    name: String,
    data_points: u64,
    exploitability: f32,
    exploitability_pct: f32,
    solve_time_secs: f64,
}

// ── Run deepstack model ──────────────────────────────────────────────────

#[cfg(feature = "onnx")]
fn run_deepstack(model_path: &str) -> Result<(f32, f64), String> {
    use postflop_solver::net::{Device, TurnValueNet};

    let device: Device = DEVICE.parse()?;
    let net = TurnValueNet::new(model_path, device)?;

    let mut game = build_game()?;
    game.allocate_memory(false);

    let target = STARTING_POT as f32 * TARGET_EXPLOIT_PCT / 100.0;

    let start = Instant::now();
    let exploitability = solve_with_locked_flop(
        &mut game, FLOP_ITERS, TURNRIVER_ITERS, target, true, &net,
    );
    Ok((exploitability, start.elapsed().as_secs_f64()))
}

#[cfg(not(feature = "onnx"))]
fn run_deepstack(_model_path: &str) -> Result<(f32, f64), String> {
    Err("Requires 'onnx' feature".into())
}

// ── Run standard solver ──────────────────────────────────────────────────

fn run_standard() -> Result<(f32, f64), String> {
    let mut game = build_game()?;
    game.allocate_memory(false);

    let target = STARTING_POT as f32 * TARGET_EXPLOIT_PCT / 100.0;
    let total_iters = FLOP_ITERS + TURNRIVER_ITERS;

    let start = Instant::now();
    let exploitability = solve(&mut game, total_iters, target, true);
    Ok((exploitability, start.elapsed().as_secs_f64()))
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Model Benchmark ===");
    println!("Board:  {}", FLOP);
    println!("Pot:    {}, Stack: {}", STARTING_POT, EFFECTIVE_STACK);
    println!("Flop iters: {}, Turn/River iters: {}", FLOP_ITERS, TURNRIVER_ITERS);
    println!("Device: {}", DEVICE);
    println!("Models: {}", MODELS.len());
    println!();

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Standard solver baseline
    println!("--- Standard solver (baseline, {} total iters) ---", FLOP_ITERS + TURNRIVER_ITERS);
    match run_standard() {
        Ok((exploit, time)) => {
            let pct = exploit / STARTING_POT as f32 * 100.0;
            println!("  Exploitability: {:.4} ({:.4}% of pot)", exploit, pct);
            println!("  Time: {:.2}s\n", time);
            results.push(BenchmarkResult {
                name: "standard_solver".to_string(),
                data_points: 0,
                exploitability: exploit,
                exploitability_pct: pct,
                solve_time_secs: time,
            });
        }
        Err(e) => eprintln!("  Standard solver FAILED: {}\n", e),
    }

    // Each deepstack model
    for (model_path, data_points) in MODELS {
        println!("--- {} ({}k data points) ---", model_path, data_points / 1000);
        match run_deepstack(model_path) {
            Ok((exploit, time)) => {
                let pct = exploit / STARTING_POT as f32 * 100.0;
                println!("  Exploitability: {:.4} ({:.4}% of pot)", exploit, pct);
                println!("  Time: {:.2}s\n", time);
                results.push(BenchmarkResult {
                    name: format!("{}k", data_points / 1000),
                    data_points: *data_points,
                    exploitability: exploit,
                    exploitability_pct: pct,
                    solve_time_secs: time,
                });
            }
            Err(e) => eprintln!("  FAILED: {}\n", e),
        }
    }

    // Write CSV
    if let Some(parent) = std::path::Path::new(OUTPUT_CSV).parent() {
        fs::create_dir_all(parent).ok();
    }
    let mut f = fs::File::create(OUTPUT_CSV).expect("Failed to create CSV");
    writeln!(f, "model,data_points,exploitability,exploitability_pct,solve_time_secs").unwrap();
    for r in &results {
        writeln!(f, "{},{},{:.6},{:.4},{:.2}",
            r.name, r.data_points, r.exploitability, r.exploitability_pct, r.solve_time_secs
        ).unwrap();
    }
    println!("CSV saved: {}", OUTPUT_CSV);

    // Summary
    println!();
    println!("=== Summary ===");
    println!("{:<30} {:>12} {:>16} {:>12}", "Model", "Data Points", "Exploit. (%pot)", "Time (s)");
    println!("{}", "-".repeat(74));
    for r in &results {
        println!("{:<30} {:>12} {:>15.4}% {:>11.2}s",
            r.name, r.data_points, r.exploitability_pct, r.solve_time_secs);
    }
}
