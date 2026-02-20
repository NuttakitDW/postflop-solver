//! Projection script: converts raw training data into bucketed training data.
//!
//! Reads the "Golden Source" files (meta.npy, ranges.npy, values.npy) produced
//! by `generate_raw_data`, and projects them into K-bucket space using the
//! current bucketing features and weights.
//!
//! This is cheap to re-run (~minutes) whenever you change:
//!   - K (number of buckets)
//!   - Feature definitions in compute_buckets()
//!   - Feature weights in FEATURE_WEIGHTS
//!   - Board feature definitions in compute_board_features()
//!
//! Usage:
//!   cargo run --example preprocess_data --release --features "rayon" -- \
//!     --input-dir ./data/solver_output \
//!     --output-dir ./data/training_data \
//!     --k 1000

use ndarray::{Array1, Array2};
use postflop_solver::*;
use rayon::prelude::*;
use std::fs;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const NUM_COMBOS: usize = 1326;
const META_DIM: usize = 6;
const BOARD_FEATURES: usize = 15;

// ---------------------------------------------------------------------------
// NPY reader
// ---------------------------------------------------------------------------

fn read_npy_2d(path: &str) -> (Vec<usize>, Vec<f32>) {
    let data = fs::read(path).unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));

    assert!(
        data.len() >= 10 && &data[..6] == b"\x93NUMPY",
        "Invalid NPY file: {}",
        path
    );

    let version = data[6];
    let header_offset;
    let header_len;

    if version == 1 {
        header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        header_offset = 10;
    } else if version == 2 {
        header_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        header_offset = 12;
    } else {
        panic!("Unsupported NPY version {} in {}", version, path);
    }

    let header =
        std::str::from_utf8(&data[header_offset..header_offset + header_len]).unwrap_or_else(
            |e| {
                panic!("Invalid header encoding in {}: {}", path, e);
            },
        );

    // Verify f32 little-endian
    assert!(
        header.contains("<f4") || header.contains("float32"),
        "Expected float32 data in {}, got header: {}",
        path,
        header
    );

    // Parse shape: ('shape': (N, M), ) or ('shape': (N,), )
    let shape_key = "'shape':";
    let key_pos = header.find(shape_key).unwrap();
    let after_key = &header[key_pos + shape_key.len()..];
    let paren_start = after_key.find('(').unwrap();
    let paren_end = after_key.find(')').unwrap();
    let shape_str = &after_key[paren_start + 1..paren_end];

    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.parse::<usize>().unwrap_or_else(|e| {
                    panic!("Bad shape component '{}' in {}: {}", trimmed, path, e)
                }))
            }
        })
        .collect();

    let data_start = header_offset + header_len;
    let float_data: Vec<f32> = data[data_start..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();

    let expected_len: usize = shape.iter().product();
    assert_eq!(
        float_data.len(),
        expected_len,
        "Shape {:?} expects {} floats but got {} in {}",
        shape,
        expected_len,
        float_data.len(),
        path,
    );

    (shape, float_data)
}

// ---------------------------------------------------------------------------
// NPY writer
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
// Per-row projection
// ---------------------------------------------------------------------------

struct ProjectedRow {
    input: Vec<f32>,  // (BOARD_FEATURES + 2*K,)
    target: Vec<f32>, // (2*K,)
}

fn project_row(
    meta_row: &[f32],
    ranges_row: &[f32],
    values_row: &[f32],
    k: usize,
) -> ProjectedRow {
    // Reconstruct board
    let board: [Card; 4] = [
        meta_row[0] as Card,
        meta_row[1] as Card,
        meta_row[2] as Card,
        meta_row[3] as Card,
    ];
    let pot = meta_row[4];
    let stack = meta_row[5];

    // Extract 1326-element arrays
    let reach_oop: [f32; NUM_COMBOS] = ranges_row[..NUM_COMBOS].try_into().unwrap();
    let reach_ip: [f32; NUM_COMBOS] = ranges_row[NUM_COMBOS..].try_into().unwrap();
    let cfv_oop: [f32; NUM_COMBOS] = values_row[..NUM_COMBOS].try_into().unwrap();
    let cfv_ip: [f32; NUM_COMBOS] = values_row[NUM_COMBOS..].try_into().unwrap();

    // Compute bucket mapping
    let bucket_mapping = compute_buckets(&board, k);

    // Project ranges and CFVs to buckets
    let range_oop_bucketed = project_range_to_buckets(&reach_oop, &bucket_mapping);
    let range_ip_bucketed = project_range_to_buckets(&reach_ip, &bucket_mapping);
    let cfv_oop_bucketed = project_cfv_to_buckets(&cfv_oop, &reach_oop, &bucket_mapping);
    let cfv_ip_bucketed = project_cfv_to_buckets(&cfv_ip, &reach_ip, &bucket_mapping);

    // Board features
    let board_features = compute_board_features(&board, pot, stack);

    // Assemble input: [board(15) | range_oop(K) | range_ip(K)]
    let input_dim = BOARD_FEATURES + 2 * k;
    let mut input = vec![0.0f32; input_dim];
    input[..BOARD_FEATURES].copy_from_slice(&board_features);
    input[BOARD_FEATURES..BOARD_FEATURES + k].copy_from_slice(&range_oop_bucketed);
    input[BOARD_FEATURES + k..].copy_from_slice(&range_ip_bucketed);

    // Assemble target: [cfv_oop(K) | cfv_ip(K)]
    let mut target = vec![0.0f32; 2 * k];
    target[..k].copy_from_slice(&cfv_oop_bucketed);
    target[k..].copy_from_slice(&cfv_ip_bucketed);

    ProjectedRow { input, target }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn parse_args() -> (String, String, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut input_dir = String::from("./data/solver_output");
    let mut output_dir = String::from("./data/training_data");
    let mut k: usize = DEFAULT_K; // 1000

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input-dir" => {
                input_dir = args[i + 1].clone();
                i += 2;
            }
            "--output-dir" => {
                output_dir = args[i + 1].clone();
                i += 2;
            }
            "--k" => {
                k = args[i + 1].parse().expect("Invalid --k");
                i += 2;
            }
            "--help" | "-h" => {
                eprintln!("Usage: preprocess_data [OPTIONS]");
                eprintln!(
                    "  --input-dir <DIR>   Raw data directory (default: ./data/solver_output)"
                );
                eprintln!(
                    "  --output-dir <DIR>  Output directory (default: ./data/training_data)"
                );
                eprintln!("  --k <N>             Number of buckets (default: {})", DEFAULT_K);
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    (input_dir, output_dir, k)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let (input_dir, output_dir, k) = parse_args();

    eprintln!("=== Raw → Bucketed Projection ===");
    eprintln!("Input dir:   {}", input_dir);
    eprintln!("Output dir:  {}", output_dir);
    eprintln!("K (buckets): {}", k);
    eprintln!();

    // Read raw data
    let read_start = Instant::now();

    let meta_path = format!("{}/meta.npy", input_dir);
    let ranges_path = format!("{}/ranges.npy", input_dir);
    let values_path = format!("{}/values.npy", input_dir);

    eprintln!("Reading {}...", meta_path);
    let (meta_shape, meta_data) = read_npy_2d(&meta_path);
    assert_eq!(meta_shape.len(), 2, "meta.npy must be 2D");
    assert_eq!(meta_shape[1], META_DIM, "meta.npy dim[1] must be {}", META_DIM);
    let n = meta_shape[0];

    eprintln!("Reading {}...", ranges_path);
    let (ranges_shape, ranges_data) = read_npy_2d(&ranges_path);
    assert_eq!(ranges_shape, vec![n, 2 * NUM_COMBOS], "ranges.npy shape mismatch");

    eprintln!("Reading {}...", values_path);
    let (values_shape, values_data) = read_npy_2d(&values_path);
    assert_eq!(values_shape, vec![n, 2 * NUM_COMBOS], "values.npy shape mismatch");

    eprintln!(
        "Loaded {} samples in {:.1}s",
        n,
        read_start.elapsed().as_secs_f64()
    );

    // Project each row in parallel
    let project_start = Instant::now();
    let processed = AtomicUsize::new(0);

    let input_dim = BOARD_FEATURES + 2 * k;
    let output_dim = 2 * k;

    let results: Vec<ProjectedRow> = (0..n)
        .into_par_iter()
        .map(|i| {
            let meta_row = &meta_data[i * META_DIM..(i + 1) * META_DIM];
            let ranges_row =
                &ranges_data[i * 2 * NUM_COMBOS..(i + 1) * 2 * NUM_COMBOS];
            let values_row =
                &values_data[i * 2 * NUM_COMBOS..(i + 1) * 2 * NUM_COMBOS];

            let row = project_row(meta_row, ranges_row, values_row, k);

            let done = processed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 1000 == 0 || done == n {
                let elapsed = project_start.elapsed().as_secs_f64();
                eprintln!(
                    "  projected {}/{} ({:.1} rows/s)",
                    done,
                    n,
                    done as f64 / elapsed,
                );
            }

            row
        })
        .collect();

    eprintln!(
        "Projection complete in {:.1}s ({:.0} rows/s)",
        project_start.elapsed().as_secs_f64(),
        n as f64 / project_start.elapsed().as_secs_f64(),
    );

    // Assemble into 2D arrays
    eprintln!("Assembling arrays...");
    let mut inputs = Array2::<f32>::zeros((n, input_dim));
    let mut targets = Array2::<f32>::zeros((n, output_dim));

    for (i, row) in results.iter().enumerate() {
        inputs.row_mut(i).assign(&Array1::from(row.input.clone()));
        targets.row_mut(i).assign(&Array1::from(row.target.clone()));
    }

    eprintln!(
        "Shapes: inputs={:?}, targets={:?}",
        inputs.shape(),
        targets.shape()
    );

    // Write output
    fs::create_dir_all(&output_dir).expect("Failed to create output directory");
    let write_start = Instant::now();

    let inputs_path = format!("{}/inputs.npy", output_dir);
    let targets_path = format!("{}/targets.npy", output_dir);

    write_npy_2d(&inputs_path, &inputs).expect("Failed to write inputs.npy");
    eprintln!("Wrote {}", inputs_path);

    write_npy_2d(&targets_path, &targets).expect("Failed to write targets.npy");
    eprintln!("Wrote {}", targets_path);

    eprintln!(
        "Write complete in {:.1}s",
        write_start.elapsed().as_secs_f64()
    );
    eprintln!("Done! {} bucketed data points with K={}", n, k);
}
