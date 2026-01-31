//! Benchmark for the subgame solving framework.
//!
//! Run with: cargo run --example subgame_bench --features subgame --release
//!
//! This benchmark measures:
//! - EHS2 computation time
//! - K-means clustering time
//! - Full abstraction computation time
//! - Subgame generation time
//! - Batch solving throughput

use postflop_solver::subgame::*;
use std::time::{Duration, Instant};

/// Run a benchmark function multiple times and report statistics.
fn bench<F>(name: &str, iterations: u32, mut f: F)
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..3 {
        f();
    }

    // Actual benchmark
    let mut times = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    let total: Duration = times.iter().sum();
    let avg = total / iterations;
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();

    println!(
        "{:40} avg={:>10.2?}  min={:>10.2?}  max={:>10.2?}",
        name, avg, min, max
    );
}

fn main() {
    println!("=== Subgame Framework Benchmarks ===\n");
    println!("Running in {} mode\n", if cfg!(debug_assertions) { "DEBUG" } else { "RELEASE" });

    // Test boards representing different textures
    let boards: [([u8; 3], &str); 3] = [
        ([0, 4, 8], "Monotone (2c3c4c)"),
        ([0, 5, 10], "Rainbow (2c3d4h)"),
        ([0, 4, 9], "Two-tone (2c3c4d)"),
    ];

    println!("=== Card Abstraction Benchmarks ===\n");

    for (flop, desc) in &boards {
        println!("Board: {} {:?}", desc, flop);

        // Benchmark abstraction with different bucket counts
        for buckets in [5u8, 10] {
            let config = AbstractionConfig::new(buckets, buckets);

            bench(
                &format!("  Abstraction {}x{} buckets", buckets, buckets),
                5,
                || {
                    #[cfg(feature = "rayon")]
                    let _ = AbstractionMapping::compute_parallel(flop, &config);
                    #[cfg(not(feature = "rayon"))]
                    let _ = AbstractionMapping::compute(flop, &config);
                },
            );
        }
        println!();
    }

    println!("=== Subgame Generation Benchmarks ===\n");

    let flop = [0, 4, 8];
    let boundary = BoundaryData::new(
        [vec![0.5; 100], vec![0.5; 100]],
        [vec![0.0; 100], vec![0.0; 100]],
        [vec![10.0; 100], vec![10.0; 100]],
        100,
        500,
        vec![],
        None,
        0,
    );

    bench("Generate turn subgames (49)", 100, || {
        let _ = generate_subgame_infos(&boundary, 0, &flop);
    });

    bench("Generate river subgames (2352)", 10, || {
        let _ = generate_river_subgame_infos(&boundary, 0, &flop);
    });

    println!();

    println!("=== Batch Solving Benchmarks ===\n");

    // Benchmark batch solving with mock solver
    let options = BatchSolveOptions {
        config: SubgameConfig::fast(),
        print_progress: false,
        ..Default::default()
    };

    println!("Using mock solver (simulates ~100µs per subgame)");

    #[cfg(feature = "rayon")]
    {
        // Benchmark parallel solving
        bench("Parallel solve 10 turn subgames", 10, || {
            let turns: Vec<u8> = (12..22).filter(|&t| !flop.contains(&t)).collect();
            let _ = solve_turn_subgames(&boundary, &flop, &turns, options.clone(), mock_solve);
        });

        bench("Parallel solve 49 turn subgames", 5, || {
            let _ = solve_all_subgames(&boundary, &flop, options.clone(), mock_solve);
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        // Sequential solving
        bench("Sequential solve 10 turn subgames", 10, || {
            let mut infos: Vec<SubgameInfo> = (12..22u8)
                .filter(|&t| !flop.contains(&t))
                .map(|turn| SubgameInfo::from_boundary(&boundary, 0, &flop, turn, None))
                .collect();
            let solver = BatchSolver::with_options(options.clone());
            let _ = solver.solve_sequential(&mut infos, mock_solve);
        });
    }

    println!();

    println!("=== Archive Benchmarks ===\n");

    #[cfg(feature = "bincode")]
    {
        // Create test data
        let config = AbstractionConfig::new(5, 5);
        let abstraction = AbstractionMapping::compute(&flop, &config);
        let boundaries = BoundaryStore::new(100, 100);
        let blueprint = Blueprint::new(
            flop,
            abstraction,
            boundaries,
            BlueprintConfig::fast(),
            0.01,
            100,
            100,
        );

        let manifest = PfsManifest::new(
            PfsConfig::default(),
            PfsGameInfo::default(),
            PfsStats::default(),
        );

        bench("Build archive (small)", 100, || {
            let _ = PfsArchiveBuilder::new()
                .manifest(manifest.clone())
                .blueprint(blueprint.clone())
                .add_subgame(12, None, vec![0u8; 100])
                .build_bytes();
        });

        // Create a larger archive
        let large_archive = PfsArchiveBuilder::new()
            .manifest(manifest.clone())
            .blueprint(blueprint.clone())
            .add_subgame(12, None, vec![0u8; 10000])
            .add_subgame(16, None, vec![0u8; 10000])
            .add_subgame(20, None, vec![0u8; 10000])
            .build_bytes()
            .unwrap();

        bench("Read archive (medium)", 100, || {
            let _ = PfsArchiveReader::from_bytes(&large_archive);
        });

        println!("Archive size: {} KB", large_archive.len() / 1024);
    }

    println!();

    println!("=== Stitched Game Benchmarks ===\n");

    // Create stitched game
    let config = AbstractionConfig::new(5, 5);
    let abstraction = AbstractionMapping::compute(&flop, &config);
    let boundaries = BoundaryStore::new(100, 100);
    let blueprint = Blueprint::new(
        flop,
        abstraction,
        boundaries,
        BlueprintConfig::fast(),
        0.01,
        100,
        100,
    );

    let stitched = StitchedGame::new(blueprint);

    // Add some precomputed subgames
    for turn in 12..20u8 {
        if flop.contains(&turn) {
            continue;
        }
        let subgame = CachedSubgame {
            info: SubgameInfo::from_boundary(&boundary, 0, &flop, turn, Some(16)),
            strategy: vec![0.5; 100],
            from_cache: false,
        };
        stitched.add_precomputed(subgame);
    }

    bench("Bucket lookup", 10000, || {
        for turn in 0..52u8 {
            let _ = stitched.turn_bucket(turn);
        }
    });

    bench("Subgame lookup (hit)", 10000, || {
        let _ = stitched.get_subgame(12, Some(16));
    });

    bench("Subgame lookup (miss)", 10000, || {
        let _ = stitched.get_subgame(40, Some(44));
    });

    println!();
    println!("=== Benchmark Complete ===");
}
