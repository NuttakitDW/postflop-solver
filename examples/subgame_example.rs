//! Example demonstrating the subgame solving framework.
//!
//! This example shows the full workflow:
//! 1. Compute card abstraction (EHS2 with k-means clustering)
//! 2. Create a blueprint with boundary data
//! 3. Generate and solve subgames in parallel
//! 4. Create and read an archive
//!
//! Run with: cargo run --example subgame_example --features subgame

use postflop_solver::subgame::*;
use std::time::Instant;

fn main() {
    println!("=== Subgame Solving Framework Demo ===\n");

    // Define a flop
    let flop = [0, 4, 8]; // 2c, 3c, 4c (monotone flop)
    println!("Flop: 2c 3c 4c (cards {:?})", flop);

    // -------------------------------------------------------------------------
    // Phase 1: Card Abstraction
    // -------------------------------------------------------------------------
    println!("\n--- Phase 1: Card Abstraction (EHS2) ---");

    let start = Instant::now();
    let config = AbstractionConfig::new(5, 5); // 5 turn buckets, 5 river buckets
    println!(
        "Configuration: {} turn buckets, {} river buckets",
        config.turn_buckets, config.river_buckets
    );

    #[cfg(feature = "rayon")]
    let abstraction = AbstractionMapping::compute_parallel(&flop, &config);
    #[cfg(not(feature = "rayon"))]
    let abstraction = AbstractionMapping::compute(&flop, &config);

    println!("Abstraction computed in {:.2?}", start.elapsed());

    // Show bucket distribution
    println!("\nTurn bucket distribution:");
    for bucket in 0..config.turn_buckets {
        let turns = abstraction.turns_in_bucket(bucket);
        println!(
            "  Bucket {}: {} cards (representative: card {})",
            bucket,
            turns.len(),
            abstraction.turn_representative(bucket)
        );
    }

    // -------------------------------------------------------------------------
    // Phase 2: Blueprint Creation
    // -------------------------------------------------------------------------
    println!("\n--- Phase 2: Blueprint Creation ---");

    let blueprint_config = BlueprintConfig::fast(); // Use fast config for demo
    println!(
        "Blueprint config: {} iterations, {:.2}% target exploitability",
        blueprint_config.iterations,
        blueprint_config.target_exploitability * 100.0
    );

    // Create boundary store (in a real implementation, this would be extracted from a solved game)
    let boundary_store = BoundaryStore::new(100, 100);

    // Build the blueprint
    let blueprint = BlueprintBuilder::new()
        .board(flop)
        .config(blueprint_config.clone())
        .num_hands(100, 100)
        .with_abstraction(abstraction.clone())
        .with_boundaries(boundary_store)
        .exploitability(0.01)
        .build()
        .expect("Failed to build blueprint");

    let stats = blueprint.stats();
    println!("Blueprint created: {}", stats);

    // -------------------------------------------------------------------------
    // Phase 3: Boundary Data
    // -------------------------------------------------------------------------
    println!("\n--- Phase 3: Boundary Data ---");

    // Create sample boundary data (in real use, extracted from solved game)
    let boundary = BoundaryData::new(
        [vec![0.5; 100], vec![0.5; 100]], // ranges
        [vec![0.0; 100], vec![0.0; 100]], // cfvalues
        [vec![10.0; 100], vec![10.0; 100]], // expected values
        100,                               // pot
        500,                               // stack
        vec![0, 1],                        // action history
        None,                              // turn (none for flop->turn boundary)
        0,                                 // street
    );

    println!(
        "Sample boundary: pot={}, stack={}, avg_ev_oop={:.2}, avg_ev_ip={:.2}",
        boundary.pot,
        boundary.stack,
        boundary.average_ev(0),
        boundary.average_ev(1)
    );

    // Create safety constraint for safe subgame solving
    let safety_constraint = SafetyConstraint::from_boundary(&boundary, 0.02);
    println!(
        "Safety constraint: reach_floor={:.2}",
        safety_constraint.reach_floor
    );

    // -------------------------------------------------------------------------
    // Phase 4: Subgame Generation
    // -------------------------------------------------------------------------
    println!("\n--- Phase 4: Subgame Generation ---");

    // Generate turn subgames
    let turn_infos = generate_subgame_infos(&boundary, 0, &flop);
    println!("Generated {} turn subgames", turn_infos.len());

    // Generate river subgames (for a specific turn)
    let river_infos = generate_river_subgame_infos(&boundary, 0, &flop);
    println!("Generated {} river subgames", river_infos.len());

    // Show sample subgame info
    if let Some(info) = turn_infos.first() {
        println!(
            "\nSample turn subgame: turn={}, board={:?}, pot={}, stack={}",
            info.turn, info.board, info.pot, info.stack
        );
    }

    // -------------------------------------------------------------------------
    // Phase 5: Batch Subgame Solving
    // -------------------------------------------------------------------------
    println!("\n--- Phase 5: Batch Subgame Solving ---");

    // Configure batch solving
    let batch_options = BatchSolveOptions {
        config: SubgameConfig::fast(),
        print_progress: false,
        progress_interval: 10,
        ..Default::default()
    };

    println!(
        "Batch config: {} iterations, {:.2}% target exploitability",
        batch_options.config.iterations,
        batch_options.config.target_exploitability * 100.0
    );

    // Solve a subset of turn subgames using mock solver
    let subset_turns: Vec<u8> = (12..22).filter(|&t| !flop.contains(&t)).collect();
    println!("Solving {} turn subgames in parallel...", subset_turns.len());

    let start = Instant::now();

    #[cfg(feature = "rayon")]
    let (solved_infos, batch_stats) =
        solve_turn_subgames(&boundary, &flop, &subset_turns, batch_options, mock_solve);

    #[cfg(not(feature = "rayon"))]
    let (solved_infos, batch_stats) = {
        let mut infos: Vec<SubgameInfo> = subset_turns
            .iter()
            .map(|&turn| SubgameInfo::from_boundary(&boundary, 0, &flop, turn, None))
            .collect();
        let solver = BatchSolver::with_options(batch_options);
        let _ = solver.solve_sequential(&mut infos, mock_solve);
        (infos, solver.stats())
    };

    println!("Batch solving completed in {:.2?}", start.elapsed());
    println!("Results: {}", batch_stats);
    println!(
        "All solved: {}",
        solved_infos.iter().all(|i| i.is_solved)
    );

    // -------------------------------------------------------------------------
    // Phase 6: Archive Creation
    // -------------------------------------------------------------------------
    println!("\n--- Phase 6: Archive Format ---");

    // Create manifest
    let manifest = PfsManifest::new(
        PfsConfig {
            turn_buckets: config.turn_buckets,
            river_buckets: config.river_buckets,
            clustering_method: "ehs2".to_string(),
            safe_solving: true,
            blueprint_iterations: blueprint_config.iterations,
            subgame_iterations: 500,
        },
        PfsGameInfo {
            board: flop,
            starting_pot: 100,
            effective_stack: 500,
            oop_range: "66+".to_string(),
            ip_range: "QQ-22".to_string(),
        },
        PfsStats {
            total_subgames: solved_infos.len() as u32,
            solved_subgames: batch_stats.succeeded as u32,
            blueprint_exploitability: 0.01,
            avg_subgame_exploitability: batch_stats.avg_exploitability,
        },
    );

    println!("Manifest version: {}", manifest.version);
    println!("Format: {}", manifest.format);

    // Build archive
    #[cfg(feature = "bincode")]
    {
        let archive_bytes = PfsArchiveBuilder::new()
            .manifest(manifest)
            .blueprint(blueprint)
            .add_subgame(12, None, vec![1, 2, 3, 4]) // Sample subgame data
            .build_bytes()
            .expect("Failed to build archive");

        println!("Archive size: {} bytes", archive_bytes.len());

        // Read archive
        let reader =
            PfsArchiveReader::from_bytes(&archive_bytes).expect("Failed to read archive");
        println!("Archive contains {} subgames", reader.num_subgames());
        println!(
            "Blueprint present: {}",
            reader.blueprint.is_some()
        );
    }

    // -------------------------------------------------------------------------
    // Phase 7: Stitched Game
    // -------------------------------------------------------------------------
    println!("\n--- Phase 7: Stitched Game ---");

    // Create a new blueprint for stitched game
    let blueprint2 = BlueprintBuilder::new()
        .board(flop)
        .config(BlueprintConfig::fast())
        .num_hands(100, 100)
        .compute_abstraction()
        .expect("Failed to compute abstraction")
        .with_boundaries(BoundaryStore::new(100, 100))
        .exploitability(0.01)
        .build()
        .expect("Failed to build blueprint");

    let stitched_config = StitchedGameConfig {
        cache_size: 50,
        enable_on_demand_solving: false,
        collect_stats: true,
        ..Default::default()
    };

    let stitched_game = StitchedGame::with_config(blueprint2, stitched_config);
    println!("Stitched game created for board {:?}", stitched_game.board());

    // Add a precomputed subgame
    let cached_subgame = CachedSubgame {
        info: SubgameInfo::from_boundary(&boundary, 0, &flop, 12, Some(16)),
        strategy: vec![0.5; 100],
        from_cache: false,
    };
    stitched_game.add_precomputed(cached_subgame);
    println!(
        "Precomputed subgames: {}",
        stitched_game.precomputed_count()
    );

    // Check subgame availability
    println!("Has subgame (12, 16): {}", stitched_game.has_subgame(12, Some(16)));
    println!("Has subgame (12, 20): {}", stitched_game.has_subgame(12, Some(20)));

    // Check bucket mappings
    println!("\nBucket mappings for sample cards:");
    for turn in [12, 20, 28, 36, 44] {
        if !flop.contains(&turn) {
            println!(
                "  Turn {} -> bucket {}",
                turn,
                stitched_game.turn_bucket(turn)
            );
        }
    }

    // Get usage stats
    let _ = stitched_game.get_subgame(12, Some(16));
    let _ = stitched_game.get_subgame(12, Some(20));
    let stats = stitched_game.stats();
    println!(
        "\nUsage stats: {} lookups ({} subgame, {} blueprint)",
        stats.total_lookups, stats.subgame_lookups, stats.blueprint_lookups
    );

    println!("\n=== Demo Complete ===");
}
