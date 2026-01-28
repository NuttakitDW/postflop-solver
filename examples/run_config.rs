use postflop_solver::*;
use serde::Deserialize;
use std::fs;
use std::time::Instant;

#[derive(Deserialize)]
struct Config {
    game: GameConfig,
    board: BoardConfig,
    ranges: RangesConfig,
    bet_sizes: BetSizesConfig,
    tree_building: TreeBuildingConfig,
    solver: SolverConfig,
    output: OutputConfig,
}

#[derive(Deserialize)]
struct GameConfig {
    starting_pot: i32,
    effective_stack: i32,
    rake_rate: f64,
    rake_cap: f64,
}

#[derive(Deserialize)]
struct BoardConfig {
    flop: String,
    turn: String,
    river: String,
}

#[derive(Deserialize)]
struct RangesConfig {
    oop: String,
    ip: String,
}

#[derive(Deserialize)]
struct BetSizesConfig {
    flop: StreetBetSizes,
    turn: StreetBetSizes,
    river: StreetBetSizes,
    donk: DonkConfig,
}

#[derive(Deserialize)]
struct StreetBetSizes {
    oop: PlayerBetSizes,
    ip: PlayerBetSizes,
}

#[derive(Deserialize)]
struct PlayerBetSizes {
    bet: String,
    raise: String,
}

#[derive(Deserialize)]
struct DonkConfig {
    turn: Option<String>,
    river: Option<String>,
}

#[derive(Deserialize)]
struct TreeBuildingConfig {
    add_allin_threshold: f64,
    force_allin_threshold: f64,
    merging_threshold: f64,
}

#[derive(Deserialize)]
struct SolverConfig {
    max_iterations: u32,
    target_exploitability_percent: f64,
    use_compression: bool,
    print_progress: bool,
}

#[derive(Deserialize)]
struct OutputConfig {
    save_file: String,
    print_strategy: bool,
    print_ev: bool,
}

fn main() {
    // Get config file from command line args
    let args: Vec<String> = std::env::args().collect();
    let config_path = if args.len() > 1 {
        &args[1]
    } else {
        "template.json"
    };

    println!("=== Postflop Solver ===");
    println!("Loading config: {}", config_path);
    println!();

    // Load and parse config
    let config_str = fs::read_to_string(config_path)
        .expect(&format!("Failed to read config file: {}", config_path));
    let config: Config = serde_json::from_str(&config_str)
        .expect("Failed to parse config JSON");

    // Determine initial state based on board cards
    let initial_state = if !config.board.river.is_empty() {
        BoardState::River
    } else if !config.board.turn.is_empty() {
        BoardState::Turn
    } else {
        BoardState::Flop
    };

    println!("=== Configuration ===");
    println!("Initial state: {:?}", initial_state);
    println!("Starting pot: {}", config.game.starting_pot);
    println!("Effective stack: {}", config.game.effective_stack);
    println!("Board: {} {} {}", config.board.flop, config.board.turn, config.board.river);
    println!();

    // Parse ranges
    let oop_range: Range = config.ranges.oop.parse()
        .expect("Failed to parse OOP range");
    let ip_range: Range = config.ranges.ip.parse()
        .expect("Failed to parse IP range");

    // Build card config
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str(&config.board.flop).expect("Invalid flop"),
        turn: if config.board.turn.is_empty() {
            NOT_DEALT
        } else {
            card_from_str(&config.board.turn).expect("Invalid turn card")
        },
        river: if config.board.river.is_empty() {
            NOT_DEALT
        } else {
            card_from_str(&config.board.river).expect("Invalid river card")
        },
    };

    // Parse bet sizes
    let flop_oop = BetSizeOptions::try_from((
        config.bet_sizes.flop.oop.bet.as_str(),
        config.bet_sizes.flop.oop.raise.as_str()
    )).expect("Invalid flop OOP bet sizes");

    let flop_ip = BetSizeOptions::try_from((
        config.bet_sizes.flop.ip.bet.as_str(),
        config.bet_sizes.flop.ip.raise.as_str()
    )).expect("Invalid flop IP bet sizes");

    let turn_oop = BetSizeOptions::try_from((
        config.bet_sizes.turn.oop.bet.as_str(),
        config.bet_sizes.turn.oop.raise.as_str()
    )).expect("Invalid turn OOP bet sizes");

    let turn_ip = BetSizeOptions::try_from((
        config.bet_sizes.turn.ip.bet.as_str(),
        config.bet_sizes.turn.ip.raise.as_str()
    )).expect("Invalid turn IP bet sizes");

    let river_oop = BetSizeOptions::try_from((
        config.bet_sizes.river.oop.bet.as_str(),
        config.bet_sizes.river.oop.raise.as_str()
    )).expect("Invalid river OOP bet sizes");

    let river_ip = BetSizeOptions::try_from((
        config.bet_sizes.river.ip.bet.as_str(),
        config.bet_sizes.river.ip.raise.as_str()
    )).expect("Invalid river IP bet sizes");

    let turn_donk = config.bet_sizes.donk.turn.as_ref().map(|s| {
        DonkSizeOptions::try_from(s.as_str()).expect("Invalid turn donk sizes")
    });

    let river_donk = config.bet_sizes.donk.river.as_ref().map(|s| {
        DonkSizeOptions::try_from(s.as_str()).expect("Invalid river donk sizes")
    });

    // Build tree config
    let tree_config = TreeConfig {
        initial_state,
        starting_pot: config.game.starting_pot,
        effective_stack: config.game.effective_stack,
        rake_rate: config.game.rake_rate,
        rake_cap: config.game.rake_cap,
        flop_bet_sizes: [flop_oop, flop_ip],
        turn_bet_sizes: [turn_oop, turn_ip],
        river_bet_sizes: [river_oop, river_ip],
        turn_donk_sizes: turn_donk,
        river_donk_sizes: river_donk,
        add_allin_threshold: config.tree_building.add_allin_threshold,
        force_allin_threshold: config.tree_building.force_allin_threshold,
        merging_threshold: config.tree_building.merging_threshold,
    };

    // Build action tree
    println!("Building action tree...");
    let tree_start = Instant::now();
    let action_tree = ActionTree::new(tree_config).expect("Failed to create action tree");
    println!("Tree built in {:.3}s", tree_start.elapsed().as_secs_f64());

    // Create game
    println!("Creating game...");
    let game_start = Instant::now();
    let mut game = PostFlopGame::with_config(card_config, action_tree)
        .expect("Failed to create game");
    println!("Game created in {:.3}s", game_start.elapsed().as_secs_f64());
    println!();

    // Print game info
    println!("=== Game Info ===");
    println!("OOP hands: {}", game.private_cards(0).len());
    println!("IP hands: {}", game.private_cards(1).len());
    let (mem_usage, mem_compressed) = game.memory_usage();
    println!("Memory (uncompressed): {:.2} MB", mem_usage as f64 / 1024.0 / 1024.0);
    println!("Memory (compressed): {:.2} MB", mem_compressed as f64 / 1024.0 / 1024.0);
    println!();

    // Allocate memory
    println!("Allocating memory (compression={})...", config.solver.use_compression);
    let alloc_start = Instant::now();
    game.allocate_memory(config.solver.use_compression);
    println!("Memory allocated in {:.3}s", alloc_start.elapsed().as_secs_f64());
    println!();

    // Solve
    let target_exploitability = game.tree_config().starting_pot as f32
        * (config.solver.target_exploitability_percent as f32 / 100.0);

    println!("=== Solving ===");
    println!("Max iterations: {}", config.solver.max_iterations);
    println!("Target exploitability: {:.4} ({:.2}% of pot)",
        target_exploitability, config.solver.target_exploitability_percent);
    println!();

    let solve_start = Instant::now();
    let exploitability = solve(
        &mut game,
        config.solver.max_iterations,
        target_exploitability,
        config.solver.print_progress,
    );
    let solve_time = solve_start.elapsed();

    println!();
    println!("=== Results ===");
    println!("Solve time: {:.2}s", solve_time.as_secs_f64());
    println!("Final exploitability: {:.4}", exploitability);
    println!("Exploitability as % of pot: {:.3}%",
        exploitability / game.tree_config().starting_pot as f32 * 100.0);

    // Print strategy at root if requested
    if config.output.print_strategy {
        println!();
        println!("=== Root Strategy ===");
        print_root_strategy(&game);
    }

    // Print EV if requested
    if config.output.print_ev {
        println!();
        println!("=== Expected Values ===");
        game.cache_normalized_weights();
        print_ev_summary(&game);
    }

    // Save file if specified
    if !config.output.save_file.is_empty() {
        println!();
        println!("Saving to: {}", config.output.save_file);

        // Create memo string with config info
        let memo = format!(
            "Board: {} {} {} | Pot: {} | Stack: {} | Exploitability: {:.4}",
            config.board.flop,
            config.board.turn,
            config.board.river,
            config.game.starting_pot,
            config.game.effective_stack,
            exploitability
        );

        // Use compression level 3 if zstd is available
        #[cfg(feature = "zstd")]
        let compression = Some(3);
        #[cfg(not(feature = "zstd"))]
        let compression = None;

        match save_data_to_file(&game, &memo, &config.output.save_file, compression) {
            Ok(_) => {
                println!("File saved successfully!");
                println!("Memo: {}", memo);

                // Print file size
                if let Ok(metadata) = std::fs::metadata(&config.output.save_file) {
                    let size_mb = metadata.len() as f64 / 1024.0 / 1024.0;
                    println!("File size: {:.2} MB", size_mb);
                }
            }
            Err(e) => {
                println!("Error saving file: {}", e);
            }
        }
    }
}

fn print_root_strategy(game: &PostFlopGame) {
    let actions = game.available_actions();
    println!("Available actions: {:?}", actions);

    let strategy = game.strategy();
    let num_hands = game.private_cards(0).len();
    let num_actions = actions.len();

    // Print action frequencies (average across all hands)
    println!();
    println!("Average action frequencies:");
    for (i, action) in actions.iter().enumerate() {
        let mut sum = 0.0;
        for h in 0..num_hands {
            sum += strategy[i * num_hands + h];
        }
        let avg = sum / num_hands as f32;
        println!("  {:?}: {:.1}%", action, avg * 100.0);
    }
}

fn print_ev_summary(game: &PostFlopGame) {
    let weights_oop = game.normalized_weights(0);
    let weights_ip = game.normalized_weights(1);

    let equity_oop = game.equity(0);
    let equity_ip = game.equity(1);

    let ev_oop = game.expected_values(0);
    let ev_ip = game.expected_values(1);

    let avg_equity_oop = compute_average(&equity_oop, weights_oop);
    let avg_equity_ip = compute_average(&equity_ip, weights_ip);

    let avg_ev_oop = compute_average(&ev_oop, weights_oop);
    let avg_ev_ip = compute_average(&ev_ip, weights_ip);

    println!("OOP - Equity: {:.1}%, EV: {:.2}", avg_equity_oop * 100.0, avg_ev_oop);
    println!("IP  - Equity: {:.1}%, EV: {:.2}", avg_equity_ip * 100.0, avg_ev_ip);
}
