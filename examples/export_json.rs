use postflop_solver::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input.flop> <output.json>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} game.flop game-complete.json", args[0]);
        eprintln!();
        eprintln!("This will export a complete JSON file including:");
        eprintln!("  - Combos (hand combinations)");
        eprintln!("  - Equity (win probability for each hand)");
        eprintln!("  - EV (expected value for each hand)");
        eprintln!("  - EQR (equity realization for each hand)");
        eprintln!("  - All node strategies and cfvalues");
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           PostFlop Solver - Complete JSON Export            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Input file:  {}", input_file);
    println!("Output file: {}", output_file);
    println!();

    // Load the game file
    print!("⏳ Loading game file... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let load_start = std::time::Instant::now();
    let (mut game, memo): (PostFlopGame, String) = match load_data_from_file(input_file, None) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("\n❌ Failed to load game file: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ Done ({:.2}s)", load_start.elapsed().as_secs_f64());

    // Display game info
    println!();
    println!("Game Information:");
    println!("  Memo:         {}", if memo.is_empty() { "(empty)" } else { &memo });
    println!("  Is solved:    {}", if game.is_solved() { "✓ Yes" } else { "✗ No" });
    println!("  Storage mode: {}", match game.storage_mode() {
        BoardState::Flop => "Flop",
        BoardState::Turn => "Turn",
        BoardState::River => "River",
    });

    let card_config = game.card_config();
    print!("  Board:        ");
    for &card in &card_config.flop {
        print!("{} ", card);
    }
    if card_config.turn != NOT_DEALT {
        print!("{} ", card_config.turn);
    }
    if card_config.river != NOT_DEALT {
        print!("{} ", card_config.river);
    }
    println!();

    let num_oop_hands = game.private_cards(0).len();
    let num_ip_hands = game.private_cards(1).len();
    println!("  OOP hands:    {}", num_oop_hands);
    println!("  IP hands:     {}", num_ip_hands);

    let total_nodes: usize = game.num_nodes().iter().map(|&x| x as usize).sum();
    println!("  Total nodes:  {}", total_nodes);

    if game.is_solved() {
        // Cache normalized weights for equity and EV calculation
        print!("\n⏳ Caching normalized weights... ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let cache_start = std::time::Instant::now();
        game.cache_normalized_weights();
        println!("✓ Done ({:.2}s)", cache_start.elapsed().as_secs_f64());

        // Show sample equity and EV data
        println!("\nSample Hand Data:");
        let oop_equity = game.equity(0);
        let oop_ev = game.expected_values(0);
        let ip_equity = game.equity(1);
        let ip_ev = game.expected_values(1);

        println!("  OOP Sample:");
        for i in 0..3.min(num_oop_hands) {
            let cards = game.private_cards(0)[i];
            let eqr = if oop_equity[i] > 0.01 {
                (oop_ev[i] / oop_equity[i]) * 100.0
            } else {
                0.0
            };
            println!("    Hand {}: [{}, {}] | Equity: {:.2}% | EV: {:.2} | EQR: {:.2}%",
                i, cards.0, cards.1, oop_equity[i] * 100.0, oop_ev[i], eqr);
        }

        println!("  IP Sample:");
        for i in 0..3.min(num_ip_hands) {
            let cards = game.private_cards(1)[i];
            let eqr = if ip_equity[i] > 0.01 {
                (ip_ev[i] / ip_equity[i]) * 100.0
            } else {
                0.0
            };
            println!("    Hand {}: [{}, {}] | Equity: {:.2}% | EV: {:.2} | EQR: {:.2}%",
                i, cards.0, cards.1, ip_equity[i] * 100.0, ip_ev[i], eqr);
        }
    }

    // Export to JSON
    print!("\n⏳ Exporting to JSON... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let export_start = std::time::Instant::now();
    let json_value = match game.to_json_value() {
        Ok(value) => value,
        Err(e) => {
            eprintln!("\n❌ Failed to export to JSON: {}", e);
            std::process::exit(1);
        }
    };
    println!("✓ Done ({:.2}s)", export_start.elapsed().as_secs_f64());

    // Write to file
    print!("⏳ Writing to file... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let write_start = std::time::Instant::now();
    let file = match std::fs::File::create(output_file) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("\n❌ Failed to create output file: {}", e);
            std::process::exit(1);
        }
    };

    if let Err(e) = serde_json::to_writer_pretty(file, &json_value) {
        eprintln!("\n❌ Failed to write JSON: {}", e);
        std::process::exit(1);
    }
    println!("✓ Done ({:.2}s)", write_start.elapsed().as_secs_f64());

    // Show file size
    if let Ok(metadata) = std::fs::metadata(output_file) {
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("\n✓ Export completed successfully!");
        println!("  Output file: {}", output_file);
        println!("  File size:   {:.2} MB", size_mb);
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║ JSON Export Fields Available:                               ║");
    println!("║   • oop_private_cards - OOP hand combinations                ║");
    println!("║   • ip_private_cards  - IP hand combinations                 ║");
    println!("║   • oop_equity        - OOP equity for each hand             ║");
    println!("║   • ip_equity         - IP equity for each hand              ║");
    println!("║   • oop_ev            - OOP expected value for each hand     ║");
    println!("║   • ip_ev             - IP expected value for each hand      ║");
    println!("║   • oop_eqr           - OOP equity realization for each hand ║");
    println!("║   • ip_eqr            - IP equity realization for each hand  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
