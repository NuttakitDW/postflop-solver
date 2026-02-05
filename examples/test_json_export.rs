use postflop_solver::*;

fn main() {
    // Load a game file
    let file_path = "game.flop";
    println!("Loading game from: {}", file_path);

    let (mut game, memo): (PostFlopGame, String) = load_data_from_file(file_path, None)
        .expect("Failed to load game file");

    println!("Game loaded successfully!");
    println!("Memo: {}", memo);
    println!("Is solved: {}", game.is_solved());

    if game.is_solved() {
        // Cache normalized weights for equity and EV calculation
        game.cache_normalized_weights();
        println!("Normalized weights cached");

        // Export to JSON
        println!("\nExporting to JSON...");
        let json_value = game.to_json_value().expect("Failed to export to JSON");

        // Check if hand_data contains the new fields
        if let Some(hand_data) = json_value.get("hand_data") {
            println!("\nHand data fields:");
            println!("- oop_private_cards: {}", hand_data.get("oop_private_cards").is_some());
            println!("- ip_private_cards: {}", hand_data.get("ip_private_cards").is_some());
            println!("- oop_equity: {}", hand_data.get("oop_equity").is_some());
            println!("- ip_equity: {}", hand_data.get("ip_equity").is_some());
            println!("- oop_ev: {}", hand_data.get("oop_ev").is_some());
            println!("- ip_ev: {}", hand_data.get("ip_ev").is_some());
            println!("- oop_eqr: {}", hand_data.get("oop_eqr").is_some());
            println!("- ip_eqr: {}", hand_data.get("ip_eqr").is_some());

            // Show sample data
            if let Some(oop_equity) = hand_data.get("oop_equity") {
                if let Some(arr) = oop_equity.as_array() {
                    println!("\nFirst 5 OOP equity values:");
                    for (i, val) in arr.iter().take(5).enumerate() {
                        println!("  Hand {}: {}", i, val);
                    }
                }
            }

            if let Some(oop_ev) = hand_data.get("oop_ev") {
                if let Some(arr) = oop_ev.as_array() {
                    println!("\nFirst 5 OOP EV values:");
                    for (i, val) in arr.iter().take(5).enumerate() {
                        println!("  Hand {}: {}", i, val);
                    }
                }
            }

            if let Some(oop_eqr) = hand_data.get("oop_eqr") {
                if let Some(arr) = oop_eqr.as_array() {
                    println!("\nFirst 5 OOP EQR values:");
                    for (i, val) in arr.iter().take(5).enumerate() {
                        println!("  Hand {}: {}", i, val);
                    }
                }
            }
        }

        // Write to file
        let output_path = "game-complete-test.json";
        println!("\nWriting to file: {}", output_path);
        let file = std::fs::File::create(output_path).expect("Failed to create file");
        serde_json::to_writer_pretty(file, &json_value).expect("Failed to write JSON");
        println!("JSON export completed successfully!");
    } else {
        println!("Game is not solved. Equity and EV data will not be available.");
    }
}
