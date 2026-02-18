//! Bucket inspector for verifying clustering quality.
//!
//! Runs K-means bucketing on a test board and checks whether strategically
//! distinct hands (nut flush draws vs weak flush draws, made hands vs draws)
//! are separated into different clusters.
//!
//! Usage:
//!   cargo run --example inspect_buckets --release

use postflop_solver::*;

// ---------------------------------------------------------------------------
// Card helpers
// ---------------------------------------------------------------------------

fn card(rank: u8, suit: u8) -> u8 {
    rank * 4 + suit
}

const C: u8 = 0; // clubs
const D: u8 = 1; // diamonds
const H: u8 = 2; // hearts
const S: u8 = 3; // spades

fn cstr(c: u8) -> String {
    card_to_string(c).unwrap_or_else(|_| format!("?{}", c))
}

fn combo_str(idx: usize) -> String {
    let (c1, c2) = index_to_card_pair(idx);
    format!("{}{}", cstr(c1), cstr(c2))
}

// ---------------------------------------------------------------------------
// Inline hand analysis (bucketing helpers are private, so we recompute)
// ---------------------------------------------------------------------------

fn check_straight(rank_bits: u16) -> bool {
    for start in 0..9 {
        let mask = 0b11111u16 << start;
        if rank_bits & mask == mask {
            return true;
        }
    }
    let wheel = (1u16 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    rank_bits & wheel == wheel
}

struct HandInfo {
    has_flush_draw: bool,
    is_nut_flush_draw: bool,
    has_made_flush: bool,
    has_made_straight: bool,
    straight_outs: u8,
}

fn analyze_hand(c1: u8, c2: u8, board: &[u8; 4]) -> HandInfo {
    let s1 = c1 & 3;
    let s2 = c2 & 3;

    let mut suit_counts = [0u8; 4];
    let mut hand_suit_counts = [0u8; 4];
    suit_counts[s1 as usize] += 1;
    suit_counts[s2 as usize] += 1;
    hand_suit_counts[s1 as usize] += 1;
    hand_suit_counts[s2 as usize] += 1;
    for &bc in board {
        suit_counts[(bc & 3) as usize] += 1;
    }

    let mut has_flush_draw = false;
    let mut is_nut_flush_draw = false;
    let mut has_made_flush = false;

    for suit in 0..4u8 {
        if hand_suit_counts[suit as usize] == 0 {
            continue;
        }
        let total = suit_counts[suit as usize];
        if total >= 5 {
            has_made_flush = true;
        } else if total == 4 {
            has_flush_draw = true;
            let hand_max = {
                let mut m = -1i8;
                if s1 == suit { m = m.max((c1 >> 2) as i8); }
                if s2 == suit { m = m.max((c2 >> 2) as i8); }
                m
            };
            let mut highest_non_board = -1i8;
            for r in (0u8..13).rev() {
                let card = (r << 2) | suit;
                if !board.contains(&card) {
                    highest_non_board = r as i8;
                    break;
                }
            }
            if hand_max == highest_non_board {
                is_nut_flush_draw = true;
            }
        }
    }

    let mut rank_bits = 0u16;
    for &c in [c1, c2].iter().chain(board.iter()) {
        rank_bits |= 1 << (c >> 2);
    }
    let has_made_straight = check_straight(rank_bits);

    let straight_outs = if has_made_straight {
        0
    } else {
        let mut outs = 0u8;
        for rank in 0u8..13 {
            if rank_bits & (1 << rank) != 0 { continue; }
            if check_straight(rank_bits | (1 << rank)) { outs += 4; }
        }
        outs
    };

    HandInfo { has_flush_draw, is_nut_flush_draw, has_made_flush, has_made_straight, straight_outs }
}

// ---------------------------------------------------------------------------
// Bucket composition analysis
// ---------------------------------------------------------------------------

struct BucketStats {
    size: usize,
    avg_equity: f32,
    min_equity: f32,
    max_equity: f32,
    flush_draws: usize,
    nut_flush_draws: usize,
    made_flushes: usize,
    made_straights: usize,
    straight_draws: usize, // 4+ outs
}

fn analyze_bucket(hands: &[usize], board: &[u8; 4], equity: &[f32; 1326]) -> BucketStats {
    let mut flush_draws = 0;
    let mut nut_flush_draws = 0;
    let mut made_flushes = 0;
    let mut made_straights = 0;
    let mut straight_draws = 0;

    let board_mask: u64 = board.iter().fold(0u64, |acc, &c| acc | (1u64 << c));

    for &idx in hands {
        let (c1, c2) = index_to_card_pair(idx);
        let hand_mask = (1u64 << c1) | (1u64 << c2);
        if hand_mask & board_mask != 0 { continue; }

        let info = analyze_hand(c1, c2, board);
        if info.has_flush_draw { flush_draws += 1; }
        if info.is_nut_flush_draw { nut_flush_draws += 1; }
        if info.has_made_flush { made_flushes += 1; }
        if info.has_made_straight { made_straights += 1; }
        if info.straight_outs >= 4 { straight_draws += 1; }
    }

    let equities: Vec<f32> = hands.iter().map(|&i| equity[i]).collect();
    let avg = equities.iter().sum::<f32>() / equities.len().max(1) as f32;
    let min = equities.iter().cloned().fold(f32::MAX, f32::min);
    let max = equities.iter().cloned().fold(f32::MIN, f32::max);

    BucketStats {
        size: hands.len(),
        avg_equity: avg,
        min_equity: min,
        max_equity: max,
        flush_draws,
        nut_flush_draws,
        made_flushes,
        made_straights,
        straight_draws,
    }
}

fn print_stats(stats: &BucketStats) {
    println!("  Equity: avg={:.3}, min={:.3}, max={:.3}", stats.avg_equity, stats.min_equity, stats.max_equity);
    println!(
        "  Composition: {} FDs ({} nut), {} made flushes, {} made straights, {} str draws",
        stats.flush_draws, stats.nut_flush_draws, stats.made_flushes,
        stats.made_straights, stats.straight_draws
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Board: Ks Ts 2d 5h — two spades, disconnected low cards
    // Key scenarios: flush draws, straight draws, top pair blockers
    let board: [u8; 4] = [
        card(11, S), // Ks
        card(8, S),  // Ts
        card(0, D),  // 2d
        card(3, H),  // 5h
    ];

    println!("=== Bucket Inspector ===");
    println!("Board: {} {} {} {}", cstr(board[0]), cstr(board[1]), cstr(board[2]), cstr(board[3]));
    println!();

    let equity = compute_equity(&board);

    let k = 50;
    println!("Clustering into {} buckets...", k);
    let mapping = compute_buckets(&board, k);
    println!("Active buckets: {}, Valid hands: {}\n", mapping.k, mapping.num_valid);

    // Group hands by bucket
    let mut buckets: Vec<Vec<usize>> = vec![vec![]; mapping.k];
    for combo_idx in 0..1326 {
        let b = mapping.hand_to_bucket[combo_idx];
        if b != BLOCKED_BUCKET {
            buckets[b as usize].push(combo_idx);
        }
    }

    // Define archetype hands: (c1, c2, label)
    let archetypes: Vec<(u8, u8, &str)> = vec![
        (card(12, S), card(10, S), "AsQs  — Nut Flush Draw"),
        (card(9, S),  card(7, S),  "Js9s  — Weak Flush Draw"),
        (card(12, S), card(9, S),  "AsJs  — Nut Flush Draw (lower)"),
        (card(10, C), card(9, D),  "QcJd  — OESD (no flush)"),
        (card(12, S), card(11, D), "AsKd  — Top Pair + NFD Blocker"),
        (card(12, C), card(11, D), "AcKd  — Top Pair, No Blocker"),
        (card(12, H), card(12, D), "AhAd  — Overpair (AA)"),
        (card(3, C),  card(1, C),  "5c3c  — Bottom Pair (weak)"),
    ];

    for (c1, c2, label) in &archetypes {
        let idx = card_pair_to_index(*c1, *c2);
        let b = mapping.hand_to_bucket[idx];
        if b == BLOCKED_BUCKET {
            println!("--- {} => BLOCKED ---\n", label);
            continue;
        }
        let b_id = b as usize;
        let eq = equity[idx];
        let info = analyze_hand(*c1, *c2, &board);

        println!("--- {} => Bucket {} (equity: {:.3}) ---", label, b_id, eq);
        println!("  Hand flags: FD={} NFD={} flush={} straight={} str_outs={}",
            info.has_flush_draw as u8, info.is_nut_flush_draw as u8,
            info.has_made_flush as u8, info.has_made_straight as u8, info.straight_outs);

        let neighbors = &buckets[b_id];
        // Show first 12 neighbors
        print!("  Neighbors: ");
        for &n_idx in neighbors.iter().take(12) {
            print!("{}({:.2}) ", combo_str(n_idx), equity[n_idx]);
        }
        if neighbors.len() > 12 {
            print!("... +{}", neighbors.len() - 12);
        }
        println!();

        let stats = analyze_bucket(neighbors, &board, &equity);
        print_stats(&stats);
        println!();
    }

    // ===================================================================
    // Verification tests
    // ===================================================================
    println!("============================================");
    println!("           VERIFICATION TESTS");
    println!("============================================\n");

    let mut pass = 0;
    let mut fail = 0;
    let mut note = 0;

    // Test 1: Nut FD vs Weak FD
    {
        let nfd_idx = card_pair_to_index(card(12, S), card(10, S)); // AsQs
        let wfd_idx = card_pair_to_index(card(9, S), card(7, S));   // Js9s
        let nfd_b = mapping.hand_to_bucket[nfd_idx];
        let wfd_b = mapping.hand_to_bucket[wfd_idx];

        print!("Test 1 — Nut FD (AsQs) vs Weak FD (Js9s): ");
        if nfd_b != wfd_b {
            println!("PASS (buckets {} vs {})", nfd_b, wfd_b);
            pass += 1;
        } else {
            println!("FAIL — same bucket {}", nfd_b);
            fail += 1;
        }
    }

    // Test 2: Nut FD cluster purity — AsQs neighbors should mostly be nut FDs
    {
        let nfd_idx = card_pair_to_index(card(12, S), card(10, S));
        let b_id = mapping.hand_to_bucket[nfd_idx] as usize;
        let stats = analyze_bucket(&buckets[b_id], &board, &equity);

        print!("Test 2 — Nut FD bucket purity: ");
        if stats.size > 0 && stats.nut_flush_draws as f32 / stats.size as f32 > 0.3 {
            println!("PASS ({}/{} hands are NFDs, {:.0}%)",
                stats.nut_flush_draws, stats.size,
                stats.nut_flush_draws as f32 / stats.size as f32 * 100.0);
            pass += 1;
        } else {
            println!("FAIL (only {}/{} are NFDs)", stats.nut_flush_draws, stats.size);
            fail += 1;
        }
    }

    // Test 3: OESD not mixed with top pair
    {
        let oesd_idx = card_pair_to_index(card(10, C), card(9, D)); // QcJd
        let tp_idx = card_pair_to_index(card(12, C), card(11, D));  // AcKd
        let oesd_b = mapping.hand_to_bucket[oesd_idx];
        let tp_b = mapping.hand_to_bucket[tp_idx];

        print!("Test 3 — OESD (QcJd) vs Top Pair (AcKd): ");
        if oesd_b != tp_b {
            println!("PASS (buckets {} vs {})", oesd_b, tp_b);
            pass += 1;
        } else {
            println!("FAIL — same bucket {}", oesd_b);
            fail += 1;
        }
    }

    // Test 4: Blocker effect — AsKd vs AcKd
    {
        let blocker_idx = card_pair_to_index(card(12, S), card(11, D)); // AsKd
        let no_block_idx = card_pair_to_index(card(12, C), card(11, D)); // AcKd
        let blocker_b = mapping.hand_to_bucket[blocker_idx];
        let no_block_b = mapping.hand_to_bucket[no_block_idx];

        print!("Test 4 — Blocker (AsKd) vs No-Blocker (AcKd): ");
        if blocker_b != no_block_b {
            println!("PASS (buckets {} vs {})", blocker_b, no_block_b);
            pass += 1;
        } else {
            println!("NOTE — same bucket {} (may be OK at k={})", blocker_b, k);
            note += 1;
        }
    }

    // Test 5: AA should not be mixed with weak hands
    {
        let aa_idx = card_pair_to_index(card(12, H), card(12, D)); // AhAd
        let weak_idx = card_pair_to_index(card(3, C), card(1, C)); // 5c3c
        let aa_b = mapping.hand_to_bucket[aa_idx];
        let weak_b = mapping.hand_to_bucket[weak_idx];

        print!("Test 5 — AA vs Bottom Pair (5c3c): ");
        if aa_b != weak_b {
            println!("PASS (buckets {} vs {})", aa_b, weak_b);
            pass += 1;
        } else {
            println!("FAIL — same bucket {}", aa_b);
            fail += 1;
        }
    }

    // Test 6: Multiple nut FDs should cluster together (AsQs, AsJs)
    {
        let nfd1_idx = card_pair_to_index(card(12, S), card(10, S)); // AsQs
        let nfd2_idx = card_pair_to_index(card(12, S), card(9, S));  // AsJs
        let nfd1_b = mapping.hand_to_bucket[nfd1_idx];
        let nfd2_b = mapping.hand_to_bucket[nfd2_idx];

        print!("Test 6 — Nut FDs cluster (AsQs vs AsJs): ");
        if nfd1_b == nfd2_b {
            println!("PASS (both in bucket {})", nfd1_b);
            pass += 1;
        } else {
            // Not a hard fail — different kickers may split them
            println!("NOTE — different buckets ({} vs {}), kicker split", nfd1_b, nfd2_b);
            note += 1;
        }
    }

    println!("\n============================================");
    println!("Results: {} PASS, {} FAIL, {} NOTE", pass, fail, note);
    if fail == 0 {
        println!(">>> CLEARED for data generation <<<");
    } else {
        println!(">>> FIX clustering before generating data <<<");
    }
    println!("============================================");

    // ===================================================================
    // Run same tests at k=1000 (production setting)
    // ===================================================================
    println!("\n\n=== Re-running at k=1000 (production) ===\n");

    let mapping_1k = compute_buckets(&board, 1000);
    println!("Active buckets: {}, Valid hands: {}\n", mapping_1k.k, mapping_1k.num_valid);

    let mut buckets_1k: Vec<Vec<usize>> = vec![vec![]; mapping_1k.k];
    for combo_idx in 0..1326 {
        let b = mapping_1k.hand_to_bucket[combo_idx];
        if b != BLOCKED_BUCKET {
            buckets_1k[b as usize].push(combo_idx);
        }
    }

    // Key checks at k=1000
    let checks: Vec<(&str, u8, u8, u8, u8)> = vec![
        ("NFD vs Weak FD", card(12, S), card(10, S), card(9, S), card(7, S)),
        ("Blocker vs No-Blocker", card(12, S), card(11, D), card(12, C), card(11, D)),
    ];

    for (label, c1a, c2a, c1b, c2b) in checks {
        let idx_a = card_pair_to_index(c1a, c2a);
        let idx_b = card_pair_to_index(c1b, c2b);
        let ba = mapping_1k.hand_to_bucket[idx_a];
        let bb = mapping_1k.hand_to_bucket[idx_b];

        print!("{}: ", label);
        if ba != bb {
            println!("SEPARATED (buckets {} vs {})", ba, bb);
        } else {
            println!("SAME bucket {}", ba);
        }
    }

    // NFD bucket purity at k=1000
    {
        let nfd_idx = card_pair_to_index(card(12, S), card(10, S));
        let b_id = mapping_1k.hand_to_bucket[nfd_idx] as usize;
        let stats = analyze_bucket(&buckets_1k[b_id], &board, &equity);
        println!("\nNut FD bucket {} at k=1000:", b_id);
        println!("  Size: {} hands", stats.size);
        print_stats(&stats);

        // Print all hands in this bucket (should be small at k=1000)
        if stats.size <= 20 {
            print!("  All hands: ");
            for &idx in &buckets_1k[b_id] {
                print!("{} ", combo_str(idx));
            }
            println!();
        }
    }
}
