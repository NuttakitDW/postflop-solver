//! Debug: compare oracle DCFR against the oracle source tree's own strategies.
//!
//! The oracle is extracted from a fully solved tree. The oracle DCFR should
//! converge to the same flop strategies as that source tree.
//!
//! cargo run --example debug_oracle --release --features "bincode rayon" -- config/C-oracle-3.json

#[path = "common/mod.rs"]
mod common;
#[path = "poc_precompute/oracle_solver.rs"]
mod oracle_solver;

use common::*;
use oracle_solver::TreeOracle;
use postflop_solver::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let config_path = &args[1];
    let config = load_config(config_path);
    let (card_config, tree_config) = parse_configs(&config).unwrap();

    let starting_pot = tree_config.starting_pot as f32;
    let target = starting_pot * config.solver.target_exploitability_percent / 100.0;
    let max_iter = config.solver.max_iterations;

    println!("=== Compare: oracle source vs oracle DCFR ===");
    println!("Board: {}, Pot: {}, Stack: {}, Target: {:.4} ({:.2}%)",
        config.board.flop, tree_config.starting_pot, tree_config.effective_stack,
        target, config.solver.target_exploitability_percent);
    println!();

    // STEP 1: Solve full tree (no compression) — this is the oracle source
    println!("STEP 1: Solve full tree (oracle source, no compression)...");
    let action_tree = ActionTree::new(tree_config.clone()).unwrap();
    let mut source_game = PostFlopGame::with_config(card_config.clone(), action_tree).unwrap();
    source_game.allocate_memory(false);
    let source_exploit = solve(&mut source_game, max_iter, target, true);
    println!("  Source exploitability: {:.4} ({:.4}%)", source_exploit, source_exploit / starting_pot * 100.0);

    // Extract oracle from the source tree
    println!();
    println!("STEP 2: Extract oracle from source tree...");
    let oracle = TreeOracle::build(&source_game, true);

    // STEP 3: Test multiple warmup lengths
    let warmup_tests = [64u32, 128, 160];
    let mut hybrid_games: Vec<PostFlopGame> = Vec::new();

    for &warmup in &warmup_tests {
        println!();
        println!("STEP 3: Hybrid solver (warmup={} + oracle)...", warmup);
        let at_h = ActionTree::new(tree_config.clone()).unwrap();
        let mut hgame = PostFlopGame::with_config(card_config.clone(), at_h).unwrap();
        hgame.allocate_memory(false);
        let hexploit = oracle_solver::solve_hybrid(
            &mut hgame,
            &oracle,
            warmup,
            max_iter,
            target,
            true,
        );
        println!("  Hybrid-{} exploitability: {:.4} ({:.4}%)", warmup, hexploit, hexploit / starting_pot * 100.0);

        let mut stats = CompareStats::default();
        walk_flop_tree(&source_game, &hgame, &mut source_game.root(), &mut hgame.root(), &mut stats);
        println!("  Hybrid-{}: nodes={} avg_diff={:.6} max_diff={:.6} >5%: {} ({:.1}%) >10%: {} ({:.1}%)",
            warmup, stats.nodes,
            stats.sum_diff / stats.elements as f64, stats.max_diff,
            stats.gt_5pct, stats.gt_5pct as f64 / stats.elements as f64 * 100.0,
            stats.gt_10pct, stats.gt_10pct as f64 / stats.elements as f64 * 100.0);

        hybrid_games.push(hgame);
    }

    let hybrid_game = &hybrid_games[0]; // Use the first (warmup=64) for detailed comparison

    // STEP 4: Detailed comparison for warmup=64
    println!();
    println!("=== STEP 4: Compare flop strategies (source vs hybrid-64) ===");

    let num_hands = [source_game.num_private_hands(0), source_game.num_private_hands(1)];
    let private_cards = [
        source_game.private_cards(0).to_vec(),
        source_game.private_cards(1).to_vec(),
    ];

    // Root comparison (warmup=64)
    compare_node(
        "ROOT",
        &source_game.root(),
        &hybrid_game.root(),
        num_hands,
        &private_cards,
    );

    // Children of root
    let src_root = source_game.root();
    let hybrid_root = hybrid_game.root();
    for a in 0..src_root.num_actions() {
        let src_child = src_root.play(a);
        let hybrid_child = hybrid_root.play(a);
        if !src_child.is_terminal() && !src_child.is_chance() {
            compare_node(
                &format!("ROOT->a{}", a),
                &src_child,
                &hybrid_child,
                num_hands,
                &private_cards,
            );
        }
    }

    // STEP 6: Also run pure oracle DCFR for comparison
    println!();
    println!("=== STEP 6: Pure oracle DCFR (for comparison) ===");
    let at_ora = ActionTree::new(tree_config.clone()).unwrap();
    let mut ora_game = PostFlopGame::with_config(card_config.clone(), at_ora).unwrap();
    ora_game.allocate_memory(false);
    let ora_exploit = oracle_solver::solve_flop_fixed_iterations(
        &ora_game,
        &oracle,
        max_iter,
        target,
        true,
    );
    println!("  Pure oracle exploitability: {:.4} ({:.4}%)", ora_exploit, ora_exploit / starting_pot * 100.0);
    let mut stats_ora = CompareStats::default();
    walk_flop_tree(&source_game, &ora_game, &mut source_game.root(), &mut ora_game.root(), &mut stats_ora);
    println!("  Pure oracle: nodes={} max_diff={:.6} avg_diff={:.6} >5%: {} ({:.1}%)",
        stats_ora.nodes, stats_ora.max_diff, stats_ora.sum_diff / stats_ora.elements as f64,
        stats_ora.gt_5pct, stats_ora.gt_5pct as f64 / stats_ora.elements as f64 * 100.0);

    // STEP 7: Variability check (two standard solves)
    println!();
    println!("=== STEP 7: Variability check (two standard solves) ===");
    let at2 = ActionTree::new(tree_config.clone()).unwrap();
    let mut game2 = PostFlopGame::with_config(card_config.clone(), at2).unwrap();
    game2.allocate_memory(false);
    let exploit2 = solve(&mut game2, max_iter, target, true);
    println!("  Second solve exploitability: {:.4} ({:.4}%)", exploit2, exploit2 / starting_pot * 100.0);

    compare_node(
        "STD1_vs_STD2",
        &source_game.root(),
        &game2.root(),
        num_hands,
        &private_cards,
    );

    let mut stats2 = CompareStats::default();
    walk_flop_tree(&source_game, &game2, &mut source_game.root(), &mut game2.root(), &mut stats2);
    println!("  Full tree: nodes={} max_diff={:.6} avg_diff={:.6} >5%: {} ({:.1}%)",
        stats2.nodes, stats2.max_diff, stats2.sum_diff / stats2.elements as f64,
        stats2.gt_5pct, stats2.gt_5pct as f64 / stats2.elements as f64 * 100.0);
}

fn compare_node(
    label: &str,
    src: &PostFlopNode,
    ora: &PostFlopNode,
    num_hands: [usize; 2],
    private_cards: &[Vec<(u8, u8)>; 2],
) {
    let player = src.player();
    let nh = num_hands[player];
    let na = src.num_actions();
    let src_s = normalize(src.strategy(), na, nh);
    let ora_s = normalize(ora.strategy(), na, nh);

    let max_d = src_s.iter().zip(&ora_s).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let avg_d: f64 = src_s.iter().zip(&ora_s).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / src_s.len() as f64;
    let gt5 = src_s.iter().zip(&ora_s).filter(|(a, b)| ((*a) - (*b)).abs() > 0.05).count();

    let pname = if player == 0 { "OOP" } else { "IP" };
    println!("  {} ({} a={} h={}): max={:.4} avg={:.4} >5%: {}/{}",
        label, pname, na, nh, max_d, avg_d, gt5, src_s.len());

    if max_d > 0.1 {
        let pc = &private_cards[player];
        let mut diffs: Vec<(usize, f32)> = src_s.iter().zip(&ora_s)
            .enumerate().map(|(i, (a, b))| (i, (a - b).abs())).collect();
        diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(idx, diff) in diffs.iter().take(3) {
            let a = idx / nh;
            let h = idx % nh;
            if h < pc.len() {
                let (c1, c2) = pc[h];
                println!("    [a={} {}{}]: src={:.4} ora={:.4} d={:.4}",
                    a, card_str(c1), card_str(c2), src_s[idx], ora_s[idx], diff);
            }
        }
    }
}

#[derive(Default)]
struct CompareStats {
    nodes: usize,
    elements: usize,
    max_diff: f32,
    sum_diff: f64,
    gt_1pct: usize,
    gt_5pct: usize,
    gt_10pct: usize,
}

fn walk_flop_tree(
    g1: &PostFlopGame, g2: &PostFlopGame,
    n1: &mut PostFlopNode, n2: &mut PostFlopNode,
    stats: &mut CompareStats,
) {
    if n1.is_terminal() { return; }
    if n1.is_chance() && n1.turn() == NOT_DEALT { return; }

    if !n1.is_chance() {
        let p = n1.player();
        let nh = if p == 0 { g1.num_private_hands(0) } else { g1.num_private_hands(1) };
        let na = n1.num_actions();
        let s1 = normalize(n1.strategy(), na, nh);
        let s2 = normalize(n2.strategy(), na, nh);
        stats.nodes += 1;
        stats.elements += s1.len();
        for (a, b) in s1.iter().zip(&s2) {
            let d = (a - b).abs();
            if d > stats.max_diff { stats.max_diff = d; }
            stats.sum_diff += d as f64;
            if d > 0.01 { stats.gt_1pct += 1; }
            if d > 0.05 { stats.gt_5pct += 1; }
            if d > 0.10 { stats.gt_10pct += 1; }
        }
    }

    for a in 0..n1.num_actions() {
        let mut c1 = n1.play(a);
        let mut c2 = n2.play(a);
        walk_flop_tree(g1, g2, &mut c1, &mut c2, stats);
    }
}

fn normalize(strategy: &[f32], na: usize, nh: usize) -> Vec<f32> {
    let mut out = strategy.to_vec();
    for h in 0..nh {
        let mut d = 0.0f32;
        for a in 0..na { d += out[a * nh + h].max(0.0); }
        if d > 0.0 {
            for a in 0..na { out[a * nh + h] = out[a * nh + h].max(0.0) / d; }
        } else {
            let u = 1.0 / na as f32;
            for a in 0..na { out[a * nh + h] = u; }
        }
    }
    out
}

fn card_str(card: u8) -> String {
    let r = match card >> 2 {
        0=>'2',1=>'3',2=>'4',3=>'5',4=>'6',5=>'7',6=>'8',7=>'9',8=>'T',9=>'J',10=>'Q',11=>'K',12=>'A',_=>'?'
    };
    let s = match card & 3 { 0=>'c',1=>'d',2=>'h',3=>'s',_=>'?' };
    format!("{}{}", r, s)
}
