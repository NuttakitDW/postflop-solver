//! Compare OOP flop strategies between two .flop files.
//!
//! Usage:
//!   cargo run --example compare_flop_files --release --features bincode -- file1.flop file2.flop

use postflop_solver::*;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <file1.flop> <file2.flop>", args[0]);
        std::process::exit(1);
    }

    println!("Loading files...");
    let (g1, m1): (PostFlopGame, String) =
        load_data_from_file(&args[1], None).expect("Failed to load file1");
    let (g2, m2): (PostFlopGame, String) =
        load_data_from_file(&args[2], None).expect("Failed to load file2");

    let nh = [g1.num_private_hands(0), g1.num_private_hands(1)];
    let pc = [g1.private_cards(0).to_vec(), g1.private_cards(1).to_vec()];

    // Full flop tree walk
    let mut stats = Stats::default();
    let mut r1 = g1.root();
    let mut r2 = g2.root();
    walk(&g1, &g2, &mut r1, &mut r2, &mut stats);

    let avg_diff_pct = stats.sum_diff / stats.elements as f64 * 100.0;

    println!();
    println!("=== Comparison: {} vs {} ===", m1, m2);
    println!();
    println!("  Full tree avg diff:  {:.2}%", avg_diff_pct);
    println!("  Nodes: {},  Elements: {}", stats.nodes, stats.elements);
    println!("  >10% diff: {:.1}%  |  >5%: {:.1}%  |  >1%: {:.1}%",
        stats.gt_10 as f64 / stats.elements as f64 * 100.0,
        stats.gt_5 as f64 / stats.elements as f64 * 100.0,
        stats.gt_1 as f64 / stats.elements as f64 * 100.0);

    // Root node detail
    println!();
    println!("=== Root Strategy (OOP) ===");
    let r1 = g1.root();
    let r2 = g2.root();
    let na = r1.num_actions();
    let s1 = normalize(r1.strategy(), na, nh[0]);
    let s2 = normalize(r2.strategy(), na, nh[0]);

    let actions: Vec<String> = g1.available_actions().iter().map(|a| match a {
        Action::Check => "Check".into(),
        Action::Fold => "Fold".into(),
        Action::Call => "Call".into(),
        Action::Bet(x) => format!("Bet{}", x),
        Action::Raise(x) => format!("Raise{}", x),
        Action::AllIn(x) => format!("AllIn{}", x),
        _ => format!("{:?}", a),
    }).collect();

    // Root avg diff
    let mut root_sum = 0.0f64;
    let root_elements = na * nh[0];
    for a in 0..na {
        for h in 0..nh[0] {
            root_sum += (s1[a * nh[0] + h] - s2[a * nh[0] + h]).abs() as f64;
        }
    }
    println!("  Root avg diff: {:.2}%", root_sum / root_elements as f64 * 100.0);

    // Per-action avg diff
    print!("  Per action:  ");
    for a in 0..na {
        let mut sum = 0.0f64;
        for h in 0..nh[0] {
            sum += (s1[a * nh[0] + h] - s2[a * nh[0] + h]).abs() as f64;
        }
        let sep = if a + 1 < na { "  |  " } else { "" };
        print!("{}: {:.2}%{}", actions[a], sum / nh[0] as f64 * 100.0, sep);
    }
    println!();

    // Top 10 worst hands
    let mut diffs: Vec<(usize, f32)> = Vec::new();
    for h in 0..nh[0] {
        let mut max_d = 0.0f32;
        for a in 0..na {
            let d = (s1[a * nh[0] + h] - s2[a * nh[0] + h]).abs();
            if d > max_d { max_d = d; }
        }
        diffs.push((h, max_d));
    }
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!("  Top 10 worst hands:");
    for &(h, max_d) in diffs.iter().take(10) {
        let (c1, c2) = pc[0][h];
        let hand = format!("{}{}", card_str(c1), card_str(c2));
        let mut parts = Vec::new();
        for a in 0..na {
            let v1 = s1[a * nh[0] + h] * 100.0;
            let v2 = s2[a * nh[0] + h] * 100.0;
            parts.push(format!("{}: {:.0}%→{:.0}%", actions[a], v1, v2));
        }
        println!("    {:<6} {:>5.1}% diff  ({})", hand, max_d * 100.0, parts.join(", "));
    }
    println!();
}

#[derive(Default)]
struct Stats {
    nodes: usize,
    elements: usize,
    max_diff: f32,
    sum_diff: f64,
    gt_1: usize,
    gt_5: usize,
    gt_10: usize,
}

fn walk(
    g1: &PostFlopGame, g2: &PostFlopGame,
    n1: &mut PostFlopNode, n2: &mut PostFlopNode,
    stats: &mut Stats,
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
            if d > 0.01 { stats.gt_1 += 1; }
            if d > 0.05 { stats.gt_5 += 1; }
            if d > 0.10 { stats.gt_10 += 1; }
        }
    }

    for a in 0..n1.num_actions() {
        let mut c1 = n1.play(a);
        let mut c2 = n2.play(a);
        walk(g1, g2, &mut c1, &mut c2, stats);
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
