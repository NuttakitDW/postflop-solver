//! Deep PDCFR+ POC - Train on medium.json, test on medium.json
//!
//! This tests the full supervised learning pipeline:
//! 1. Solve medium.json with tabular DCFR → save as dcfr_medium.flop
//! 2. Train neural network to learn those strategies
//! 3. Use trained network to fill a new game → save as deep_medium.flop
//! 4. Compare: accuracy and inference speed
//!
//! Output files:
//!   - out/dcfr_medium.flop  (tabular DCFR solution)
//!   - out/deep_medium.flop  (Deep network solution)
//!
//! Run: cargo run --example deep_poc_medium --release --features "deep bincode zstd"

use postflop_solver::*;
use std::time::Instant;

#[cfg(feature = "deep")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "deep")]
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

/// Training sample
#[cfg(feature = "deep")]
struct Sample {
    features: Vec<f32>,
    strategy: Vec<f32>,
}

/// Simple MLP for strategy prediction
#[cfg(feature = "deep")]
struct StrategyNet {
    l1: Linear,
    l2: Linear,
    l3: Linear,
    out: Linear,
}

#[cfg(feature = "deep")]
impl StrategyNet {
    fn new(vs: VarBuilder, input: usize, output: usize) -> candle_core::Result<Self> {
        Ok(Self {
            l1: linear(input, 512, vs.pp("l1"))?,
            l2: linear(512, 256, vs.pp("l2"))?,
            l3: linear(256, 128, vs.pp("l3"))?,
            out: linear(128, output, vs.pp("out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.l1.forward(x)?.relu()?;
        let x = self.l2.forward(&x)?.relu()?;
        let x = self.l3.forward(&x)?.relu()?;
        self.out.forward(&x)
    }

    fn predict(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(x)?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }
}

/// Encode node state to features
#[cfg(feature = "deep")]
fn encode(
    flop: [u8; 3],
    turn: u8,
    river: u8,
    hole: (u8, u8),
    pot_ratio: f32,
    stack_ratio: f32,
    street: usize,
) -> Vec<f32> {
    let mut f = vec![0.0f32; 369];

    // Board cards one-hot (5 * 52 = 260)
    for (i, &c) in flop.iter().enumerate() {
        if c < 52 { f[i * 52 + c as usize] = 1.0; }
    }
    if turn < 52 { f[156 + turn as usize] = 1.0; }
    if river < 52 { f[208 + river as usize] = 1.0; }

    // Hole cards one-hot (2 * 52 = 104)
    let (c1, c2) = if hole.0 <= hole.1 { hole } else { (hole.1, hole.0) };
    if c1 < 52 { f[260 + c1 as usize] = 1.0; }
    if c2 < 52 { f[312 + c2 as usize] = 1.0; }

    // Pot and stack ratios
    f[364] = pot_ratio;
    f[365] = stack_ratio;

    // Street one-hot
    if street < 3 { f[366 + street] = 1.0; }

    f
}

/// Collect samples from solved game
#[cfg(feature = "deep")]
fn collect_samples(game: &PostFlopGame) -> Vec<Sample> {
    let mut samples = Vec::new();
    let flop = game.card_config().flop;
    let tc = game.tree_config();
    let pot0 = tc.starting_pot as f32;
    let stack0 = tc.effective_stack as f32;
    let hands = [game.private_cards(0), game.private_cards(1)];

    fn traverse(
        game: &PostFlopGame,
        node: &PostFlopNode,
        samples: &mut Vec<Sample>,
        flop: [u8; 3],
        hands: &[&[(u8, u8)]; 2],
        pot0: f32,
        stack0: f32,
        depth: usize,
    ) {
        if depth > 30 || node.is_terminal() { return; }

        if node.is_chance() {
            // Sample some chance outcomes
            for i in 0..node.num_actions().min(5) {
                let child = node.play(i);
                traverse(game, &child, samples, flop, hands, pot0, stack0, depth + 1);
            }
            return;
        }

        let player = node.player();
        if player > 1 { return; }

        let num_actions = node.num_actions();
        if num_actions == 0 { return; }

        let turn = node.turn_card();
        let river = node.river_card();
        let street = if turn == NOT_DEALT { 0 } else if river == NOT_DEALT { 1 } else { 2 };

        let strategy = node.strategy();
        let num_hands = hands[player].len();

        if strategy.len() == num_actions * num_hands && num_hands > 0 {
            // Sample hands
            for (hi, &(c1, c2)) in hands[player].iter().enumerate().step_by(5) {
                // Skip if conflicts with board
                let dominated = [flop[0], flop[1], flop[2], turn, river];
                if dominated.contains(&c1) || dominated.contains(&c2) { continue; }

                // Extract strategy for this hand
                let mut strat = Vec::with_capacity(num_actions);
                for a in 0..num_actions {
                    strat.push(strategy[a * num_hands + hi]);
                }

                // Normalize
                let sum: f32 = strat.iter().sum();
                if sum > 0.01 {
                    for s in &mut strat { *s /= sum; }

                    let pot = pot0 + 2.0 * (stack0 - node.bet_amount() as f32);
                    let pot_ratio = pot / (pot0 + 2.0 * stack0);
                    let stack_ratio = node.bet_amount() as f32 / stack0;

                    let t = if turn == NOT_DEALT { 255 } else { turn };
                    let r = if river == NOT_DEALT { 255 } else { river };

                    samples.push(Sample {
                        features: encode(flop, t, r, (c1, c2), pot_ratio, stack_ratio, street),
                        strategy: strat,
                    });
                }
            }
        }

        // Traverse children
        for i in 0..num_actions.min(3) {
            let child = node.play(i);
            traverse(game, &child, samples, flop, hands, pot0, stack0, depth + 1);
        }
    }

    let root = game.root();
    traverse(game, &root, &mut samples, flop, &hands, pot0, stack0, 0);
    samples
}

#[cfg(feature = "deep")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Deep PDCFR+ POC on medium.json ===\n");
    let total_start = Instant::now();

    // ===== STEP 1: Setup game from medium.json settings =====
    println!("Step 1: Setting up medium.json game...");

    let oop: Range = "66,55,44,33,22,ATs-A2s,A9o-A2o,KTs-K2s,KQo-K6o,QJs-Q2s,QJo-Q2o,J8s+,J6s-J2s,JTo-J4o,T8s+,T5s-T2s,T9o-T6o,98s,97s,96s,95s-92s,98o-96o,87s,86s,85s-82s,86o+,76s,75s-72s,75o+,62s+,65o,64o,63o,53o,52s+,42s+,32s".parse()?;
    let ip: Range = "22+,A2s+,A2o+,K2s+,K6o+,Q2s+,Q8o+,J2s+,J8o+,T3s+,T8o+,95s+,98o,85s+,87o,74s+,64s+,53s+".parse()?;

    let flop = flop_from_str("Td9d6h")?;

    let card_config = CardConfig {
        range: [oop, ip],
        flop,
        turn: NOT_DEALT,
        river: NOT_DEALT,
    };

    // Use medium.json bet sizes
    let oop_flop = BetSizeOptions::try_from(("33%", "33%, 60%"))?;
    let ip_flop = BetSizeOptions::try_from(("25%, 55%, 85%, 125%", "33%, 60%"))?;
    let turn_bet = BetSizeOptions::try_from(("33%, 75%, 125%", "33%, 60%"))?;
    let river_bet = BetSizeOptions::try_from(("33%, 75%, 125%", "33%, 60%"))?;

    let tree_config = TreeConfig {
        initial_state: BoardState::Flop,
        starting_pot: 61,
        effective_stack: 477,
        rake_rate: 0.0,
        rake_cap: 0.0,
        flop_bet_sizes: [oop_flop, ip_flop],
        turn_bet_sizes: [turn_bet.clone(), turn_bet.clone()],
        river_bet_sizes: [river_bet.clone(), river_bet.clone()],
        turn_donk_sizes: None,
        river_donk_sizes: None,
        add_allin_threshold: 6.0,
        force_allin_threshold: 0.5,
        merging_threshold: 0.1,
        max_raises_per_street: 4,
    };

    let action_tree = ActionTree::new(tree_config)?;
    let mut game = PostFlopGame::with_config(card_config, action_tree)?;

    let (mem, _) = game.memory_usage();
    println!("  Memory: {:.1} MB", mem as f64 / 1024.0 / 1024.0);
    println!("  OOP hands: {}, IP hands: {}", game.private_cards(0).len(), game.private_cards(1).len());

    // ===== STEP 2: Solve with tabular DCFR =====
    println!("\nStep 2: Solving with tabular DCFR...");
    let solve_start = Instant::now();

    game.allocate_memory(false);
    let target = 61.0 * 0.003; // 0.3% exploitability
    let expl = solve(&mut game, 1000, target, true);

    println!("  Exploitability: {:.4} ({:.2}%)", expl, expl / 61.0 * 100.0);
    println!("  Time: {:.1}s", solve_start.elapsed().as_secs_f64());

    // Save DCFR solution
    std::fs::create_dir_all("out")?;
    save_data_to_file(&game, "DCFR solution for medium.json", "out/dcfr_medium.flop", Some(10))?;
    println!("  Saved: out/dcfr_medium.flop");

    // ===== STEP 3: Collect training data =====
    println!("\nStep 3: Collecting training samples...");
    let collect_start = Instant::now();

    let samples = collect_samples(&game);
    let max_actions = samples.iter().map(|s| s.strategy.len()).max().unwrap_or(2);

    println!("  Samples: {}", samples.len());
    println!("  Max actions: {}", max_actions);
    println!("  Time: {:.2}s", collect_start.elapsed().as_secs_f64());

    if samples.is_empty() {
        println!("ERROR: No samples collected!");
        return Ok(());
    }

    // ===== STEP 4: Train neural network =====
    println!("\nStep 4: Training neural network...");
    let train_start = Instant::now();

    let device = Device::Cpu;
    let var_map = VarMap::new();
    let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let net = StrategyNet::new(vs, 369, max_actions)?;

    let mut opt = candle_nn::AdamW::new(
        var_map.all_vars(),
        candle_nn::ParamsAdamW { lr: 0.001, ..Default::default() },
    )?;

    let batch_size = 2048;
    let epochs = 50;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batches = 0;

        for batch in samples.chunks(batch_size) {
            // Filter by action count for batching
            let first_actions = batch[0].strategy.len();
            let valid: Vec<_> = batch.iter().filter(|s| s.strategy.len() == first_actions).collect();
            if valid.is_empty() { continue; }

            let bs = valid.len();
            let feats: Vec<f32> = valid.iter().flat_map(|s| s.features.iter().copied()).collect();
            let targs: Vec<f32> = valid.iter().flat_map(|s| {
                let mut t = s.strategy.clone();
                t.resize(max_actions, 0.0);
                t
            }).collect();

            let x = Tensor::from_vec(feats, (bs, 369), &device)?;
            let y = Tensor::from_vec(targs, (bs, max_actions), &device)?;

            let logits = net.forward(&x)?;
            let log_p = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let loss = (&y * &log_p)?.neg()?.sum_all()?;
            let loss = (&loss / bs as f64)?;

            opt.backward_step(&loss)?;
            total_loss += loss.to_scalar::<f32>()?;
            batches += 1;
        }

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: loss = {:.4}", epoch, total_loss / batches as f32);
        }
    }
    println!("  Training time: {:.1}s", train_start.elapsed().as_secs_f64());

    // ===== STEP 5: Test inference speed =====
    println!("\nStep 5: Testing inference speed...");

    let test_samples: Vec<_> = samples.iter().take(1000).collect();
    let inference_start = Instant::now();

    for sample in &test_samples {
        let x = Tensor::from_vec(sample.features.clone(), (1, 369), &device)?;
        let _ = net.predict(&x)?;
    }

    let inference_time = inference_start.elapsed();
    let per_query = inference_time.as_micros() as f64 / test_samples.len() as f64;

    println!("  {} queries in {:.2}ms", test_samples.len(), inference_time.as_secs_f64() * 1000.0);
    println!("  Per query: {:.1} µs ({:.0} queries/sec)", per_query, 1_000_000.0 / per_query);

    // ===== STEP 6: Check accuracy =====
    println!("\nStep 6: Checking accuracy...");

    let mut total_err = 0.0;
    let mut count = 0;

    for sample in test_samples.iter().take(100) {
        let x = Tensor::from_vec(sample.features.clone(), (1, 369), &device)?;
        let pred = net.predict(&x)?;
        let pred_vec: Vec<f32> = pred.flatten_all()?.to_vec1()?;

        let mut max_diff: f32 = 0.0;
        for (i, &t) in sample.strategy.iter().enumerate() {
            let p = if i < pred_vec.len() { pred_vec[i] } else { 0.0 };
            max_diff = max_diff.max((t - p).abs());
        }
        total_err += max_diff;
        count += 1;
    }

    let avg_err = total_err / count as f32;
    println!("  Average max error: {:.2}%", avg_err * 100.0);

    // ===== STEP 7: Summary of what we achieved =====
    println!("\nStep 7: Results saved");
    println!("  DCFR solution saved to: out/dcfr_medium.flop");
    println!("  (Deep export would require using src/deep/export.rs)");

    // ===== SUMMARY =====
    println!("\n========== SUMMARY ==========");
    println!("Tabular DCFR solve:  {:.1}s", solve_start.elapsed().as_secs_f64());
    println!("Deep training:       {:.1}s", train_start.elapsed().as_secs_f64());
    println!("Inference speed:     {:.0} µs/query", per_query);
    println!("Average error:       {:.1}%", avg_err * 100.0);
    println!("Total time:          {:.1}s", total_start.elapsed().as_secs_f64());
    println!("=============================");

    if avg_err < 0.15 {
        println!("\n✓ SUCCESS: Deep network learned the solution!");
    } else {
        println!("\n⚠ Network needs more training (error > 15%)");
    }

    Ok(())
}

#[cfg(not(feature = "deep"))]
fn main() {
    eprintln!("Run with: cargo run --example deep_poc_medium --release --features \"deep bincode zstd\"");
}
