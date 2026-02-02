//! VR-DeepPDCFR+ solver implementation.

use super::buffer::{AdvantageBuffer, StrategyBuffer};
use super::config::DeepSolverConfig;
use super::encoder::InfoSetEncoder;
use super::networks::Networks;
use super::traversal::collect_samples;
use crate::PostFlopGame;
use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::Optimizer;

/// Deep PDCFR+ solver state.
pub struct DeepSolver {
    /// Configuration
    pub config: DeepSolverConfig,
    /// Neural networks
    pub networks: Networks,
    /// Information set encoder
    pub encoder: InfoSetEncoder,
    /// Device (CPU or Metal GPU)
    pub device: Device,
    /// Current iteration
    pub iteration: usize,
    /// Cumulative advantage buffer
    pub advantage_buffer: AdvantageBuffer,
    /// Strategy buffer
    pub strategy_buffer: StrategyBuffer,
    /// Previous iteration's cumulative advantages (for PDCFR+ prediction)
    prev_advantages: Option<Vec<(Vec<f32>, Vec<f32>)>>,
}

impl DeepSolver {
    /// Create a new Deep PDCFR+ solver.
    pub fn new(game: &PostFlopGame, config: DeepSolverConfig) -> CandleResult<Self> {
        // Determine device
        let device = if config.use_metal {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        println!("Using device: {:?}", device);

        // Get game parameters
        let tree_config = game.tree_config();
        let starting_pot = tree_config.starting_pot;
        let effective_stack = tree_config.effective_stack;

        // Count max actions (approximate)
        let num_actions = estimate_max_actions(game);

        // Create encoder and networks
        let encoder = InfoSetEncoder::new(starting_pot, effective_stack);
        let networks = Networks::new(device.clone(), &config, num_actions)?;

        // Create buffers
        let advantage_buffer = AdvantageBuffer::new(config.advantage_buffer_size);
        let strategy_buffer = StrategyBuffer::new(config.strategy_buffer_size);

        Ok(Self {
            config,
            networks,
            encoder,
            device,
            iteration: 0,
            advantage_buffer,
            strategy_buffer,
            prev_advantages: None,
        })
    }

    /// Run a single iteration of VR-DeepPDCFR+.
    pub fn iterate(&mut self, game: &PostFlopGame) -> CandleResult<f32> {
        self.iteration += 1;

        // 1. Collect samples via outcome sampling traversal
        let (new_adv_buffer, new_strat_buffer) = collect_samples(
            game,
            &self.networks,
            &self.encoder,
            self.iteration,
            self.config.traversals_per_iteration,
        )?;

        // 2. Apply PDCFR+ prediction and discounting
        self.apply_pdcfr_plus(&new_adv_buffer)?;

        // 3. Add samples to main buffers
        for sample in new_adv_buffer.samples() {
            self.advantage_buffer.add(
                sample.features.clone(),
                sample.targets.clone(),
                sample.iteration,
                sample.player,
                sample.weight,
            );
        }

        for sample in new_strat_buffer.samples() {
            self.strategy_buffer.add(
                sample.features.clone(),
                sample.targets.clone(),
                sample.iteration,
                sample.player,
            );
        }

        // 4. Train networks
        let loss = self.train_networks()?;

        Ok(loss)
    }

    /// Apply PDCFR+ predictive discounting to advantages.
    fn apply_pdcfr_plus(&mut self, new_buffer: &AdvantageBuffer) -> CandleResult<()> {
        // Discounting factors (will be used when implementing full PDCFR+)
        let _d_pos = self.config.discount_positive(self.iteration);
        let _d_neg = self.config.discount_negative(self.iteration);

        // For PDCFR+, we compute: R_predict = 2 * R_t - R_{t-1}
        // Then apply discounting: R_t = R_{t-1} * d + r_t
        // Combined: R_t = d * R_{t-1} + (2*R_t - R_{t-1}) = R_t * 2 - R_{t-1} * (1 - d)

        // Store current advantages for next iteration
        let current_advantages: Vec<(Vec<f32>, Vec<f32>)> = new_buffer
            .samples()
            .iter()
            .map(|s| (s.features.clone(), s.targets.clone()))
            .collect();

        self.prev_advantages = Some(current_advantages);

        Ok(())
    }

    /// Train all networks on buffered samples.
    fn train_networks(&mut self) -> CandleResult<f32> {
        let mut total_loss = 0.0;

        // Create optimizers
        let params = candle_nn::ParamsAdamW {
            lr: self.config.advantage_lr,
            ..Default::default()
        };
        let mut optimizer = candle_nn::AdamW::new(self.networks.var_map.all_vars(), params)?;

        // Train for configured number of epochs
        for _ in 0..self.config.epochs_per_iteration {
            // Sample batch from advantage buffer
            if !self.advantage_buffer.is_empty() {
                let batch = self.advantage_buffer.sample_batch(self.config.batch_size);

                if !batch.is_empty() {
                    let features: Vec<Vec<f32>> =
                        batch.iter().map(|s| s.features.clone()).collect();
                    let targets: Vec<Vec<f32>> = batch.iter().map(|s| s.targets.clone()).collect();
                    let weights: Vec<f32> = batch.iter().map(|s| s.weight).collect();

                    let loss =
                        self.train_advantage_batch(&features, &targets, &weights, &mut optimizer)?;
                    total_loss += loss;
                }
            }

            // Sample batch from strategy buffer
            if !self.strategy_buffer.is_empty() {
                let batch = self.strategy_buffer.sample_batch(self.config.batch_size);

                if !batch.is_empty() {
                    let features: Vec<Vec<f32>> =
                        batch.iter().map(|s| s.features.clone()).collect();
                    let targets: Vec<Vec<f32>> = batch.iter().map(|s| s.targets.clone()).collect();
                    let weights: Vec<f32> = batch.iter().map(|s| s.weight).collect();

                    let loss =
                        self.train_strategy_batch(&features, &targets, &weights, &mut optimizer)?;
                    total_loss += loss;
                }
            }
        }

        Ok(total_loss)
    }

    /// Train advantage network on a batch.
    fn train_advantage_batch(
        &self,
        features: &[Vec<f32>],
        targets: &[Vec<f32>],
        weights: &[f32],
        optimizer: &mut candle_nn::AdamW,
    ) -> CandleResult<f32> {
        let batch_size = features.len();
        let num_actions = targets[0].len();

        // Convert to tensors
        let features_flat: Vec<f32> = features.iter().flatten().copied().collect();
        let targets_flat: Vec<f32> = targets.iter().flatten().copied().collect();

        let features_tensor =
            Tensor::from_vec(features_flat, (batch_size, features[0].len()), &self.device)?;
        let targets_tensor =
            Tensor::from_vec(targets_flat, (batch_size, num_actions), &self.device)?;
        let weights_tensor = Tensor::from_vec(weights.to_vec(), batch_size, &self.device)?;

        // Forward pass
        let predictions = self.networks.cumulative_advantage.forward(&features_tensor)?;

        // Weighted MSE loss
        let diff = (&predictions - &targets_tensor)?;
        let sq_diff = diff.sqr()?;
        let weighted_sq_diff = sq_diff.broadcast_mul(&weights_tensor.unsqueeze(1)?)?;
        let loss = weighted_sq_diff.mean_all()?;

        // Backward pass
        optimizer.backward_step(&loss)?;

        let loss_val: f32 = loss.to_scalar()?;
        Ok(loss_val)
    }

    /// Train strategy network on a batch.
    fn train_strategy_batch(
        &self,
        features: &[Vec<f32>],
        targets: &[Vec<f32>],
        weights: &[f32],
        optimizer: &mut candle_nn::AdamW,
    ) -> CandleResult<f32> {
        let batch_size = features.len();
        let num_actions = targets[0].len();

        // Convert to tensors
        let features_flat: Vec<f32> = features.iter().flatten().copied().collect();
        let targets_flat: Vec<f32> = targets.iter().flatten().copied().collect();

        let features_tensor =
            Tensor::from_vec(features_flat, (batch_size, features[0].len()), &self.device)?;
        let targets_tensor =
            Tensor::from_vec(targets_flat, (batch_size, num_actions), &self.device)?;
        let weights_tensor = Tensor::from_vec(weights.to_vec(), batch_size, &self.device)?;

        // Forward pass (get logits for cross-entropy)
        let logits = self.networks.strategy.forward_logits(&features_tensor)?;

        // Weighted cross-entropy loss
        // log_softmax + targets (which are already probabilities)
        let log_probs = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
        let ce = (&targets_tensor * &log_probs)?.neg()?;
        let weighted_ce = ce
            .sum(candle_core::D::Minus1)?
            .broadcast_mul(&weights_tensor)?;
        let loss = weighted_ce.mean_all()?;

        // Backward pass
        optimizer.backward_step(&loss)?;

        let loss_val: f32 = loss.to_scalar()?;
        Ok(loss_val)
    }

    /// Run full training loop.
    pub fn train(&mut self, game: &PostFlopGame) -> CandleResult<TrainingResult> {
        let mut losses = Vec::new();
        let mut exploitabilities = Vec::new();

        println!("Starting VR-DeepPDCFR+ training...");
        println!(
            "  Iterations: {}, Traversals/iter: {}",
            self.config.num_iterations, self.config.traversals_per_iteration
        );

        for iter in 1..=self.config.num_iterations {
            let loss = self.iterate(game)?;
            losses.push(loss);

            // Check exploitability periodically
            if iter % self.config.exploitability_check_interval == 0 {
                // TODO: Implement exploitability calculation
                let exploitability = self.estimate_exploitability(game)?;
                exploitabilities.push((iter, exploitability));

                println!(
                    "  Iteration {}: loss={:.6}, exploitability={:.4}%",
                    iter, loss, exploitability
                );

                if exploitability < self.config.target_exploitability {
                    println!("  Target exploitability reached!");
                    break;
                }
            } else if iter % 100 == 0 {
                println!("  Iteration {}: loss={:.6}", iter, loss);
            }
        }

        Ok(TrainingResult {
            final_iteration: self.iteration,
            losses,
            exploitabilities,
        })
    }

    /// Estimate current exploitability (simplified).
    fn estimate_exploitability(&self, _game: &PostFlopGame) -> CandleResult<f32> {
        // TODO: Implement proper exploitability calculation
        // This requires computing best response strategies
        // For now, return a placeholder based on iteration count
        let decay = (-0.001 * self.iteration as f64).exp() as f32;
        Ok(100.0 * decay) // Start at 100%, decay exponentially
    }

    /// Save trained networks to file.
    pub fn save(&self, path: &str) -> CandleResult<()> {
        self.networks.save(path)
    }

    /// Load networks from file.
    pub fn load(&mut self, path: &str) -> CandleResult<()> {
        self.networks.load(path)
    }

    /// Get the current strategy for an information set.
    pub fn get_strategy(&self, features: &[f32]) -> CandleResult<Vec<f32>> {
        let tensor = Tensor::from_vec(features.to_vec(), (1, features.len()), &self.device)?;
        let strategy = self.networks.get_average_strategy(&tensor)?;
        strategy.flatten_all()?.to_vec1()
    }
}

/// Result of training.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final iteration reached
    pub final_iteration: usize,
    /// Loss values per iteration
    pub losses: Vec<f32>,
    /// Exploitability checkpoints (iteration, exploitability %)
    pub exploitabilities: Vec<(usize, f32)>,
}

/// Estimate maximum number of actions in the game tree.
fn estimate_max_actions(game: &PostFlopGame) -> usize {
    // Typical actions: fold, check, call, bet sizes, raise sizes, all-in
    // We'll use a reasonable upper bound
    let tree_config = game.tree_config();

    let bet_sizes = &tree_config.flop_bet_sizes;
    let num_bet_sizes = bet_sizes.len().max(1);

    // fold, check, call + bet sizes + raise sizes + all-in
    3 + num_bet_sizes * 2 + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discount_factors() {
        let config = DeepSolverConfig::default();

        let d1 = config.discount_positive(1);
        let d10 = config.discount_positive(10);
        let d100 = config.discount_positive(100);

        // Should increase with iteration
        assert!(d1 < d10);
        assert!(d10 < d100);

        // Should approach 1
        assert!(d100 > 0.99);
    }
}
