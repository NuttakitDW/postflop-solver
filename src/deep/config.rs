//! Configuration and hyperparameters for Deep PDCFR+.

/// Hyperparameters for the VR-DeepPDCFR+ algorithm.
#[derive(Clone, Debug)]
pub struct DeepSolverConfig {
    // Discounting parameters (from VR-DeepPDCFR+ paper)
    /// Alpha parameter for positive regret discounting (default: 2.3)
    pub alpha: f32,
    /// Gamma parameter for negative regret discounting (default: 2.0)
    pub gamma: f32,

    // Network architecture
    /// Hidden layer sizes for advantage networks
    pub advantage_hidden_sizes: Vec<usize>,
    /// Hidden layer sizes for strategy network
    pub strategy_hidden_sizes: Vec<usize>,
    /// Hidden layer sizes for value network (VR baseline)
    pub value_hidden_sizes: Vec<usize>,

    // Training parameters
    /// Learning rate for advantage networks (R and r)
    pub advantage_lr: f64,
    /// Learning rate for strategy network (Π)
    pub strategy_lr: f64,
    /// Learning rate for value network (Q)
    pub value_lr: f64,
    /// Batch size for network training
    pub batch_size: usize,
    /// Number of training epochs per iteration
    pub epochs_per_iteration: usize,

    // Reservoir buffer
    /// Maximum samples in advantage buffer
    pub advantage_buffer_size: usize,
    /// Maximum samples in strategy buffer
    pub strategy_buffer_size: usize,

    // Solver settings
    /// Number of CFR iterations
    pub num_iterations: usize,
    /// Number of traversals per iteration (for sampling variance reduction)
    pub traversals_per_iteration: usize,
    /// Target exploitability percentage to stop early
    pub target_exploitability: f32,
    /// How often to check exploitability (every N iterations)
    pub exploitability_check_interval: usize,

    // Hardware
    /// Use Metal GPU acceleration (macOS)
    pub use_metal: bool,
}

impl Default for DeepSolverConfig {
    fn default() -> Self {
        Self {
            // Discounting (VR-DeepPDCFR+ optimal values)
            alpha: 2.3,
            gamma: 2.0,

            // Network architecture
            advantage_hidden_sizes: vec![256, 128, 64],
            strategy_hidden_sizes: vec![256, 128, 64],
            value_hidden_sizes: vec![256, 128, 64],

            // Training parameters
            advantage_lr: 0.001,
            strategy_lr: 0.0001,
            value_lr: 0.001,
            batch_size: 256,
            epochs_per_iteration: 1,

            // Reservoir buffer sizes
            advantage_buffer_size: 1_000_000,
            strategy_buffer_size: 1_000_000,

            // Solver settings
            num_iterations: 10000,
            traversals_per_iteration: 100,
            target_exploitability: 0.5, // 0.5% of pot
            exploitability_check_interval: 100,

            // Hardware
            use_metal: true,
        }
    }
}

impl DeepSolverConfig {
    /// Create a fast config for quick testing/POC
    pub fn fast() -> Self {
        Self {
            num_iterations: 1000,
            traversals_per_iteration: 50,
            advantage_buffer_size: 100_000,
            strategy_buffer_size: 100_000,
            target_exploitability: 5.0, // 5% for faster convergence
            exploitability_check_interval: 50,
            ..Default::default()
        }
    }

    /// Create a high-quality config for production
    pub fn production() -> Self {
        Self {
            num_iterations: 50000,
            traversals_per_iteration: 200,
            advantage_buffer_size: 5_000_000,
            strategy_buffer_size: 5_000_000,
            target_exploitability: 0.1, // 0.1% of pot
            exploitability_check_interval: 500,
            ..Default::default()
        }
    }

    /// Compute discount factor for iteration t
    pub fn discount_positive(&self, t: usize) -> f32 {
        let t = t as f32;
        t.powf(self.alpha) / (t.powf(self.alpha) + 1.0)
    }

    /// Compute negative regret discount factor for iteration t
    pub fn discount_negative(&self, t: usize) -> f32 {
        let t = t as f32;
        t.powf(self.gamma) / (t.powf(self.gamma) + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discount_factors() {
        let config = DeepSolverConfig::default();

        // At t=1, discount should be ~0.5 (for alpha=2.3)
        let d1 = config.discount_positive(1);
        assert!(d1 > 0.4 && d1 < 0.6, "d1 = {}", d1);

        // Discount should increase with t
        let d10 = config.discount_positive(10);
        let d100 = config.discount_positive(100);
        assert!(d10 > d1);
        assert!(d100 > d10);
        assert!(d100 > 0.99); // Should approach 1
    }
}
