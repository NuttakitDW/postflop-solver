//! Neural network architectures for Deep PDCFR+.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};

use super::config::DeepSolverConfig;
use super::encoder::ENCODING_DIM;

/// Multi-layer perceptron with ReLU activations.
#[derive(Debug)]
pub struct MLP {
    layers: Vec<Linear>,
    output_layer: Linear,
}

impl MLP {
    /// Create a new MLP.
    ///
    /// # Arguments
    /// * `vs` - Variable builder for weight initialization
    /// * `input_dim` - Input dimension
    /// * `hidden_dims` - Hidden layer dimensions
    /// * `output_dim` - Output dimension
    pub fn new(
        vs: VarBuilder,
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        for (i, &dim) in hidden_dims.iter().enumerate() {
            layers.push(linear(prev_dim, dim, vs.pp(format!("layer_{}", i)))?);
            prev_dim = dim;
        }

        let output_layer = linear(prev_dim, output_dim, vs.pp("output"))?;

        Ok(Self {
            layers,
            output_layer,
        })
    }

    /// Forward pass with ReLU activations.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?.relu()?;
        }
        self.output_layer.forward(&current)
    }
}

/// Advantage network R(I,a|θ) for cumulative advantages.
#[derive(Debug)]
pub struct AdvantageNetwork {
    mlp: MLP,
}

impl AdvantageNetwork {
    pub fn new(vs: VarBuilder, config: &DeepSolverConfig, num_actions: usize) -> Result<Self> {
        let mlp = MLP::new(
            vs,
            ENCODING_DIM,
            &config.advantage_hidden_sizes,
            num_actions,
        )?;
        Ok(Self { mlp })
    }

    /// Get advantages for all actions given an information set.
    pub fn forward(&self, info_set: &Tensor) -> Result<Tensor> {
        self.mlp.forward(info_set)
    }
}

/// Instantaneous advantage network r(I,a|φ) for PDCFR+ prediction.
#[derive(Debug)]
pub struct InstantAdvantageNetwork {
    mlp: MLP,
}

impl InstantAdvantageNetwork {
    pub fn new(vs: VarBuilder, config: &DeepSolverConfig, num_actions: usize) -> Result<Self> {
        let mlp = MLP::new(
            vs,
            ENCODING_DIM,
            &config.advantage_hidden_sizes,
            num_actions,
        )?;
        Ok(Self { mlp })
    }

    pub fn forward(&self, info_set: &Tensor) -> Result<Tensor> {
        self.mlp.forward(info_set)
    }
}

/// Strategy network Π(I,a|ψ) for average strategy.
#[derive(Debug)]
pub struct StrategyNetwork {
    mlp: MLP,
}

impl StrategyNetwork {
    pub fn new(vs: VarBuilder, config: &DeepSolverConfig, num_actions: usize) -> Result<Self> {
        let mlp = MLP::new(vs, ENCODING_DIM, &config.strategy_hidden_sizes, num_actions)?;
        Ok(Self { mlp })
    }

    /// Get strategy (probability distribution) via softmax.
    pub fn forward(&self, info_set: &Tensor) -> Result<Tensor> {
        let logits = self.mlp.forward(info_set)?;
        candle_nn::ops::softmax(&logits, candle_core::D::Minus1)
    }

    /// Get raw logits (for training with cross-entropy loss).
    pub fn forward_logits(&self, info_set: &Tensor) -> Result<Tensor> {
        self.mlp.forward(info_set)
    }
}

/// Value network Q(h,a|w) for variance reduction baseline.
#[derive(Debug)]
pub struct ValueNetwork {
    mlp: MLP,
}

impl ValueNetwork {
    pub fn new(vs: VarBuilder, config: &DeepSolverConfig) -> Result<Self> {
        let mlp = MLP::new(vs, ENCODING_DIM, &config.value_hidden_sizes, 1)?;
        Ok(Self { mlp })
    }

    /// Get value estimate for a history.
    pub fn forward(&self, info_set: &Tensor) -> Result<Tensor> {
        self.mlp.forward(info_set)
    }
}

/// Collection of all networks for VR-DeepPDCFR+.
pub struct Networks {
    pub var_map: VarMap,
    pub device: Device,
    pub cumulative_advantage: AdvantageNetwork,
    pub instant_advantage: InstantAdvantageNetwork,
    pub strategy: StrategyNetwork,
    pub value: ValueNetwork,
    num_actions: usize,
}

impl Networks {
    /// Create all networks with random initialization.
    pub fn new(device: Device, config: &DeepSolverConfig, num_actions: usize) -> Result<Self> {
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let cumulative_advantage =
            AdvantageNetwork::new(vs.pp("cumulative_advantage"), config, num_actions)?;
        let instant_advantage =
            InstantAdvantageNetwork::new(vs.pp("instant_advantage"), config, num_actions)?;
        let strategy = StrategyNetwork::new(vs.pp("strategy"), config, num_actions)?;
        let value = ValueNetwork::new(vs.pp("value"), config)?;

        Ok(Self {
            var_map,
            device,
            cumulative_advantage,
            instant_advantage,
            strategy,
            value,
            num_actions,
        })
    }

    /// Get current strategy from advantages using regret matching.
    pub fn get_strategy(&self, info_set: &Tensor) -> Result<Tensor> {
        let advantages = self.cumulative_advantage.forward(info_set)?;
        regret_matching(&advantages)
    }

    /// Get strategy from the strategy network directly.
    pub fn get_average_strategy(&self, info_set: &Tensor) -> Result<Tensor> {
        self.strategy.forward(info_set)
    }

    /// Convert info set features to tensor.
    pub fn encode_batch(&self, features: &[Vec<f32>]) -> Result<Tensor> {
        let batch_size = features.len();
        let flat: Vec<f32> = features.iter().flatten().copied().collect();
        Tensor::from_vec(flat, (batch_size, ENCODING_DIM), &self.device)
    }

    /// Get number of actions.
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Create Adam optimizers for training.
    pub fn create_optimizers(
        &self,
        config: &DeepSolverConfig,
    ) -> Result<(
        candle_nn::AdamW,
        candle_nn::AdamW,
        candle_nn::AdamW,
        candle_nn::AdamW,
    )> {
        let params = candle_nn::ParamsAdamW {
            lr: config.advantage_lr,
            ..Default::default()
        };

        // For simplicity, we'll train all networks with same var_map
        // In production, you might want separate var_maps
        let adv_opt = candle_nn::AdamW::new(self.var_map.all_vars(), params.clone())?;

        let instant_params = candle_nn::ParamsAdamW {
            lr: config.advantage_lr,
            ..Default::default()
        };
        let instant_opt = candle_nn::AdamW::new(self.var_map.all_vars(), instant_params)?;

        let strategy_params = candle_nn::ParamsAdamW {
            lr: config.strategy_lr,
            ..Default::default()
        };
        let strategy_opt = candle_nn::AdamW::new(self.var_map.all_vars(), strategy_params)?;

        let value_params = candle_nn::ParamsAdamW {
            lr: config.value_lr,
            ..Default::default()
        };
        let value_opt = candle_nn::AdamW::new(self.var_map.all_vars(), value_params)?;

        Ok((adv_opt, instant_opt, strategy_opt, value_opt))
    }

    /// Save network weights to file.
    pub fn save(&self, path: &str) -> Result<()> {
        self.var_map.save(path)
    }

    /// Load network weights from file.
    pub fn load(&mut self, path: &str) -> Result<()> {
        self.var_map.load(path)
    }
}

/// Regret matching: convert advantages to strategy probabilities.
pub fn regret_matching(advantages: &Tensor) -> Result<Tensor> {
    // Clamp negative advantages to 0
    let positive_advantages = advantages.maximum(&Tensor::zeros_like(advantages)?)?;

    // Sum of positive advantages
    let sum = positive_advantages
        .sum_keepdim(candle_core::D::Minus1)?
        .clamp(1e-8, f64::INFINITY)?;

    // Normalize to get probabilities
    let strategy = positive_advantages.broadcast_div(&sum)?;

    // Handle case where all advantages are non-positive (uniform strategy)
    let uniform = Tensor::ones_like(&strategy)?.broadcast_div(&Tensor::new(
        advantages.dim(candle_core::D::Minus1)? as f32,
        advantages.device(),
    )?)?;

    // Use uniform if sum is too small
    let mask = sum.lt(1e-6)?.to_dtype(DType::F32)?;
    let inv_mask = (1.0 - &mask)?;
    let result = (mask.broadcast_mul(&uniform)? + inv_mask.broadcast_mul(&strategy)?)?;

    Ok(result)
}

/// Compute MSE loss for regression.
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let diff = (predictions - targets)?;
    diff.sqr()?.mean_all()
}

/// Compute cross-entropy loss for strategy network.
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, candle_core::D::Minus1)?;
    let loss = (targets * &log_probs)?.neg()?.sum_all()?;
    loss / logits.dim(0)? as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_creation() -> Result<()> {
        let device = Device::Cpu;
        let var_map = VarMap::new();
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let mlp = MLP::new(vs, 100, &[64, 32], 10)?;

        // Test forward pass
        let input = Tensor::randn(0f32, 1.0, (1, 100), &device)?;
        let output = mlp.forward(&input)?;

        assert_eq!(output.dims(), &[1, 10]);
        Ok(())
    }

    #[test]
    fn test_regret_matching() -> Result<()> {
        let device = Device::Cpu;

        // Positive advantages
        let advantages = Tensor::new(&[[2.0f32, 1.0, 0.0]], &device)?;
        let strategy = regret_matching(&advantages)?;
        let strategy_vec: Vec<f32> = strategy.flatten_all()?.to_vec1()?;

        // Should be [2/3, 1/3, 0]
        assert!((strategy_vec[0] - 2.0 / 3.0).abs() < 1e-5);
        assert!((strategy_vec[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((strategy_vec[2] - 0.0).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_networks_creation() -> Result<()> {
        let device = Device::Cpu;
        let config = DeepSolverConfig::default();

        let networks = Networks::new(device, &config, 6)?;

        // Test forward pass
        let input = Tensor::randn(0f32, 1.0, (1, ENCODING_DIM), &networks.device)?;

        let strategy = networks.get_strategy(&input)?;
        assert_eq!(strategy.dims(), &[1, 6]);

        let avg_strategy = networks.get_average_strategy(&input)?;
        assert_eq!(avg_strategy.dims(), &[1, 6]);

        Ok(())
    }
}
