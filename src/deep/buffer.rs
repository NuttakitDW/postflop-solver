//! Reservoir sampling buffer for Deep PDCFR+ training data.

use rand::Rng;

/// A sample containing information set features and target values.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Encoded information set features
    pub features: Vec<f32>,
    /// Target values (advantages or strategy)
    pub targets: Vec<f32>,
    /// Iteration when sample was collected
    pub iteration: usize,
    /// Player who made the decision (0=OOP, 1=IP)
    pub player: usize,
    /// Sampling weight (for importance sampling)
    pub weight: f32,
}

impl Sample {
    pub fn new(
        features: Vec<f32>,
        targets: Vec<f32>,
        iteration: usize,
        player: usize,
        weight: f32,
    ) -> Self {
        Self {
            features,
            targets,
            iteration,
            player,
            weight,
        }
    }
}

/// Reservoir sampling buffer with uniform distribution.
///
/// Maintains a fixed-size buffer of samples, where each sample has equal
/// probability of being retained regardless of when it was added.
#[derive(Clone, Debug)]
pub struct ReservoirBuffer {
    samples: Vec<Sample>,
    capacity: usize,
    total_seen: usize,
}

impl ReservoirBuffer {
    /// Create a new reservoir buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            total_seen: 0,
        }
    }

    /// Add a sample to the buffer using reservoir sampling.
    pub fn add(&mut self, sample: Sample) {
        self.total_seen += 1;

        if self.samples.len() < self.capacity {
            // Buffer not full, just add
            self.samples.push(sample);
        } else {
            // Reservoir sampling: replace with probability capacity/total_seen
            let mut rng = rand::thread_rng();
            let idx = rng.gen_range(0..self.total_seen);
            if idx < self.capacity {
                self.samples[idx] = sample;
            }
        }
    }

    /// Add multiple samples.
    pub fn add_batch(&mut self, samples: impl IntoIterator<Item = Sample>) {
        for sample in samples {
            self.add(sample);
        }
    }

    /// Get current number of samples in buffer.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get total number of samples seen (including replaced ones).
    pub fn total_seen(&self) -> usize {
        self.total_seen
    }

    /// Sample a random batch from the buffer.
    pub fn sample_batch(&self, batch_size: usize) -> Vec<&Sample> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let actual_batch_size = batch_size.min(self.samples.len());

        (0..actual_batch_size)
            .map(|_| {
                let idx = rng.gen_range(0..self.samples.len());
                &self.samples[idx]
            })
            .collect()
    }

    /// Get all samples as a slice.
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.total_seen = 0;
    }

    /// Get samples filtered by player.
    pub fn samples_for_player(&self, player: usize) -> Vec<&Sample> {
        self.samples.iter().filter(|s| s.player == player).collect()
    }

    /// Get samples from a specific iteration range (inclusive).
    pub fn samples_in_range(&self, from_iter: usize, to_iter: usize) -> Vec<&Sample> {
        self.samples
            .iter()
            .filter(|s| s.iteration >= from_iter && s.iteration <= to_iter)
            .collect()
    }
}

/// Strategy buffer for iteration-weighted average strategy.
///
/// Unlike the advantage buffer, this stores samples with iteration weights
/// for computing the average strategy.
#[derive(Clone, Debug)]
pub struct StrategyBuffer {
    inner: ReservoirBuffer,
}

impl StrategyBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: ReservoirBuffer::new(capacity),
        }
    }

    /// Add a strategy sample with iteration weighting.
    ///
    /// The weight is automatically set based on the iteration number
    /// following the linear weighting scheme: weight = t
    pub fn add(&mut self, features: Vec<f32>, strategy: Vec<f32>, iteration: usize, player: usize) {
        let sample = Sample::new(
            features,
            strategy,
            iteration,
            player,
            iteration as f32, // Linear weighting
        );
        self.inner.add(sample);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn sample_batch(&self, batch_size: usize) -> Vec<&Sample> {
        self.inner.sample_batch(batch_size)
    }

    /// Get all samples as a slice.
    pub fn samples(&self) -> &[Sample] {
        self.inner.samples()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Advantage buffer for cumulative/instantaneous advantages.
#[derive(Clone, Debug)]
pub struct AdvantageBuffer {
    inner: ReservoirBuffer,
}

impl AdvantageBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: ReservoirBuffer::new(capacity),
        }
    }

    /// Add an advantage sample.
    pub fn add(
        &mut self,
        features: Vec<f32>,
        advantages: Vec<f32>,
        iteration: usize,
        player: usize,
        weight: f32,
    ) {
        let sample = Sample::new(features, advantages, iteration, player, weight);
        self.inner.add(sample);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn sample_batch(&self, batch_size: usize) -> Vec<&Sample> {
        self.inner.sample_batch(batch_size)
    }

    pub fn samples_for_player(&self, player: usize) -> Vec<&Sample> {
        self.inner.samples_for_player(player)
    }

    /// Get all samples as a slice.
    pub fn samples(&self) -> &[Sample] {
        self.inner.samples()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_basic() {
        let mut buffer = ReservoirBuffer::new(10);

        // Add samples
        for i in 0..5 {
            buffer.add(Sample::new(vec![i as f32], vec![0.0], i, 0, 1.0));
        }

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.total_seen(), 5);
    }

    #[test]
    fn test_reservoir_overflow() {
        let mut buffer = ReservoirBuffer::new(10);

        // Add more samples than capacity
        for i in 0..100 {
            buffer.add(Sample::new(vec![i as f32], vec![0.0], i, 0, 1.0));
        }

        assert_eq!(buffer.len(), 10); // Capped at capacity
        assert_eq!(buffer.total_seen(), 100);
    }

    #[test]
    fn test_sample_batch() {
        let mut buffer = ReservoirBuffer::new(100);

        for i in 0..50 {
            buffer.add(Sample::new(vec![i as f32], vec![0.0], i, 0, 1.0));
        }

        let batch = buffer.sample_batch(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_player_filter() {
        let mut buffer = ReservoirBuffer::new(100);

        for i in 0..20 {
            let player = i % 2;
            buffer.add(Sample::new(vec![i as f32], vec![0.0], i, player, 1.0));
        }

        let player0_samples = buffer.samples_for_player(0);
        let player1_samples = buffer.samples_for_player(1);

        assert_eq!(player0_samples.len(), 10);
        assert_eq!(player1_samples.len(), 10);
    }
}
