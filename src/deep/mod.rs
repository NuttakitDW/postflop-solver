//! Deep PDCFR+ implementation for fast poker solving.
//!
//! This module implements VR-DeepPDCFR+ (Variance-Reduced Deep Predictive Discounted CFR+)
//! which uses neural networks to approximate strategies across different game states.
//!
//! # Architecture
//! - **Advantage Networks**: Learn cumulative and instantaneous advantages
//! - **Strategy Network**: Outputs action probabilities via regret matching
//! - **Value Network**: Provides variance reduction baseline
//!
//! # Usage
//! 1. Create a `DeepSolverConfig` with desired hyperparameters
//! 2. Initialize `DeepSolver` with game configuration
//! 3. Call `train()` to run VR-DeepPDCFR+ iterations
//! 4. Export to PostFlopGame format for UI compatibility

mod buffer;
mod config;
mod encoder;
mod export;
mod networks;
mod solver;
mod traversal;

pub use buffer::*;
pub use config::*;
pub use encoder::*;
pub use export::*;
pub use networks::*;
pub use solver::*;
pub use traversal::*;
