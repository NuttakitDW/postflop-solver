//! ONNX oracle for predicting counterfactual values at turn chance nodes.
//!
//! The oracle replaces turn+river subtree recursion with a neural network prediction,
//! enabling deepstack-style solving that only builds/solves the flop-level tree.

use crate::card::*;
use crate::game::*;
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

const NUM_COMBOS: usize = 1326;
const PER_COMBO_FEATURES: usize = 14;
const GLOBAL_FEATURES: usize = 20;
/// Fixed batch size for CoreML compatibility (max turn cards = 52 - 3 flop = 49).
const FIXED_BATCH: usize = 49;

/// Device selection for ONNX inference execution provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU-only execution (always available).
    Cpu,
    /// CoreML execution (macOS, requires `onnx-coreml` feature).
    CoreML,
    /// CUDA execution (NVIDIA GPU, requires `onnx-cuda` feature).
    Cuda,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::CoreML => write!(f, "coreml"),
            Device::Cuda => write!(f, "cuda"),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "coreml" | "mps" => Ok(Device::CoreML),
            "cuda" | "gpu" => Ok(Device::Cuda),
            _ => Err(format!("Unknown device '{}'. Use: cpu, cuda, coreml", s)),
        }
    }
}

/// ONNX-based oracle for predicting CFVs at turn boundary nodes.
///
/// Uses a pool of sessions for concurrent inference from multiple rayon threads.
pub struct OnnxOracle {
    sessions: Vec<Mutex<Session>>,
    next_idx: AtomicUsize,
    device: Device,
    /// Total oracle calls (atomic for thread-safe counting).
    call_count: AtomicU64,
    /// Total inference time in microseconds (atomic).
    inference_us: AtomicU64,
    /// Total feature extraction time in microseconds (atomic).
    feature_us: AtomicU64,
}

// OnnxOracle is Send + Sync: sessions are behind Mutex, AtomicUsize is inherently thread-safe
unsafe impl Send for OnnxOracle {}
unsafe impl Sync for OnnxOracle {}

impl OnnxOracle {
    /// Load an ONNX model with the specified device.
    pub fn new(model_path: &str, device: Device) -> Result<Self, String> {
        let pool_size = default_pool_size();
        Self::new_pool(model_path, pool_size, device)
    }

    fn new_pool(model_path: &str, pool_size: usize, device: Device) -> Result<Self, String> {
        let pool_size = pool_size.max(1);
        let mut sessions = Vec::with_capacity(pool_size);

        for _ in 0..pool_size {
            let builder = Session::builder()
                .map_err(|e| format!("Failed to create session builder: {}", e))?;

            let builder = match device {
                Device::Cpu => builder,
                Device::CoreML => {
                    #[cfg(feature = "onnx-coreml")]
                    {
                        builder
                            .with_execution_providers([
                                ort::ep::CoreML::default()
                                    .with_static_input_shapes(true)
                                    .build(),
                                ort::ep::CPU::default().build(),
                            ])
                            .map_err(|e| format!("Failed to set CoreML EP: {}", e))?
                    }
                    #[cfg(not(feature = "onnx-coreml"))]
                    {
                        return Err(
                            "CoreML not compiled. Rebuild with --features onnx-coreml".into(),
                        );
                    }
                }
                Device::Cuda => {
                    #[cfg(feature = "onnx-cuda")]
                    {
                        builder
                            .with_execution_providers([
                                ort::ep::CUDA::default().build(),
                                ort::ep::CPU::default().build(),
                            ])
                            .map_err(|e| format!("Failed to set CUDA EP: {}", e))?
                    }
                    #[cfg(not(feature = "onnx-cuda"))]
                    {
                        return Err(
                            "CUDA not compiled. Rebuild with --features onnx-cuda".into(),
                        );
                    }
                }
            };

            let session = builder
                .commit_from_file(model_path)
                .map_err(|e| format!("Failed to load model '{}': {}", model_path, e))?;

            sessions.push(Mutex::new(session));
        }

        Ok(Self {
            sessions,
            next_idx: AtomicUsize::new(0),
            device,
            call_count: AtomicU64::new(0),
            inference_us: AtomicU64::new(0),
            feature_us: AtomicU64::new(0),
        })
    }

    /// Number of sessions in the pool.
    pub fn pool_size(&self) -> usize {
        self.sessions.len()
    }

    /// The device this oracle was configured with.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Reset profiling counters and return (calls, inference_ms, feature_ms).
    pub fn reset_stats(&self) -> (u64, f64, f64) {
        let calls = self.call_count.swap(0, Ordering::Relaxed);
        let inf_us = self.inference_us.swap(0, Ordering::Relaxed);
        let feat_us = self.feature_us.swap(0, Ordering::Relaxed);
        (calls, inf_us as f64 / 1000.0, feat_us as f64 / 1000.0)
    }

    /// Record feature extraction time (called from solver code).
    pub fn record_feature_time(&self, us: u64) {
        self.feature_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Run batch inference with fixed batch padding for CoreML compatibility.
    ///
    /// Thread-safe: acquires one session from the pool for the duration of inference.
    /// Returns CFVs as a flat `Vec<f32>` of shape `[actual_batch, 1326, 2]` (row-major).
    pub fn predict(
        &self,
        combo_features: &Array3<f32>,
        global_features: &Array2<f32>,
    ) -> Result<Vec<f32>, String> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        let actual_batch = combo_features.shape()[0];

        // Pad to fixed batch size BEFORE acquiring session (minimize lock time)
        let (combo_padded, global_padded);
        let (combo_ref_arr, global_ref_arr) = if actual_batch == FIXED_BATCH {
            (combo_features, global_features)
        } else {
            combo_padded = pad_3d(combo_features, FIXED_BATCH);
            global_padded = pad_2d(global_features, FIXED_BATCH);
            (&combo_padded, &global_padded)
        };

        let combo_ref = TensorRef::from_array_view(combo_ref_arr.view())
            .map_err(|e| format!("Failed to create combo tensor: {}", e))?;
        let global_ref = TensorRef::from_array_view(global_ref_arr.view())
            .map_err(|e| format!("Failed to create global tensor: {}", e))?;

        // Acquire a session from the pool (round-robin with try_lock fallback)
        let inf_start = std::time::Instant::now();
        let mut session = self.acquire_session()?;

        let outputs = session
            .run(ort::inputs![combo_ref, global_ref])
            .map_err(|e| format!("Inference failed: {}", e))?;

        let cfv_view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract output: {}", e))?;

        let full_output = cfv_view.1;
        self.inference_us.fetch_add(inf_start.elapsed().as_micros() as u64, Ordering::Relaxed);

        // Trim to actual batch size
        if actual_batch == FIXED_BATCH {
            Ok(full_output.to_vec())
        } else {
            let trimmed_len = actual_batch * NUM_COMBOS * 2;
            Ok(full_output.iter().copied().take(trimmed_len).collect())
        }
    }

    /// Acquire a session from the pool using round-robin + try_lock.
    fn acquire_session(&self) -> Result<std::sync::MutexGuard<'_, Session>, String> {
        let n = self.sessions.len();
        let start = self.next_idx.fetch_add(1, Ordering::Relaxed) % n;

        // Fast path: try to find an unlocked session starting from our assigned slot
        for i in 0..n {
            let idx = (start + i) % n;
            if let Ok(guard) = self.sessions[idx].try_lock() {
                return Ok(guard);
            }
        }

        // All busy: wait on our assigned slot
        self.sessions[start]
            .lock()
            .map_err(|e| format!("Failed to lock session: {}", e))
    }
}

fn default_pool_size() -> usize {
    // Allow override via ORACLE_POOL_SIZE env var for tuning
    if let Ok(val) = std::env::var("ORACLE_POOL_SIZE") {
        if let Ok(n) = val.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Build the mapping from 1326-combo index to solver hand index for each player.
///
/// Returns `combo_to_hand[1326]` for a single player where `usize::MAX` means
/// the combo is not in that player's private cards.
pub fn build_combo_to_hand(game: &PostFlopGame, player: usize) -> Vec<usize> {
    let mut mapping = vec![usize::MAX; NUM_COMBOS];
    for (hand_idx, &(c1, c2)) in game.private_cards(player).iter().enumerate() {
        let combo_idx = card_pair_to_index(c1, c2);
        mapping[combo_idx] = hand_idx;
    }
    mapping
}

/// Extract combo_features and global_features for a batch of turn cards.
///
/// Returns `(combo_features[batch, 1326, 14], global_features[batch, 20])`.
///
/// `reach_oop_all` and `reach_ip_all` are 1326-element arrays indexed by combo index.
/// Combos not in a player's range should have reach = 0.0.
pub fn extract_features(
    game: &PostFlopGame,
    reach_oop_all: &[f32],
    reach_ip_all: &[f32],
    turn_cards: &[Card],
) -> (Array3<f32>, Array2<f32>) {
    let batch = turn_cards.len();
    let mut combo_feat = Array3::<f32>::zeros((batch, NUM_COMBOS, PER_COMBO_FEATURES));
    let mut global_feat = Array2::<f32>::zeros((batch, GLOBAL_FEATURES));

    let flop = game.card_config().flop;
    let tree_config = game.tree_config();
    let pot = tree_config.starting_pot as f32;
    let stack = tree_config.effective_stack as f32;
    let pot_norm = pot / 200.0;
    let stack_norm = stack / 200.0;
    let spr = if pot > 0.0 { stack / pot } else { 0.0 };

    for (b, &turn_card) in turn_cards.iter().enumerate() {
        let board = [flop[0], flop[1], flop[2], turn_card];

        fill_global_features(
            global_feat.row_mut(b).as_slice_mut().unwrap(),
            &board,
            pot_norm,
            stack_norm,
            spr,
        );

        for combo_idx in 0..NUM_COMBOS {
            let (c1, c2) = index_to_card_pair(combo_idx);
            let blocked = board.iter().any(|&bc| bc == c1 || bc == c2);

            let row = combo_feat
                .slice_mut(ndarray::s![b, combo_idx, ..])
                .into_slice()
                .unwrap();

            fill_combo_features(
                row,
                c1,
                c2,
                &board,
                reach_oop_all[combo_idx],
                reach_ip_all[combo_idx],
                blocked,
            );
        }
    }

    (combo_feat, global_feat)
}

/// Fill the 14 per-combo features for one combo.
fn fill_combo_features(
    out: &mut [f32],
    c1: Card,
    c2: Card,
    board: &[Card; 4],
    reach_oop: f32,
    reach_ip: f32,
    blocked: bool,
) {
    let r1 = (c1 >> 2) as f32;
    let r2 = (c2 >> 2) as f32;
    let s1 = c1 & 3;
    let s2 = c2 & 3;

    let is_pair = if r1 == r2 { 1.0 } else { 0.0 };
    let is_suited = if s1 == s2 { 1.0 } else { 0.0 };
    let rank_gap = (r1 - r2).abs() / 12.0;

    let mut suit_match_c1 = 0.0f32;
    let mut suit_match_c2 = 0.0f32;
    let mut pairs_with_board = 0.0f32;
    let board_max_rank = board.iter().map(|&c| (c >> 2) as f32).fold(0.0f32, f32::max);

    for &bc in board.iter() {
        let br = bc >> 2;
        let bs = bc & 3;
        if s1 == bs {
            suit_match_c1 += 0.25;
        }
        if s2 == bs {
            suit_match_c2 += 0.25;
        }
        if (c1 >> 2) == br || (c2 >> 2) == br {
            pairs_with_board += 1.0;
        }
    }

    let mut overcards = 0.0f32;
    if r1 > board_max_rank {
        overcards += 0.5;
    }
    if r2 > board_max_rank {
        overcards += 0.5;
    }

    let flush_potential = {
        let mut max_flush = 0u8;
        for suit in 0..4u8 {
            let mut count = 0u8;
            if s1 == suit {
                count += 1;
            }
            if s2 == suit {
                count += 1;
            }
            for &bc in board.iter() {
                if (bc & 3) == suit {
                    count += 1;
                }
            }
            max_flush = max_flush.max(count);
        }
        max_flush as f32 / 5.0
    };

    let hand_strength =
        (pairs_with_board * 0.1 + is_pair as f32 * 0.05 + overcards * 0.05).min(1.0);

    out[0] = reach_oop;
    out[1] = reach_ip;
    out[2] = r1 / 12.0;
    out[3] = r2 / 12.0;
    out[4] = is_pair;
    out[5] = is_suited;
    out[6] = rank_gap;
    out[7] = suit_match_c1;
    out[8] = suit_match_c2;
    out[9] = pairs_with_board;
    out[10] = overcards;
    out[11] = flush_potential;
    out[12] = hand_strength;
    out[13] = if blocked { 1.0 } else { 0.0 };
}

/// Fill the 20 global features for one board configuration.
fn fill_global_features(out: &mut [f32], board: &[Card; 4], pot: f32, stack: f32, spr: f32) {
    let mut ranks: Vec<f32> = board.iter().map(|&c| (c >> 2) as f32 / 12.0).collect();
    ranks.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let suits: Vec<f32> = board.iter().map(|&c| (c & 3) as f32 / 3.0).collect();

    let mut suit_counts = [0u8; 4];
    for &c in board.iter() {
        suit_counts[(c & 3) as usize] += 1;
    }

    let mut rank_counts = [0u8; 13];
    for &c in board.iter() {
        rank_counts[(c >> 2) as usize] += 1;
    }
    let board_paired = rank_counts.iter().any(|&count| count >= 2);
    let board_trips = rank_counts.iter().any(|&count| count >= 3);
    let monotone = suit_counts.iter().any(|&count| count >= 3);

    let high_card = ranks[0];
    let connectivity = {
        let mut sorted_ranks: Vec<u8> = board.iter().map(|&c| c >> 2).collect();
        sorted_ranks.sort();
        sorted_ranks.dedup();
        if sorted_ranks.len() < 2 {
            0.0
        } else {
            let span = (sorted_ranks.last().unwrap() - sorted_ranks.first().unwrap()) as f32;
            1.0 - (span / 12.0)
        }
    };

    out[0] = ranks[0];
    out[1] = ranks[1];
    out[2] = ranks[2];
    out[3] = ranks[3];
    out[4] = suits[0];
    out[5] = suits[1];
    out[6] = suits[2];
    out[7] = suits[3];
    out[8] = suit_counts[0] as f32 / 4.0;
    out[9] = suit_counts[1] as f32 / 4.0;
    out[10] = suit_counts[2] as f32 / 4.0;
    out[11] = suit_counts[3] as f32 / 4.0;
    out[12] = if board_paired { 1.0 } else { 0.0 };
    out[13] = if board_trips { 1.0 } else { 0.0 };
    out[14] = if monotone { 1.0 } else { 0.0 };
    out[15] = high_card;
    out[16] = connectivity;
    out[17] = pot;
    out[18] = stack;
    out[19] = spr;
}

/// Pad a 3D array along the first axis to `target` rows (zero-filled).
fn pad_3d(arr: &Array3<f32>, target: usize) -> Array3<f32> {
    let shape = arr.shape();
    let mut padded = Array3::<f32>::zeros((target, shape[1], shape[2]));
    padded.slice_mut(ndarray::s![..shape[0], .., ..]).assign(arr);
    padded
}

/// Pad a 2D array along the first axis to `target` rows (zero-filled).
fn pad_2d(arr: &Array2<f32>, target: usize) -> Array2<f32> {
    let shape = arr.shape();
    let mut padded = Array2::<f32>::zeros((target, shape[1]));
    padded.slice_mut(ndarray::s![..shape[0], ..]).assign(arr);
    padded
}
