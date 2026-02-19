//! Bucketed value network for DeepStack-style turn CFV prediction.
//!
//! Wraps an ONNX model that maps a game-state vector (board features + bucketed ranges)
//! to bucketed counterfactual values.
//!
//! Input:  `[batch, 2015]` = board(15) + range_oop(1000) + range_ip(1000)
//! Output: `[batch, 2000]` = cfv_oop(1000) + cfv_ip(1000)

use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

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

const INPUT_DIM: usize = 2015;
const OUTPUT_DIM: usize = 2000;

/// ONNX-based bucketed value network for predicting turn CFVs.
///
/// Uses a pool of sessions for concurrent inference from multiple rayon threads.
/// Each call takes a `[batch, 2015]` input and returns `[batch, 2000]` output.
pub struct TurnValueNet {
    sessions: Vec<Mutex<Session>>,
    next_idx: AtomicUsize,
    device: Device,
    call_count: AtomicU64,
    inference_us: AtomicU64,
}

// TurnValueNet is Send + Sync: sessions are behind Mutex, atomics are inherently thread-safe
unsafe impl Send for TurnValueNet {}
unsafe impl Sync for TurnValueNet {}

impl TurnValueNet {
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
        })
    }

    /// Number of sessions in the pool.
    pub fn pool_size(&self) -> usize {
        self.sessions.len()
    }

    /// The device this net was configured with.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Reset profiling counters and return (calls, inference_ms).
    pub fn reset_stats(&self) -> (u64, f64) {
        let calls = self.call_count.swap(0, Ordering::Relaxed);
        let inf_us = self.inference_us.swap(0, Ordering::Relaxed);
        (calls, inf_us as f64 / 1000.0)
    }

    /// Run inference for a single game state.
    ///
    /// `input` must be exactly 2015 floats: board(15) + range_oop(1000) + range_ip(1000).
    /// Returns 2000 floats: cfv_oop(1000) + cfv_ip(1000).
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        assert_eq!(input.len(), INPUT_DIM, "Input must be {} floats", INPUT_DIM);
        self.predict_batch(input, 1)
    }

    /// Run batched inference.
    ///
    /// `inputs` must be `batch_size * 2015` floats in row-major order.
    /// Returns `batch_size * 2000` floats in row-major order.
    pub fn predict_batch(&self, inputs: &[f32], batch_size: usize) -> Result<Vec<f32>, String> {
        assert_eq!(
            inputs.len(),
            batch_size * INPUT_DIM,
            "Expected {} floats for batch_size={}, got {}",
            batch_size * INPUT_DIM,
            batch_size,
            inputs.len()
        );

        self.call_count.fetch_add(1, Ordering::Relaxed);

        let input_arr =
            Array2::from_shape_vec((batch_size, INPUT_DIM), inputs.to_vec())
                .map_err(|e| format!("Failed to create input array: {}", e))?;

        let input_ref = TensorRef::from_array_view(input_arr.view())
            .map_err(|e| format!("Failed to create input tensor: {}", e))?;

        let inf_start = std::time::Instant::now();
        let mut session = self.acquire_session()?;

        let outputs = session
            .run(ort::inputs![input_ref])
            .map_err(|e| format!("Inference failed: {}", e))?;

        let output_view = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract output: {}", e))?;

        let output_data = output_view.1;
        self.inference_us
            .fetch_add(inf_start.elapsed().as_micros() as u64, Ordering::Relaxed);

        let expected_len = batch_size * OUTPUT_DIM;
        Ok(output_data.iter().copied().take(expected_len).collect())
    }

    /// Acquire a session from the pool using round-robin + try_lock.
    fn acquire_session(&self) -> Result<std::sync::MutexGuard<'_, Session>, String> {
        let n = self.sessions.len();
        let start = self.next_idx.fetch_add(1, Ordering::Relaxed) % n;

        for i in 0..n {
            let idx = (start + i) % n;
            if let Ok(guard) = self.sessions[idx].try_lock() {
                return Ok(guard);
            }
        }

        self.sessions[start]
            .lock()
            .map_err(|e| format!("Failed to lock session: {}", e))
    }
}

fn default_pool_size() -> usize {
    if let Ok(val) = std::env::var("NET_POOL_SIZE") {
        if let Ok(n) = val.parse::<usize>() {
            return n.max(1);
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
