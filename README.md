 # CPU only
  cargo run --example backend_solver --release \
    --features "bincode rayon zstd jemalloc onnx" \
    -- config/template.json --deepstack model_2.onnx --device cpu

  # CUDA (NVIDIA GPU)
  cargo run --example backend_solver --release \
    --features "bincode rayon zstd jemalloc onnx-cuda" \
    -- config/template.json --deepstack model_2.onnx --device cuda

  # CoreML (macOS)
  cargo run --example backend_solver --release \
    --features "bincode rayon zstd jemalloc onnx-coreml" \
    -- config/template.json --deepstack model_2.onnx --device coreml