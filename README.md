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

# Standard DCFR
make start CONFIG=config/template.json


# Batch calling benchmark 
python scripts/test_model.py 
provider        batch=1   batch=49    49×loop   ratio b49/b1  ratio loop/b1
--------------------------------------------------------------------------------
CPU              10.1ms    441.1ms    261.4ms         43.6x         25.8x
CoreML-ML         2.3ms     70.3ms    121.2ms         31.0x         53.5x
CoreML-GPU        2.5ms     70.2ms    121.4ms         27.8x         48.1x


# Generate training data
1. Generate (expensive, do once):
cargo run --example generate_raw_data --release --features "rayon" -- \         
    --output-dir ./raw_training_data --num-samples 100000 --seed 42

2. Project (cheap, re-run anytime you change K, features, or weights)
cargo run --example project_data --release --features "rayon" -- \
    --input-dir ./raw_training_data --output-dir ./bucketed_training_data --k 1000