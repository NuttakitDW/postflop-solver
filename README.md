 # CPU only
  cargo run --example backend_solver --release \
    --features "bincode rayon zstd jemalloc onnx" \
    -- config/template.json --deepstack models/model_placeholder.onnx --device cpu

  # CUDA (NVIDIA GPU)
  cargo run --example backend_solver --release \
    --features "bincode rayon zstd jemalloc onnx-cuda" \
    -- config/template.json --deepstack models/model_placeholder.onnx --device cuda

  # CoreML (macOS)
  cargo run --example backend_solver --release \
    --features "bincode rayon zstd jemalloc onnx-coreml" \
    -- config/template.json --deepstack models/model_placeholder.onnx --device coreml

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

Step 1 — Generate raw solver data (expensive, ~4 min for 100 samples):
Randomly samples board/ranges/pot/stack, runs full DCFR solver on each turn-start game,
and saves raw 1326-combo reaches + CFVs as NPY files to data/solver_output/.

cargo run --release --example generate_raw_data --features "rayon" -- \
      --num-samples 100 --target-exploit 0.5 --seed 42

Step 2 — Project to training format (cheap, <1s):
Reads raw data from step 1, clusters 1326 combos into K=1000 buckets per board,
and projects reaches/CFVs into bucket space. Outputs model-ready inputs.npy [N,2015]
and targets.npy [N,2000] to data/training_data/.

cargo run --release --example project_data --features "rayon"