 # Model Spec
 Input [2015]:
  - [0-11] Board texture (12 features) — ranks, suits, paired, monotone, connectivity
  - [12-14] Stack geometry (3 features) — log SPR, pot fraction, stack fraction
  - [15-1014] OOP bucketed range (1000 buckets, sums to 1)
  - [1015-2014] IP bucketed range (1000 buckets, sums to 1)

  Output [2000]:
  - [0-999] OOP pot-normalized CFVs per bucket
  - [1000-1999] IP pot-normalized CFVs per bucket
 
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
    -- config/template.json --deepstack models/experiment/model_1000.onnx --device coreml

# Standard DCFR
make start CONFIG=config/template.json


# Generate training data

Step 1 — Generate raw solver data (expensive, ~4 min for 100 samples):
Randomly samples board/ranges/pot/stack, runs full DCFR solver on each turn-start game,
and saves raw 1326-combo reaches + CFVs as NPY files to data/solver_output/.

cargo run --release --example generate_raw_data --features "rayon" -- --output-dir ./data/solver_output_100k --num-samples 100000 --target-exploit 0.5 --seed 3321

Step 2 — Preprocess into training format (cheap, <1s):
Reads raw data from step 1, clusters 1326 combos into K=1000 buckets per board,
and projects reaches/CFVs into bucket space. Outputs model-ready inputs.npy [N,2015]
and targets.npy [N,2000].

cargo run --release --example preprocess_data --features "rayon" -- \
  --input-dir ./data/solver_output_100k \
  --output-dir ./data/training_data

Options:
  --input-dir <DIR>   Raw data directory (default: ./data/solver_output)
  --output-dir <DIR>  Output directory (default: ./data/training_data)
  --k <N>             Number of buckets (default: 1000)


# Oracle Lookup Table
cargo run --example build_lookup_table --release --features "bincode rayon" -- config/A-oracle-1.json
cargo run --example solve_with_oracle --release --features "bincode rayon" -- config/A-oracle.json