# Oracle Lookup Table
cargo run --example build_pairs_v2 --release --features "bincode rayon" -- config/KcQh7s.json
cargo run --example solve_with_pairs_v2 --release --features "bincode rayon" -- config/KcQh7s.json


ORT_DYLIB_PATH=/opt/homebrew/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.1.24.1.dylib cargo run --example solve_with_model_v1 --release --features "bincode rayon" -- config/KcQh7s.json models/bt1_KcQh7s