# bt1 + Model
# build training data
cargo run --example build_bt1 --release --features "bincode rayon" -- config/KcQh7s.json 

# training
python trainings/train_bt1.py

# solve with model
export ORT_DYLIB_PATH=/opt/homebrew/lib/python3.14/site-packages/onnxruntime/capi/libonnxruntime.1.24.1.dylib
cargo run --example solve_with_model_v1 --release --features "bincode rayon" -- config/KcQh7s.json models/bt1_KcQh7s

# output file path
bt1 = data/bt1
model = models/bt1_KcQh7s
flop = data/out