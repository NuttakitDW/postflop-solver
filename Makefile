.PHONY: start start-debug start-deepstack build clean

CONFIG ?= config/template.json
MODEL ?= models/model_placeholder.onnx

start:
	cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc" -- $(CONFIG)

start-debug:
	RUST_LOG=debug cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc logging" -- $(CONFIG)

start-deepstack:
	cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc onnx" -- $(CONFIG) --deepstack $(MODEL)

build:
	cargo build --example backend_solver --release --features "bincode rayon zstd jemalloc"

clean:
	cargo clean
