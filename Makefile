.PHONY: start build clean

CONFIG ?= config/50bb.json

start:
	cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc" -- $(CONFIG)

build:
	cargo build --example backend_solver --release --features "bincode rayon zstd jemalloc"

clean:
	cargo clean
