.PHONY: start build clean

CONFIG ?= config/50bb.json

start:
	cargo run --example backend_solver --release --features "bincode rayon zstd" -- $(CONFIG)

build:
	cargo build --example backend_solver --release --features "bincode rayon"

clean:
	cargo clean
