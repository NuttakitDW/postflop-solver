.PHONY: start start-debug build clean

CONFIG ?= config/template.json

start:
	cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc" -- $(CONFIG)

start-debug:
	RUST_LOG=debug cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc logging" -- $(CONFIG)

build:
	cargo build --example backend_solver --release --features "bincode rayon zstd jemalloc"

clean:
	cargo clean
