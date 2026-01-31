.PHONY: start build clean

CONFIG ?= config/template.json

start:
	cargo run --example backend_solver --release --features "subgame bincode rayon zstd jemalloc" -- $(CONFIG)

build:
	cargo build --example backend_solver --release --features "subgame bincode rayon zstd jemalloc"

clean:
	cargo clean
