.PHONY: start start-debug build clean export-json export-json-game2 export-json-console

CONFIG ?= config/template.json
FLOP_FILE ?= game.flop
OUTPUT_JSON ?= game-export.json

start:
	cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc" -- $(CONFIG)

start-debug:
	RUST_LOG=debug cargo run --example backend_solver --release --features "bincode rayon zstd jemalloc logging" -- $(CONFIG)

build:
	cargo build --example backend_solver --release --features "bincode rayon zstd jemalloc"

clean:
	cargo clean

# Export complete JSON with Combos, Equity, EV, and EQR
# Usage: make export-json FLOP_FILE=game2.flop OUTPUT_JSON=output.json
export-json:
	@echo "=== Exporting $(FLOP_FILE) to $(OUTPUT_JSON) ==="
	@cargo run --release --features "bincode rayon zstd json-export" --example export_json -- $(FLOP_FILE) $(OUTPUT_JSON)
	@echo "=== Export completed ==="
	@ls -lh $(OUTPUT_JSON)

# Export game2.flop specifically
export-json-game2:
	@echo "=== Exporting game2.flop to game2-complete.json ==="
	@cargo run --release --features "bincode rayon zstd json-export" --example export_json -- game2.flop game2-complete.json
	@echo "=== Export completed ==="
	@ls -lh game2-complete.json
	@echo ""
	@echo "=== Sample Hand Data ==="
	@jq '.hand_data | {oop_hands: (.oop_private_cards | length), ip_hands: (.ip_private_cards | length), sample_oop: {combo: .oop_private_cards[0], equity: .oop_equity[0], ev: .oop_ev[0], eqr: .oop_eqr[0]}, sample_ip: {combo: .ip_private_cards[0], equity: .ip_equity[0], ev: .ip_ev[0], eqr: .ip_eqr[0]}}' game2-complete.json

# Export to console (first 100 lines of JSON)
export-json-console:
	@echo "=== Exporting $(FLOP_FILE) to console ==="
	@cargo run --release --features "bincode rayon zstd json-export" --example export_json -- $(FLOP_FILE) /tmp/temp-export.json
	@echo ""
	@echo "=== JSON Structure ==="
	@jq '{metadata: {version, is_solved, storage_mode}, hand_counts: {oop: (.hand_data.oop_private_cards | length), ip: (.hand_data.ip_private_cards | length)}, fields: (.hand_data | keys), sample_hands: {oop: [{combo: .hand_data.oop_private_cards[0], equity: .hand_data.oop_equity[0], ev: .hand_data.oop_ev[0], eqr: .hand_data.oop_eqr[0]}, {combo: .hand_data.oop_private_cards[1], equity: .hand_data.oop_equity[1], ev: .hand_data.oop_ev[1], eqr: .hand_data.oop_eqr[1]}], ip: [{combo: .hand_data.ip_private_cards[0], equity: .hand_data.ip_equity[0], ev: .hand_data.ip_ev[0], eqr: .hand_data.ip_eqr[0]}, {combo: .hand_data.ip_private_cards[1], equity: .hand_data.ip_equity[1], ev: .hand_data.ip_ev[1], eqr: .hand_data.ip_eqr[1]}]}}' /tmp/temp-export.json
	@rm /tmp/temp-export.json
