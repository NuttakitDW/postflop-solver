#!/bin/bash
# Build oracles for all boards listed in config/boards.txt
# Uses template.json as template, replacing only the board name.
#
# Usage: bash scripts/build_all_oracles.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BOARDS_FILE="$ROOT_DIR/config/boards.txt"
TEMPLATE="$ROOT_DIR/config/template.json"

if [ ! -f "$BOARDS_FILE" ]; then
    echo "Error: $BOARDS_FILE not found"
    exit 1
fi

if [ ! -f "$TEMPLATE" ]; then
    echo "Error: Template config $TEMPLATE not found"
    exit 1
fi

# Build release binary once upfront
echo "=== Building release binary ==="
cargo build --example build_pairs_v2 --release --features "bincode rayon"
echo ""

TOTAL=0
FAILED=0
SKIPPED=0

while IFS= read -r board || [ -n "$board" ]; do
    # Skip empty lines and comments
    [[ -z "$board" || "$board" == \#* ]] && continue

    TOTAL=$((TOTAL + 1))
    CONFIG="$ROOT_DIR/config/${board}.json"
    ORACLE="$ROOT_DIR/data/oracles/${board}.dpairs2"

    # Skip if oracle already exists
    if [ -f "$ORACLE" ]; then
        echo "[$TOTAL] SKIP $board (oracle exists: $ORACLE)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Generate config from template if it doesn't exist
    if [ ! -f "$CONFIG" ]; then
        sed -e "s/BOARD_NAME/$board/g" "$TEMPLATE" > "$CONFIG"
        echo "[$TOTAL] Generated config: $CONFIG"
    fi

    echo "[$TOTAL] Building oracle for $board ..."
    START_TIME=$SECONDS

    if cargo run --example build_pairs_v2 --release --features "bincode rayon" -- "$CONFIG"; then
        ELAPSED=$((SECONDS - START_TIME))
        echo "[$TOTAL] Done $board (${ELAPSED}s)"
    else
        ELAPSED=$((SECONDS - START_TIME))
        echo "[$TOTAL] FAILED $board (${ELAPSED}s)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done < "$BOARDS_FILE"

echo "=== All done ==="
echo "Total: $TOTAL, Skipped: $SKIPPED, Failed: $FAILED, Built: $((TOTAL - SKIPPED - FAILED))"
