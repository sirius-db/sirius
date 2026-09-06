#!/usr/bin/env bash
# Generate the transferred notes' pinned, exact-decimal SF100 dataset on this host.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$PROJECT_DIR/test_datasets/tpchgen-rs"
OUTPUT="${1:-$PROJECT_DIR/test_datasets/tpch_parquet_sf100}"
JOBS="${2:-4}"
PIN=cdcf74def0072f94bf1886667e8d2ac51feb8721
cd "$PROJECT_DIR"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "Jobs must be positive" >&2; exit 1; }
[[ "$(git -C "$GENERATOR" rev-parse HEAD)" == "$PIN" ]] || {
    echo "Expected tpchgen-rs checkout at $PIN: $GENERATOR" >&2
    exit 1
}
[[ -x "$GENERATOR/target/release/tpchgen-cli" ]] || {
    echo "Build the pinned generator first with pixi run, RUSTFLAGS='-C target-cpu=native' and cargo build --release -p tpchgen-cli -j 4" >&2
    exit 1
}
if [[ -e "$OUTPUT" ]]; then
    [[ -f "$OUTPUT/generation-manifest.json" && ! -e "$OUTPUT/_GENERATION_INCOMPLETE" ]] || {
        echo "Refusing to skip or overwrite an incomplete/existing dataset: $OUTPUT" >&2
        exit 1
    }
    echo "Existing dataset: performing full verification again."
else
    mkdir -p "$OUTPUT"
    date -u +%FT%TZ > "$OUTPUT/_GENERATION_INCOMPLETE"
    pixi run --frozen python -u "$GENERATOR/scripts/generate_tpch.py" \
        -s 100 -f parquet -j "$JOBS" -o "$OUTPUT"
fi
pixi run --frozen python -u "$SCRIPT_DIR/verify-sf100.py" "$OUTPUT" \
    --generator "$GENERATOR" --jobs "$JOBS"
rm -f "$OUTPUT/_GENERATION_INCOMPLETE"
