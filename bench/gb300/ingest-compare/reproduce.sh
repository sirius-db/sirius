#!/usr/bin/env bash
# Re-run the SF1000 Parquet/Simpatico/Vortex ingestion microbenchmark from PR #2717.
set -euo pipefail

source_tree=${1:?usage: reproduce.sh PR2717_CHECKOUT SF1000_DATASET_DIR OUTPUT_DIR}
dataset_dir=${2:?usage: reproduce.sh PR2717_CHECKOUT SF1000_DATASET_DIR OUTPUT_DIR}
output_dir=${3:?usage: reproduce.sh PR2717_CHECKOUT SF1000_DATASET_DIR OUTPUT_DIR}

source_tree=$(cd "$source_tree" && pwd)
dataset_dir=$(cd "$dataset_dir" && pwd)
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

expected_commit=9fbe978487c8b6f2ef33b5328d9c740030950dd5
actual_commit=$(git -C "$source_tree" rev-parse HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "Expected PR #2717 source commit $expected_commit; found $actual_commit" >&2
  exit 1
fi
for table in lineitem orders; do
  test -f "$dataset_dir/$table/part.0.parquet" || {
    echo "Missing $dataset_dir/$table/part.0.parquet" >&2
    exit 1
  }
done

build_dir="$source_tree/build/simpatico"
test -f "$build_dir/CMakeCache.txt" || {
  echo "Configure the PR #2717 Simpatico CMake build at $build_dir first" >&2
  exit 1
}

pixi run --manifest-path "$source_tree/pixi.toml" cmake --build "$build_dir" \
  --parallel 12 --target simpatico_cli parquet_ingest_bench_cli
pixi run --manifest-path "$source_tree/pixi.toml" cargo build --release \
  --manifest-path "$source_tree/src/compression/simpatico_codegen/cli/vortex_gpu_ingest_bench/Cargo.toml"

BUILD_DIR="$build_dir" VORTEX_DIR="$output_dir/vortex-cache" WARMUP=3 ITERS=10 \
  bash "$source_tree/src/compression/simpatico_codegen/cli/run_ingest_bench.sh" \
  "$output_dir/results" "$dataset_dir" 2>&1 | tee "$output_dir/run.log"

python3 "$(dirname "$0")/plot_ratio_vs_decode.py" \
  --results-dir "$output_dir/results" --output "$output_dir/ratio-vs-decode.svg"
