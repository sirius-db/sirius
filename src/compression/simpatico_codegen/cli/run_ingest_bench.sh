#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Reproduce the GPU ingestion benchmark: Parquet (cuDF), Simpatico and Vortex (vortex-cuda) decode
# over one SF1000 partition of lineitem and orders, per column and full table. Writes the CSVs
# plot_ingest_bench.py reads. On shared machines, run it under a lock so nothing else uses the GPU.
#
# Usage: run_ingest_bench.sh OUT_DIR DATASET_DIR
#   BUILD_DIR    simpatico CMake build dir (default: <repo>/build/simpatico)
#   ENGINES      subset of "parquet simpatico vortex_gpu" (default: all)
#   VORTEX_GPU_CHUNK_ROWS  rows per block of the Vortex file (default: 16777216, the cuDF
#                NDS-H Vortex writer's)
#   WARMUP/ITERS (default: 3/10)
#   VORTEX_DIR   where the converted .vortex inputs are cached (default: ~/.cache/vortex_ingest_bench)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
out="${1:?usage: $0 OUT_DIR DATASET_DIR}"
data="${2:?usage: $0 OUT_DIR DATASET_DIR}"
build="${BUILD_DIR:-$repo/build/simpatico}"
engines="${ENGINES:-parquet simpatico vortex_gpu}"
gpu_chunk_rows="${VORTEX_GPU_CHUNK_ROWS:-16777216}"
vortex_gpu_bin="$here/vortex_gpu_ingest_bench/target/release/vortex_gpu_ingest_bench"
warmup="${WARMUP:-3}"
iters="${ITERS:-10}"
vortex_dir="${VORTEX_DIR:-$HOME/.cache/vortex_ingest_bench}"
plans="$here/../plans/tpch_sf1000"

mkdir -p "$out" "$vortex_dir"
for table in lineitem orders; do
  input="$data/$table/part.0.parquet"
  for mode in per-column full-table; do
    for engine in $engines; do
      csv="$out/${engine}_${table}_${mode}.csv"
      echo "== $engine $table $mode"
      case "$engine" in
        parquet)
          pixi run --manifest-path "$repo/pixi.toml" "$build/parquet_ingest_bench" --input "$input" \
            --mode "$mode" --warmup "$warmup" --iters "$iters" --csv-out "$csv" ;;
        simpatico)
          pixi run --manifest-path "$repo/pixi.toml" "$build/simpatico" benchmark --input "$input" \
            --plan "$plans/$table.txt" --mode "$mode" --warmup "$warmup" --iters "$iters" \
            --csv-out "$csv" ;;
        vortex_gpu)
          "$vortex_gpu_bin" --input "$input" --chunk-rows "$gpu_chunk_rows" \
            --vortex-file "$vortex_dir/${table}_part0_gpu$((gpu_chunk_rows >> 20))M.vortex" \
            --mode "$mode" --warmup "$warmup" --iters "$iters" --csv-out "$csv" ;;
      esac
    done
  done
done
