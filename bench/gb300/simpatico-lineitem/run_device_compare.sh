#!/usr/bin/env bash
set -euo pipefail

start_part=${1:-0}
end_part=${2:-59}
dataset_root=${DATASET_ROOT:-/mnt/datasets/tpch_parquet_sf1000/lineitem}
output_dir=${OUTPUT_DIR:-test_datasets/simpatico-lineitem-device-compare}
plan=src/compression/simpatico_codegen/plans/tpch_sf1000/lineitem.txt
binary=build/release/simpatico_codegen/simpatico

mkdir -p "$output_dir"
for ((part=start_part; part<=end_part; part++)); do
  echo "[$(date -u +%FT%TZ)] Starting partition $part" | tee -a "$output_dir/progress.log"
  pixi run "$binary" device-compare \
    --input "$dataset_root/part.$part.parquet" --plan "$plan" \
    --warmup 3 --iters 10 --csv-out "$output_dir/part.$part.csv" \
    > "$output_dir/part.$part.log" 2>&1
  echo "[$(date -u +%FT%TZ)] Finished partition $part" | tee -a "$output_dir/progress.log"
done
