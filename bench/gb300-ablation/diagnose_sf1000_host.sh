#!/usr/bin/env bash
# Untimed checks that the enabled scan and join paths were actually selected.
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 ABLATION_OUTPUT" >&2
  exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="$(cd "$1" && pwd)"
dataset="${DATASET_ROOT:-/mnt/datasets}/tpch_parquet_sf1000"
if [[ -n "${PIXI_BIN:-}" ]]; then
  pixi="$PIXI_BIN"
elif [[ -x "$repo/test_datasets/.generation-tools/pixi" ]]; then
  pixi="$repo/test_datasets/.generation-tools/pixi"
elif [[ -x "$HOME/sirius/test_datasets/.generation-tools/pixi" ]]; then
  pixi="$HOME/sirius/test_datasets/.generation-tools/pixi"
else
  pixi=pixi
fi
mkdir -p "$output/diagnostics"
export SIRIUS_PIN_COMPRESSION_TABLES=lineitem,orders
export SIRIUS_PRE_SQL='SET enable_duckdb_fallback = false;'
export SIRIUS_LOG_LEVEL=debug
cd "$repo"

for spec in fused_decode:3:6 build_probe:4:3 dynamic_filter:5:3; do
  IFS=: read -r label stage query <<< "$spec"
  case_dir="$output/diagnostics/$label"
  if [[ -e "$case_dir" ]]; then
    echo "Diagnostic output already exists: $case_dir" >&2
    exit 1
  fi
  mkdir -p "$case_dir"
  source_config="$output/stage${stage}-${label}/config.yaml"
  cp "$source_config" "$case_dir/config.yaml"
  if [[ "$label" == fused_decode ]]; then
    printf '    fused_scan_diagnostics: true\n' >> "$case_dir/config.yaml"
  fi
  echo "Checking $label with Q$query"
  set +e
  timeout --signal=TERM --kill-after=30s 180s "$pixi" run --locked python \
    test/tpch_performance/performance_test.py \
    --input "$dataset" --scale-factor 1000 --engine gpu --profile hot \
    --iterations 1 --queries "$query" --config "$case_dir/config.yaml" \
    --pin host --pin-compression --output "$case_dir" \
    --name "sf1000_host_ablation_diagnostic_$label" \
    > "$case_dir/progress.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$case_dir/exit-code"
  if (( status != 0 )); then
    echo "$label diagnostic failed ($status); inspect $case_dir/progress.log" >&2
    exit "$status"
  fi
done
