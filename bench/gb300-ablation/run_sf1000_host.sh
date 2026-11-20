#!/usr/bin/env bash
# Cumulative SF1000 host-pin ablation on the GB300: baseline, then five switches.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dry_run=false
comparison=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    --baseline-vs-simpatico) comparison=true ;;
    *) echo "usage: $0 [--dry-run] [--baseline-vs-simpatico] (set ABLATION_OUTPUT to choose the output directory)" >&2; exit 2 ;;
  esac
done
dataset="${DATASET_ROOT:-/mnt/datasets}/tpch_parquet_sf1000"
config="$repo/bench/gb300-pin/config.yaml"
if [[ ! -d "$dataset" || ! -f "$config" ]]; then
  echo "SF1000 dataset or GB300 pin config missing" >&2
  exit 1
fi
mode="$(awk -F '"' '/^CoherentGPUMemoryMode:/ {print $2}' /proc/driver/nvidia/params 2>/dev/null || true)"
if [[ "$mode" != driver ]]; then
  echo "GB300 CDMM driver mode is required; found ${mode:-unknown}" >&2
  exit 1
fi
if [[ -n "${PIXI_BIN:-}" ]]; then
  pixi="$PIXI_BIN"
elif [[ -x "$repo/test_datasets/.generation-tools/pixi" ]]; then
  pixi="$repo/test_datasets/.generation-tools/pixi"
elif [[ -x "$HOME/sirius/test_datasets/.generation-tools/pixi" ]]; then
  pixi="$HOME/sirius/test_datasets/.generation-tools/pixi"
else
  pixi="pixi"
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output="${ABLATION_OUTPUT:-$repo/test_datasets/gb300-ablation/sf1000-host-$run_id}"
if [[ "$dry_run" == false ]]; then
  if [[ -e "$output" ]]; then
    echo "Output already exists: $output" >&2
    exit 1
  fi
  mkdir -p "$output"
  output="$(cd "$output" && pwd)"
  exec > >(tee "$output/run.log") 2>&1
fi

unset SIRIUS_CONFIG_FILE SIRIUS_PRE_SQL SIRIUS_DISABLE
export SIRIUS_PIN_COMPRESSION_TABLES=lineitem,orders
cd "$repo"
echo "Source: $(git rev-parse HEAD)"
echo "Dataset: $dataset"
echo "Config: $config"
echo "CDMM: $mode"
echo "Extension SHA-256: $(sha256sum "$repo/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension" | cut -d ' ' -f 1)"
echo "Output: $output"

failed=0
stages=(0 1 2 3 4 5)
if [[ "$comparison" == true ]]; then stages=(0 6); fi
for stage in "${stages[@]}"; do
  case "$stage" in
    0) label=all_off;       narrow=false; compress=false; fused=false; build_probe=false; dynamic=false ;;
    1) label=type_narrow;   narrow=true;  compress=false; fused=false; build_probe=false; dynamic=false ;;
    2) label=compressed_pin;narrow=true;  compress=true;  fused=false; build_probe=false; dynamic=false ;;
    3) label=fused_decode;  narrow=true;  compress=true;  fused=true;  build_probe=false; dynamic=false ;;
    4) label=build_probe;   narrow=true;  compress=true;  fused=true;  build_probe=true;  dynamic=false ;;
    5) label=dynamic_filter;narrow=true;  compress=true;  fused=true;  build_probe=true;  dynamic=true ;;
    6) label=simpatico_only;narrow=false; compress=true;  fused=false; build_probe=false; dynamic=false ;;
  esac
  stage_dir="$output/stage${stage}-${label}"
  sql="SET enable_duckdb_fallback = false;"
  pin_option=--no-pin-compression
  if [[ "$compress" == true ]]; then pin_option=--pin-compression; fi
  command=("$pixi" run --locked python test/tpch_performance/performance_test.py
    --input "$dataset" --scale-factor 1000 --engine gpu --profile hot
    --iterations 1 --queries 1-22 --config "$stage_dir/config.yaml" --pin host
    "$pin_option" --output "$stage_dir" --name "sf1000_host_ablation_stage${stage}")
  echo "Stage $stage ($label): narrowing=$narrow compressed_pin=$compress fused_decode=$fused BUILD_PROBE=$build_probe dynamic_filter=$dynamic"
  if [[ "$dry_run" == true ]]; then
    printf 'SIRIUS_PRE_SQL=%q ' "$sql"
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi
  mkdir -p "$stage_dir"
  cp "$config" "$stage_dir/config.yaml"
  {
    printf '\n  operator_params:\n'
    printf '    enable_compressed_materialization: %s\n' "$narrow"
    printf '    enable_fused_scan_filter: %s\n' "$fused"
    printf '    enable_dynamic_filter: %s\n' "$dynamic"
    if [[ "$build_probe" == false ]]; then
      printf '    max_build_hash_table_bytes: 0\n'
      printf '    max_build_probe_resident_bytes: 0\n'
    fi
  } >> "$stage_dir/config.yaml"
  printf '%s\n' "$sql" > "$stage_dir/settings.sql"
  printf 'SIRIUS_PRE_SQL=%q SIRIUS_PIN_COMPRESSION_TABLES=%q ' \
    "$sql" "$SIRIUS_PIN_COMPRESSION_TABLES" > "$stage_dir/command.sh"
  printf '%q ' "${command[@]}" >> "$stage_dir/command.sh"
  printf '\n' >> "$stage_dir/command.sh"
  export SIRIUS_PRE_SQL="$sql"
  set +e
  timeout --signal=TERM --kill-after=30s 600s "${command[@]}" \
    > "$stage_dir/progress.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$stage_dir/exit-code"
  echo "Stage $stage finished with exit code $status: $stage_dir"
  if (( status != 0 )); then failed=1; fi
done
exit "$failed"
