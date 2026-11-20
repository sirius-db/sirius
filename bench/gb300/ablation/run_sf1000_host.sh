#!/usr/bin/env bash
# Cumulative SF1000 host-pin ablation on the GB300: baseline, then five switches.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
dry_run=false
comparison=false
all_on_comparison=false
join_first_order=false
narrow_first_order=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    --baseline-vs-simpatico) comparison=true ;;
    --all-on-vs-simpatico) all_on_comparison=true ;;
    --join-first-order) join_first_order=true ;;
    --narrow-first-order) narrow_first_order=true ;;
    *) echo "usage: $0 [--dry-run] [--baseline-vs-simpatico|--all-on-vs-simpatico|--join-first-order|--narrow-first-order] (set ABLATION_OUTPUT to choose the output directory)" >&2; exit 2 ;;
  esac
done
selected_modes=0
for selected in "$comparison" "$all_on_comparison" "$join_first_order" "$narrow_first_order"; do
  if [[ "$selected" == true ]]; then ((selected_modes+=1)); fi
done
if (( selected_modes > 1 )); then
  echo "Choose only one run mode" >&2
  exit 2
fi
dataset="${DATASET_ROOT:-/mnt/datasets}/tpch_parquet_sf1000"
config="$repo/bench/gb300/configs/pinned.yaml"
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
if [[ "$all_on_comparison" == true ]]; then stages=(7 8); fi
if [[ "$join_first_order" == true ]]; then stages=(9 10 11 12 13 14); fi
if [[ "$narrow_first_order" == true ]]; then stages=(15 16 17 18 19 20); fi
for stage in "${stages[@]}"; do
  case "$stage" in
    0) label=all_off;       narrow=false; compress=false; fused=false; build_probe=false; dynamic=false ;;
    1) label=type_narrow;   narrow=true;  compress=false; fused=false; build_probe=false; dynamic=false ;;
    2) label=compressed_pin;narrow=true;  compress=true;  fused=false; build_probe=false; dynamic=false ;;
    3) label=fused_decode;  narrow=true;  compress=true;  fused=true;  build_probe=false; dynamic=false ;;
    4) label=build_probe;   narrow=true;  compress=true;  fused=true;  build_probe=true;  dynamic=false ;;
    5) label=dynamic_filter;narrow=true;  compress=true;  fused=true;  build_probe=true;  dynamic=true ;;
    6) label=simpatico_only;narrow=false; compress=true;  fused=false; build_probe=false; dynamic=false ;;
    7) label=all_on_uncompressed;narrow=true; compress=false; fused=true; build_probe=true; dynamic=true ;;
    8) label=all_on_compressed;narrow=true; compress=true; fused=true; build_probe=true; dynamic=true ;;
    9) label=all_off;        narrow=false; compress=false; fused=false; build_probe=false; dynamic=false ;;
    10) label=build_probe;   narrow=false; compress=false; fused=false; build_probe=true; dynamic=false ;;
    11) label=dynamic_filter;narrow=false; compress=false; fused=false; build_probe=true; dynamic=true ;;
    12) label=type_narrow;   narrow=true;  compress=false; fused=false; build_probe=true; dynamic=true ;;
    13) label=compressed_pin;narrow=true;  compress=true;  fused=false; build_probe=true; dynamic=true ;;
    14) label=fused_decode;  narrow=true;  compress=true;  fused=true;  build_probe=true; dynamic=true ;;
    15) label=all_off;        narrow=false; compress=false; fused=false; build_probe=false; dynamic=false ;;
    16) label=type_narrow;    narrow=true;  compress=false; fused=false; build_probe=false; dynamic=false ;;
    17) label=build_probe;    narrow=true;  compress=false; fused=false; build_probe=true;  dynamic=false ;;
    18) label=dynamic_filter; narrow=true;  compress=false; fused=false; build_probe=true;  dynamic=true ;;
    19) label=compressed_pin; narrow=true;  compress=true;  fused=false; build_probe=true;  dynamic=true ;;
    20) label=fused_decode;   narrow=true;  compress=true;  fused=true;  build_probe=true;  dynamic=true ;;
  esac
  stage_dir="$output/stage${stage}-${label}"
  sql="SET enable_duckdb_fallback = false;"
  pin_option=--no-pin-compression
  stage_compression_tables=""
  if [[ "$compress" == true ]]; then
    pin_option=--pin-compression
    stage_compression_tables=lineitem,orders
  fi
  # An explicit per-table compression=true overrides the session's false
  # setting, so the allowlist must be empty for every uncompressed stage.
  export SIRIUS_PIN_COMPRESSION_TABLES="$stage_compression_tables"
  command=("$pixi" run --locked python test/tpch_performance/performance_test.py
    --input "$dataset" --scale-factor 1000 --engine gpu --profile hot
    --iterations 1 --queries 1-22 --config "$stage_dir/config.yaml" --pin host
    "$pin_option" --output "$stage_dir" --name "sf1000_host_ablation_stage${stage}")
  echo "Stage $stage ($label): narrowing=$narrow compressed_pin=$compress fused_decode=$fused BUILD_PROBE=$build_probe dynamic_filter=$dynamic pin_compression_tables=${stage_compression_tables:-none}"
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
  if (( status == 0 )); then
    actual_compression=false
    if rg -q 'SQL: CALL pin_table.*compression=true' "$stage_dir" -g 'sirius_*.log'; then
      actual_compression=true
    fi
    if [[ "$actual_compression" != "$compress" ]]; then
      echo "Stage $stage pin log disagrees with expected compressed_pin=$compress" >&2
      failed=1
    fi
  fi
done
exit "$failed"
