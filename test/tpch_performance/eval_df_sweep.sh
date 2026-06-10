#!/usr/bin/env bash
# Sweep dynamic-filter (enable, wait_ms) configs for one query on SF30, reporting
# warm wall time + how many probe splits actually got the filter at read / post-decode.
set -uo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
PQ="$PROJECT_DIR/test_datasets/tpch_parquet_sf30"
export SIRIUS_CONFIG_FILE="$PROJECT_DIR/test/cpp/integration/integration.yaml"
QUERY_FILE="$1"; ITERS="${2:-3}"

li=$(ls "$PQ"/lineitem/*.parquet | sed "s/.*/'&'/" | paste -sd, -)
od=$(ls "$PQ"/orders/*.parquet   | sed "s/.*/'&'/" | paste -sd, -)
pt=$(ls "$PQ"/part/*.parquet     | sed "s/.*/'&'/" | paste -sd, -)
VIEWS="CREATE VIEW lineitem AS SELECT * FROM read_parquet([$li]);
CREATE VIEW orders AS SELECT * FROM read_parquet([$od]);
CREATE VIEW part AS SELECT * FROM read_parquet([$pt]);"

run_cfg() {  # label enable wait_ms
  local label="$1" enable="$2" wait="$3"
  local ld="/tmp/dfsw_$label"; rm -rf "$ld"; mkdir -p "$ld"
  local sql="/tmp/dfsw_$label.sql"
  {
    echo "$VIEWS"
    echo "SET enable_dynamic_filter_pushdown=$enable;"
    echo "SET dynamic_filter_wait_ms=$wait;"
    echo ".timer on"
    for i in $(seq 1 "$ITERS"); do cat "$QUERY_FILE"; done
  } > "$sql"
  local out
  out=$(SIRIUS_LOG_DIR="$ld" SIRIUS_LOG_LEVEL=debug timeout 300 "$DUCKDB" -f "$sql" 2>&1)
  local times merged postd pushed result
  times=$(echo "$out" | grep -oE "real [0-9.]+" | awk '{printf "%s ", $2}')
  pushed=$(grep -h "Pushed .* dynamic filter" "$ld"/*.log 2>/dev/null | head -1 | grep -oE "\([0-9]+ build rows" | grep -oE "[0-9]+")
  merged=$(grep -hc "Merged dynamic filter fragments" "$ld"/*.log 2>/dev/null | paste -sd+ - | bc 2>/dev/null || echo 0)
  postd=$(grep -h "Applied dynamic filter post-decode" "$ld"/*.log 2>/dev/null | grep -oE "[0-9]+ -> [0-9]+" | head -1)
  result=$(echo "$out" | grep -E "^│ *[0-9]" | head -1 | tr -s ' ')
  printf "%-16s build_rows=%-9s merged@read=%-4s postdec=[%s]\n                 times=[ %s]\n                 result=%s\n" \
    "$label" "${pushed:-0}" "$merged" "${postd:-none}" "$times" "$result"
}

echo "############ $(basename "$QUERY_FILE")  ($ITERS iters) ############"
run_cfg "off"      false 0
run_cfg "on_w0"    true  0
run_cfg "on_w100"  true  100
run_cfg "on_w400"  true  400
