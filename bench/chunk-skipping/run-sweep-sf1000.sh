#!/usr/bin/env bash
# Phase 1 validation at production scale (CHUNK_SKIPPING_PLAN.md §9).
#
# Same experiment as run-sweep.sh, at SF1000 on the clustered dataset, reusing the tuned
# sf1000-repro config so the only variables are scan_task_batch_size and
# enable_pinned_zone_map_pruning. Read the ON-vs-OFF delta at a fixed batch size.
#
#   /home/nvidia/joost/bench-lock.sh bash bench/chunk-skipping/run-sweep-sf1000.sh
#
# MUST run under bench-lock.sh: this config pins 471.2 GB of host memory on a 494 GB box.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-/datasets/tpch_sf1000_sorted}"
PLANS="${PLANS:-$REPO/bench/sf1000-repro/plans}"
BASE_CFG="${BASE_CFG:-$REPO/bench/sf1000-repro/sirius-sf1000.yaml}"
BATCHES="${BATCHES:-8GB 2GB}"
# TIER=host is the experiment that matters for fetch-bound skipping: on a host pin every scan
# batch pays payload H2D -> sync -> decode on the critical path, so a skipped chunk saves the
# transfer, not just the decode. On a GPU pin the payload is already resident (ceiling ~2.4%,
# CHUNK_SKIPPING_PLAN.md §3.9).
TIER="${TIER:-gpu}"
ITERS="${ITERS:-3}"
OUT="${OUT:-$REPO/bench/chunk-skipping/results-sf1000-$TIER}"
DOWNGRADE_DIR="${DOWNGRADE_DIR:-/datasets/.sirius_disk_memory}"

[ -d "$DATA" ] || { echo "ERROR: no dataset at $DATA (set DATA=)"; exit 1; }
mkdir -p "$OUT" "$DOWNGRADE_DIR"

export SIRIUS_EXP_LATE_MAT="${SIRIUS_EXP_LATE_MAT:-1}"
export SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS="${SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS:-c_custkey,n_name,n_nationkey}"
export SIRIUS_EXP_FUSED_SCAN_FILTER="${SIRIUS_EXP_FUSED_SCAN_FILTER:-1}"
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export "SIRIUS_PIN_TIER_$t=$TIER"
done
export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '$PLANS'; \
SET expression_evaluator_strategy = 'ast_jit'"

cd "$REPO"
for batch in $BATCHES; do
  for prune in on off; do
    if [ "$prune" = on ]; then flag=true; else flag=false; fi
    cfg="$OUT/cfg-$batch-$prune.yaml"
    # Replace the base config's own scan_task_batch_size and append the pruning flag.
    # Computing the flag BEFORE the sed matters: '&' is special in sed replacement text, so an
    # inlined $( ... && ... ) silently yields "false" for both arms (see §9 Phase 1 process note).
    # The base config's downgrade_root_dirs points at a /localhome path that does not exist on
    # this box; repoint it so a downgrade does not fail on a missing directory.
    sed -e "s|^        scan_task_batch_size: .*|        scan_task_batch_size: $batch\n        enable_pinned_zone_map_pruning: $flag|" \
        -e "s|^            downgrade_root_dirs: .*|            downgrade_root_dirs: \"$DOWNGRADE_DIR\"|" \
      "$BASE_CFG" > "$cfg"
    grep -q "enable_pinned_zone_map_pruning: $flag" "$cfg" || { echo "ERROR: config injection failed"; exit 1; }
    grep -q "scan_task_batch_size: $batch" "$cfg" || { echo "ERROR: batch injection failed"; exit 1; }

    name="sf1000_$(basename "$DATA")_${TIER}_${batch}_prune-${prune}"
    echo "################ $name"
    pixi run python test/tpch_performance/performance_test.py \
      --input "$DATA" --data-source parquet \
      --mode grouped --iterations "$ITERS" --engine gpu --pin "$TIER" \
      --queries 1-22 --config "$cfg" --name "$name" --output "$OUT"

    if [ "$prune" = on ]; then
      run_dir=$(ls -dt "$OUT"/tpch_*"$name" 2>/dev/null | head -1)
      if ! grep -rq "zone-map pruning for pinned entry" "$run_dir/log_dir" 2>/dev/null; then
        echo "ERROR: prune-on arm at $batch pruned nothing — check the config and the pin tier"
        exit 1
      fi
    fi
  done
done
echo "results under $OUT"
