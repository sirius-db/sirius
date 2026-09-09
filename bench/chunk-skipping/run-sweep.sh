#!/usr/bin/env bash
# Zone-map pruning: benefit vs. pin-chunk granularity, at SF100.
#
# Sweeps scan_task_batch_size (a free granularity dial: smaller batch => smaller pin chunk =>
# finer zone maps) crossed with enable_pinned_zone_map_pruning on/off, on the clustered SF100
# dataset. The number that matters is the ON-vs-OFF delta AT A FIXED batch size: smaller batches
# cost throughput on their own, so absolute times across batch sizes are confounded but the
# delta is not.
#
#   pixi run bash bench/chunk-skipping/run-sweep.sh
#
# Env:
#   DATA      dataset root (default the clustered SF100)
#   BATCHES   space-separated scan_task_batch_size values
#   ITERS     iterations per query (default 3, reported best-of)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-/datasets/tpch_sf100_sorted}"
PLANS="${PLANS:-$REPO/bench/sf1000-repro/plans}"
BATCHES="${BATCHES:-8GB 2GB 512MB 128MB}"
ITERS="${ITERS:-3}"
OUT="${OUT:-$REPO/bench/chunk-skipping/results}"

[ -d "$DATA" ] || { echo "ERROR: no dataset at $DATA (set DATA=)"; exit 1; }
mkdir -p "$OUT" /datasets/.sirius_disk_memory

export SIRIUS_EXP_LATE_MAT="${SIRIUS_EXP_LATE_MAT:-1}"
export SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS="${SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS:-c_custkey,n_name,n_nationkey}"
export SIRIUS_EXP_FUSED_SCAN_FILTER="${SIRIUS_EXP_FUSED_SCAN_FILTER:-1}"
for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
  export "SIRIUS_PIN_TIER_$t=gpu"
done

cd "$REPO"
for batch in $BATCHES; do
  for prune in on off; do
    cfg="$OUT/cfg-$batch-$prune.yaml"
    # enable_pinned_zone_map_pruning gates BOTH pin-time capture and serve-time pruning, so
    # 'off' is a statless pin — a clean "feature absent" arm. The pinned DATA is identical
    # either way (narrowing is driven by pinned_column_types, not by this flag); only the
    # sidecar differs.
    # Compute the flag BEFORE the sed. Do not inline a $( ... && ... ) here: '&' is special in
    # sed's replacement text, escaping it as '\&\&' breaks the shell's AND operator, and the
    # substitution then silently yields "false" for BOTH arms — which is exactly how the first
    # run of this sweep produced a meaningless result.
    if [ "$prune" = on ]; then flag=true; else flag=false; fi
    sed "s|^    operator_params:|    operator_params:\n        scan_task_batch_size: $batch\n        enable_pinned_zone_map_pruning: $flag|" \
      "$HERE/sirius-sf100.yaml" > "$cfg"
    grep -q "enable_pinned_zone_map_pruning: $flag" "$cfg" || { echo "ERROR: config injection failed"; exit 1; }
    export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET pin_table_input_compression_plan_dir = '$PLANS'; \
SET expression_evaluator_strategy = 'ast_jit'"
    name="sf100_$(basename "$DATA")_${batch}_prune-${prune}"
    echo "################ $name"
    pixi run python test/tpch_performance/performance_test.py \
      --input "$DATA" --data-source parquet \
      --mode grouped --iterations "$ITERS" --engine gpu --pin gpu \
      --queries 1-22 --config "$cfg" --name "$name" --output "$OUT"

    # Guard: the ON arm MUST show pruning. A silently-misconfigured arm would otherwise be
    # reported as "the feature does not help", which is the one wrong answer this sweep can give.
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
