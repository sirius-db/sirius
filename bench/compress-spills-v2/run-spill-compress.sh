#!/usr/bin/env bash
# Spill-compression TPC-H SF1000 arm: identical to run-baseline.sh except that
# the config turns on compression.enable_spill_compression and reserves the
# 4 GiB compression device arena. Run from the repo root:
#   pixi run bash bench/compress-spills-v2/run-spill-compress.sh
#
# The arena has to come from the config file: SiriusContext::initialize() is the
# only caller of init_compression_device_pool(), so `SET spill_compression = true`
# at session start flips the flag but leaves the encoder allocating from the very
# query pool whose exhaustion triggered the spill -- the documented pathology
# (compression latching off/on, downgrade request storm, query killed).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-/mnt/datasets/tpch/sf1000}"
CFG="${CFG:-$HERE/sirius-rtx6000-spill.yaml}"
PLANS="${PLANS:-$REPO/src/compression/simpatico_codegen/plans/tpch_sf1000}"
NAME="${NAME:-spill_compress_enabled}"
QUERIES="${QUERIES:-1-22}"
ITERS="${ITERS:-3}"
# gpu | host -- which cache tier the input tables are pinned into.
PIN="${PIN:-gpu}"
# PIN=mixed: per-table tiering. PIN_HOST_TABLES lists the tables that go to the
# host tier; everything else goes to the GPU tier. Measured at SF1000: q1's
# compressed lineitem pin is 71.2 GB on the GPU tier -- larger than this box's
# entire 62 GB of RAM -- so LINEITEM cannot be host-pinned here at any ceiling.
# The workable split is therefore the inverse of "compressed tables to host":
# keep the big compressed tables (lineitem, orders) on the GPU and push the
# small ones to host, freeing GPU tier for the query itself.
PIN_HOST_TABLES="${PIN_HOST_TABLES:-PART CUSTOMER SUPPLIER NATION REGION PARTSUPP}"


case "$DATA" in
  s3://*) : ;;  # remote: resolved by sirius_httpfs at bind time, not on this fs
  *) [ -d "$DATA" ] || { echo "ERROR: dataset not found at $DATA"; exit 1; } ;;
esac
[ -d "$PLANS" ] || { echo "ERROR: compression plans not found at $PLANS"; exit 1; }
[ -f "$CFG" ]  || { echo "ERROR: config not found at $CFG"; exit 1; }
mkdir -p /mnt/datasets/sirius_spill

export SIRIUS_CONFIG_FILE="$CFG"
export SIRIUS_LOG_LEVEL="${SIRIUS_LOG_LEVEL:-debug}"
export SIRIUS_EXP_FUSED_SCAN_FILTER=1
# enable_duckdb_fallback=false: when the GPU OOMs (q18 at SF1000 does, at
# MERGE_GROUP_BY), the transparent path otherwise replays the query on DuckDB
# CPU inside the same transaction. On this box that is fatal -- the host tier
# pins 40 GB of 62 GB, so a CPU q18 at SF1000 has ~22 GB to work in, and the
# OOM killer took the whole machine down mid-run. We want the GPU error
# surfaced and the query recorded as failed, not silently re-run on the CPU:
# a CPU fallback time would not be a Sirius measurement anyway.
export SIRIUS_PRE_SQL="SET expression_evaluator_strategy = 'ast_jit'; SET enable_duckdb_fallback = false"

# PIN=none leaves the per-table tier vars unset: pin_table is never called, and
# an s3:// input cannot be pinned at all (pin_table globs local files).
if [ "$PIN" != "none" ]; then
  for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
    if [ "$PIN" = "mixed" ]; then
      case " $PIN_HOST_TABLES " in *" $t "*) tier=host ;; *) tier=gpu ;; esac
    else
      tier="$PIN"
    fi
    export "SIRIUS_PIN_TIER_$t=$tier"
  done
fi

echo "data      : $DATA"
echo "config    : $CFG"
HARNESS_PIN="$PIN"
# Compression happens at pin time, so the flag needs a pinned tier.
PIN_COMPRESSION_ARGS="--pin-compression --compression-plan-dir $PLANS"
[ "$PIN" = "none" ] && PIN_COMPRESSION_ARGS=""
[ "$PIN" = "mixed" ] && HARNESS_PIN=gpu
echo "pin tier  : $PIN"
[ "$PIN" = "mixed" ] && echo "host tables: $PIN_HOST_TABLES"
echo "run name  : $NAME"
echo "spill dir : /mnt/datasets/sirius_spill"
echo

cd "$REPO"
# The harness must run in the duckdb-python env: the default env ships the stock
# duckdb 1.5.5 wheel, into which sirius_duckdb_cpp_init throws "Attempted to
# dereference unique_ptr that is NULL". The env-local module is built from the
# repo submodule via:
#   OVERRIDE_GIT_DESCRIBE=v1.5.5 pixi run -e duckdb-python pip install \
#     --no-build-isolation --force-reinstall \
#     --config-settings="cmake.define.DUCKDB_SOURCE_PATH=$REPO/duckdb" ./duckdb-python
# (OVERRIDE_GIT_DESCRIBE is required -- duckdb-python is tagged v1.5.4 and the
# extension refuses to load into a mismatched engine version. --config-settings
# is also required: their build backend asserts config_settings is not None.)
# No --execution flag: the named profiles override the config given here, and
# `--execution hot` additionally drops the OS cache at start.
pixi run -e duckdb-python python test/tpch_performance/performance_test.py \
  --input "$DATA" \
  --iterations "$ITERS" --engine gpu --pin "$HARNESS_PIN" \
  $PIN_COMPRESSION_ARGS \
  --queries "$QUERIES" --name "$NAME"
