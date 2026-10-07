#!/usr/bin/env bash
# One TPC-H harness run with a rendered Sirius config. By default the inputs come
# straight from S3 with no pinning, so every query scans over the REST backend and
# scan throughput against the NIC (~12.5 GB/s for 100 Gbit/s) is directly visible.
# Run from the repo root:
#
#   bash bench/s3-sf1000/run.sh                                   # SF1000 from S3
#   SPILL_COMPRESSION=1 NAME=spillcomp bash bench/s3-sf1000/run.sh
#   SF=3000 QUERIES=6 bash bench/s3-sf1000/run.sh
#
# Local dataset with pinned input tables (the earlier RTX PRO 6000 setup; see
# docs/compression/spill-compression-handoff.md for that box's settings):
#
#   DATA=/mnt/datasets/tpch/sf1000 PIN=gpu PIN_COMPRESSION=1 bash bench/s3-sf1000/run.sh
#   DATA=... PIN=mixed PIN_HOST_TABLES="PART CUSTOMER" bash bench/s3-sf1000/run.sh
#
# Knobs here (env): SF, DATA, QUERIES, ITERS, NAME,
#   PIN=none|gpu|host|mixed      input-table pinning; S3 inputs cannot be pinned
#   PIN_HOST_TABLES="..."        PIN=mixed: tables pinned to host, rest to GPU
#   PIN_COMPRESSION=1            compress pinned tables with the offline plans
#   FUSED_SCAN_FILTER=1          experimental fused filter in the compressed-scan decode
#   EXPR_EVAL=ast_jit            cuDF JIT expression evaluator instead of the AST walker
# All render-config.sh knobs pass through. S3 inputs need a live `aws sso login`:
# the config is re-rendered with fresh credentials on every invocation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export PATH="$HOME/.pixi/bin:$PATH"

SF="${SF:-1000}"
DATA="${DATA:-s3://sirius-s3-test/datasets/tpch_sf$SF/}"
QUERIES="${QUERIES:-1-22}"
ITERS="${ITERS:-2}"
PIN="${PIN:-none}"
export NAME="${NAME:-baseline}"
export SPILL_DIR="${SPILL_DIR:-/mnt/nvme/sirius_spill}"

case "$DATA" in
  s3://*)
    export NEED_S3=1
    # pin_table globs local files; an s3:// input cannot be pinned.
    [ "$PIN" = none ] || { echo "ERROR: PIN=$PIN needs a local DATA path"; exit 1; } ;;
  *)
    export NEED_S3=0
    [ -d "$DATA" ] || { echo "ERROR: dataset not found at $DATA"; exit 1; } ;;
esac

case "$SPILL_DIR" in
  /mnt/nvme/*) mountpoint -q /mnt/nvme ||
                 { echo "ERROR: /mnt/nvme not mounted; run setup-nvme-scratch.sh"; exit 1; } ;;
esac
mkdir -p "$SPILL_DIR"

CFG=$(bash "$HERE/render-config.sh")
export SIRIUS_CONFIG_FILE="$CFG"

# debug emits the downgrade executor's per-request spill summaries, which is
# what tells the spilling queries apart; every arm runs at the same level.
export SIRIUS_LOG_LEVEL="${SIRIUS_LOG_LEVEL:-debug}"
# Per-query REST reactor counters (bytes, requests, fused reads) to stderr at
# query end -- the scan-throughput evidence for NIC saturation.
export SIRIUS_IO_PROFILE=1
# No CPU replay on a GPU OOM: it would not be a Sirius measurement, it is not
# supported for S3 inputs anyway, and DuckDB has only the host memory the pinned
# tier leaves (a CPU q18 at SF1000 once OOM-killed a 62 GB box this way).
PRE_SQL="${SIRIUS_PRE_SQL:-SET enable_duckdb_fallback = false}"
if [ -n "${EXPR_EVAL:-}" ]; then
  PRE_SQL="SET expression_evaluator_strategy = '$EXPR_EVAL'; $PRE_SQL"
fi
export SIRIUS_PRE_SQL="$PRE_SQL"
# Spread each read over every REST reactor rather than the historical 2: with 2,
# extra reactors add no capacity to any one read (see templated_ioctx.hpp).
export SIRIUS_DISPATCH_FANOUT="${SIRIUS_DISPATCH_FANOUT:-all}"
[ "${FUSED_SCAN_FILTER:-0}" = 1 ] && export SIRIUS_EXP_FUSED_SCAN_FILTER=1

PIN_ARGS=(--pin none)
if [ "$PIN" != none ]; then
  HARNESS_PIN="$PIN"
  if [ "$PIN" = mixed ]; then
    # Per-table tiers; the harness pin flag only has to name some tier.
    HARNESS_PIN=gpu
    for t in LINEITEM ORDERS PART CUSTOMER SUPPLIER NATION REGION PARTSUPP; do
      case " ${PIN_HOST_TABLES:-} " in *" $t "*) tier=host ;; *) tier=gpu ;; esac
      export "SIRIUS_PIN_TIER_$t=$tier"
    done
  fi
  PIN_ARGS=(--pin "$HARNESS_PIN")
  if [ "${PIN_COMPRESSION:-0}" = 1 ]; then
    PIN_ARGS+=(--pin-compression --compression-plan-dir
               "${PLAN_DIR:-$REPO/src/compression/simpatico_codegen/plans/tpch_sf1000}")
  fi
fi

echo "data      : $DATA"
echo "config    : $CFG"
echo "pin       : $PIN${PIN_HOST_TABLES:+ (host: $PIN_HOST_TABLES)}"
echo "run name  : sf${SF}_$NAME  (queries $QUERIES, $ITERS iterations)"

cd "$REPO"
# Sample the NIC once a second alongside the run; the receive rate is the
# ground truth for "are we saturating 100 Gbit/s".
# The default-route interface, from the kernel's routing table (no `ip` needed).
NIC=$(awk '$2=="00000000"{print $1; exit}' /proc/net/route)
STAMP=$(date +%Y%m%d_%H%M%S)
NICLOG="$REPO/test/tpch_performance/output/nic_sf${SF}_${NAME}_$STAMP.csv"
mkdir -p "$(dirname "$NICLOG")"
# Also the aggregate CPU jiffies (busy, total), so a NIC plateau can be told
# apart from the REST reactors running out of CPU (TLS + curl).
(
  echo "epoch_s,rx_bytes,cpu_busy,cpu_total,mem_avail_kb"
  while :; do
    read -r _ u n s i w q sq st _ < /proc/stat
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    echo "$(date +%s),$(cat /sys/class/net/$NIC/statistics/rx_bytes),$((u+n+s+q+sq+st)),$((u+n+s+i+w+q+sq+st)),$avail"
    sleep 1
  done
) > "$NICLOG" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

pixi run python test/tpch_performance/performance_test.py \
  --input "$DATA" --scale-factor "$SF" \
  --iterations "$ITERS" --engine gpu "${PIN_ARGS[@]}" \
  --queries "$QUERIES" --name "sf${SF}_$NAME"

echo "NIC samples: $NICLOG"
