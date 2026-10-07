#!/usr/bin/env bash
# TPC-H SF1000 (later SF3000) straight from S3, no pinning: every query scans
# its inputs over the REST backend, so scan throughput vs the 100 Gbit/s NIC
# (~12.5 GB/s) is directly visible. Run from the repo root:
#
#   bash bench/s3-sf1000/run.sh                          # 1 MiB host blocks, baseline
#   HOST_BLOCK_SIZE=4194304 NAME=blk4m bash bench/s3-sf1000/run.sh
#   SPILL_COMPRESSION=1 NAME=spillcomp bash bench/s3-sf1000/run.sh
#
# All render-config.sh knobs pass through (thread counts, reactors, batch size).
# Needs a live `aws sso login`: the config is re-rendered with fresh
# credentials on every invocation, and a run must finish before they expire.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export PATH="$HOME/.pixi/bin:$PATH"

SF="${SF:-1000}"
DATA="${DATA:-s3://sirius-s3-test/datasets/tpch_sf$SF/}"
QUERIES="${QUERIES:-1-22}"
ITERS="${ITERS:-2}"
export NAME="${NAME:-baseline}"

mountpoint -q /mnt/nvme || { echo "ERROR: /mnt/nvme not mounted; run setup-nvme-scratch.sh"; exit 1; }
mkdir -p /mnt/nvme/sirius_spill

CFG=$(bash "$HERE/render-config.sh")
export SIRIUS_CONFIG_FILE="$CFG"

# debug emits the downgrade executor's per-request spill summaries, which is
# what tells the spilling queries apart; every arm runs at the same level.
export SIRIUS_LOG_LEVEL="${SIRIUS_LOG_LEVEL:-debug}"
# Per-query REST reactor counters (bytes, requests, fused reads) to stderr at
# query end -- the scan-throughput evidence for NIC saturation.
export SIRIUS_IO_PROFILE=1
# No CPU replay on a GPU OOM: it would not be a Sirius measurement, and DuckDB
# has only the ~20 GB the pinned host tier leaves.
export SIRIUS_PRE_SQL="${SIRIUS_PRE_SQL:-SET enable_duckdb_fallback = false}"
# Spread each read over every REST reactor rather than the historical 2: with 2,
# extra reactors add no capacity to any one read (see templated_ioctx.hpp).
export SIRIUS_DISPATCH_FANOUT="${SIRIUS_DISPATCH_FANOUT:-all}"

echo "data      : $DATA"
echo "config    : $CFG"
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
  --iterations "$ITERS" --engine gpu --pin none \
  --queries "$QUERIES" --name "sf${SF}_$NAME"

echo "NIC samples: $NICLOG"
