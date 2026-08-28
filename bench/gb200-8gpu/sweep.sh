#!/usr/bin/env bash
# TPC-H sweep against an already-running 8-CN cluster. Does not launch CNs.
#
#   ./bench/gb200-8gpu/sweep.sh 1000
#   SCALE_FACTOR=3000 ./bench/gb200-8gpu/sweep.sh
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
export SCALE_FACTOR=${SCALE_FACTOR:-${1:?usage: sweep.sh 1000|3000|10000}}
KNOBS=$HERE/sf${SCALE_FACTOR}/env.sh
[ -f "$KNOBS" ] || { echo "sweep: no knobs at $KNOBS" >&2; exit 1; }
# shellcheck disable=SC1090
. "$KNOBS"

SR=$REPO/experimental/starrocks
# shellcheck disable=SC1091
source /scratch/prestouser/aocsa/env.sh
export PATH=$SR/.pixi/envs/default/bin:$PATH
MYSQL=$SR/.pixi/envs/default/bin/mysql

"$MYSQL" -h127.0.0.1 -P9030 -uroot -e \
  "SET GLOBAL enable_pipeline_engine=true; SET GLOBAL pipeline_dop=${PIPELINE_DOP}; SET GLOBAL query_timeout=${FE_QUERY_TIMEOUT};"

OUT=${OUT:-/scratch/prestouser/aocsa/bench-results/sf${SCALE_FACTOR}-8gpu-2host-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
{
  echo "when=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "SCALE_FACTOR=$SCALE_FACTOR"
  echo "GPU_MEM=$GPU_MEM HOST_MEM=$HOST_MEM STAGING=$STAGING"
  echo "pipeline_dop=$PIPELINE_DOP datasource=$SIRIUS_CN_USE_SIRIUS_DATASOURCE"
  echo "TPCH_DATA=$TPCH_DATA Q11_FRACTION=$Q11_FRACTION"
} > "$OUT/MANIFEST.txt"

Q11=$SR/benchmarks/tpch/queries/q11.sql
cp "$Q11" "$OUT/q11.sql.orig"
restore() { git -C "$REPO" checkout HEAD -- experimental/starrocks/benchmarks/tpch/queries/q11.sql || true; }
trap restore EXIT
sed -i "s/0\\.0001000000/${Q11_FRACTION}/" "$Q11"

cd "$SR"
TPCH_DATA=$TPCH_DATA FE_PORT=9030 \
QUERY_TIMEOUT=$QUERY_TIMEOUT COLD_TIMEOUT=$COLD_TIMEOUT MIN_BACKENDS=$MIN_BACKENDS \
  ./benchmarks/tpch/bench.sh --cold "$OUT/timings.csv" 3 \
  | tee "$OUT/bench.log"

restore
trap - EXIT
echo "sweep: $OUT/timings.csv"
