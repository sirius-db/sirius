#!/usr/bin/env bash
# The three-arm comparison of HANDOVER-dev-baseline.md: what did the zonemap project buy?
#
#   dev parquet                upstream/dev, the starting point
#   ours parquet + cluster_by  what the zone-map work gave PARQUET (no new format needed)
#   ours .hpln  + cluster_by   what the new format adds on top
#
# Both builds must sit on the SAME cucascade (e9929fff) -- the merge bumped it, and every number
# in CHUNK_SKIPPING_PLAN.md predating the merge was taken against the older one. Checked below
# rather than assumed.
#
#   /home/nvidia/joost/bench-lock.sh bash bench/chunk-skipping/run-dev-baseline.sh
#
# MUST run under bench-lock.sh: this config pins 471.2 GB of host memory on a 494 GB box.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DEV_REPO="${DEV_REPO:-/home/nvidia/joost/sirius-dev}"
DEV_EXT="$DEV_REPO/build/release/extension/sirius/sirius.duckdb_extension"

PARQUET="${PARQUET:-/datasets/tpch_sf1000}"
HPLN="${HPLN:-/datasets/tpch_sf1000_hpln_cluster}"
ITERS="${ITERS:-3}"
OUT="${OUT:-$REPO/bench/chunk-skipping/results-dev-baseline}"

# The 8 GB batch config: measured best for host pins post-clustering (2/4/8 GB = 9.830/9.073/8.958).
export SIRIUS_CONFIG_FILE="${SIRIUS_CONFIG_FILE:-$REPO/bench/chunk-skipping/sirius-sf1000-pin.yaml}"
# The fused-scan gate is asymmetric -- it gates .hpln's decode-time filtering entirely while
# parquet's reader filter is untouched -- so a run without it understates .hpln on join-heavy
# queries. Late-mat is inert on host pins (1512 declines, parquet included); set for parity.
export SIRIUS_EXP_FUSED_SCAN_FILTER=1
export SIRIUS_EXP_LATE_MAT=1
export SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS=c_custkey,n_name,n_nationkey
# ast_interpret, NOT ast_jit: ast_jit helps only plain parquet and is a cold regression (+13.9%)
# and hurts both clustered arms. It is also the default; set explicitly so the log records it.
export SIRIUS_PRE_SQL="SET expression_evaluator_strategy = 'ast_interpret'"

[ -x "$DEV_EXT" ] || { echo "ERROR: no dev build at $DEV_EXT"; exit 1; }
[ -d "$PARQUET" ] || { echo "ERROR: no parquet dataset at $PARQUET"; exit 1; }
[ -d "$HPLN" ]    || { echo "ERROR: no .hpln dataset at $HPLN"; exit 1; }

ours_cc=$(cd "$REPO"     && git submodule status cucascade | awk '{print $1}' | tr -d '+-')
dev_cc=$(cd "$DEV_REPO"  && git submodule status cucascade | awk '{print $1}' | tr -d '+-')
[ "$ours_cc" = "$dev_cc" ] || { echo "ERROR: cucascade differs (ours $ours_cc, dev $dev_cc)"; exit 1; }
echo "cucascade matched: $ours_cc"
echo "dev HEAD: $(cd "$DEV_REPO" && git log --oneline -1)"

mkdir -p "$OUT" /datasets/.sirius_disk_memory

require_idle_gpu() {
  for _ in $(seq 1 180); do
    local mem util
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -dc '0-9')
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -dc '0-9')
    if [ "${mem:-9999}" -lt 2048 ] && [ "${util:-100}" -lt 20 ]; then return 0; fi
    echo "waiting for an idle GPU (used ${mem} MiB, util ${util}%)"
    sleep 20
  done
  echo "ERROR: GPU still busy after 60 min; refusing to measure"; exit 1
}

cd "$REPO"

PHASES="${PHASES:-pinned cold}"
has_phase() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

if has_phase pinned; then
# dev first, in its own process: it is a different .so, and it has no .hpln at all, so its only
# possible arm is plain parquet. A pin is cold by nature, so the page cache is dropped for it.
require_idle_gpu
echo "################ arm: dev parquet"
SIRIUS_EXT_PATH="$DEV_EXT" pixi run python bench/chunk-skipping/hpln-pin-bench.py \
  --arms parquet --parquet "$PARQUET" --tier host --iterations "$ITERS" \
  --drop-cache --out "$OUT/dev" 2>&1 | tee "$OUT/dev.log"

# ours: all three of our arms in one process so the harness cross-checks their results against
# each other query by query (arm 0 is the reference).
require_idle_gpu
echo "################ arms: ours parquet, parquet+cluster_by, .hpln+cluster_by"
pixi run python bench/chunk-skipping/hpln-pin-bench.py \
  --arms parquet,parquet-clustered,hpln-unsorted \
  --parquet "$PARQUET" --hpln-unsorted "$HPLN" \
  --tier host --iterations "$ITERS" --drop-cache --out "$OUT/ours" 2>&1 | tee "$OUT/ours.log"
fi

# The COLD arms: no pin at all, the page cache dropped before every query, so each one is measured
# against storage. A different config -- the pinned one reserves 471.2 GB of host memory for pins
# that do not exist here. `dev` has no .hpln, so its .hpln column fails per query by design and
# only its parquet column is read; that column is the check that the untouched cold parquet path
# did not move under the merge.
if has_phase cold; then
COLD_CFG="${COLD_CFG:-$REPO/bench/chunk-skipping/sirius-sf1000-hpln.yaml}"
require_idle_gpu
echo "################ cold: dev parquet (its .hpln column fails by design)"
SIRIUS_EXT_PATH="$DEV_EXT" pixi run python bench/chunk-skipping/hpln-suite.py \
  --parquet "$PARQUET" --hpln "$HPLN" --config "$COLD_CFG" --drop-cache \
  2>&1 | tee "$OUT/cold-dev.log"

require_idle_gpu
echo "################ cold: ours parquet vs .hpln + cluster_by"
pixi run python bench/chunk-skipping/hpln-suite.py \
  --parquet "$PARQUET" --hpln "$HPLN" --config "$COLD_CFG" --drop-cache \
  2>&1 | tee "$OUT/cold-ours.log"
fi

echo "done; reports in $OUT/{dev,ours}/report.json, cold logs in $OUT/cold-*.log"
