#!/usr/bin/env bash
# Run the simpatico GPU compression codecs on one TPC-H SF1000 Parquet part file using the
# Pareto-picked plans in bench/sf1000-repro/plans/ (the same plans run.sh pins with).
#
# Run from the repo root inside the pixi env:
#   pixi run bash bench/sf1000-repro/run-compression.sh <compress|decompress|benchmark|all> [flags]
#
# Examples:
#   pixi run bash bench/sf1000-repro/run-compression.sh all --table lineitem
#   pixi run bash bench/sf1000-repro/run-compression.sh benchmark --input $DATA/orders/part.3.parquet
#   MODE=full-table THREADS=9 pixi run bash bench/sf1000-repro/run-compression.sh benchmark --table orders
#
# Each stage is one `simpatico` invocation and is wall-clocked separately, with JIT-compile
# statistics printed at process exit (SIMPATICO_JIT_STATS). On a ~3 GB part file every stage
# here is seconds to low minutes; if you see many minutes, look at the `compiles` count first
# (cold NVRTC cache, or an unwritable $HOME disabling the on-disk cache) and make sure you
# did not run `simpatico explore` (the beam-search plan sweep) by mistake — that one takes
# tens of minutes per table by design.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:-$HOME/tpch_parquet_sf1000}"          # SF1000 parquet, one dir per table
PLANS="${PLANS:-$HERE/plans}"                      # <table>.txt plan DSL files
OUT_DIR="${OUT_DIR:-$REPO/build/compression-bench}" # .hpln / .parquet / .csv outputs
SIMPATICO="${SIMPATICO:-}"                         # path to the simpatico CLI; built if unset
BUILD_DIR="${BUILD_DIR:-$REPO/src/compression/simpatico_codegen/build}"
WARMUP="${WARMUP:-3}"                              # benchmark: untimed iterations per column
ITERS="${ITERS:-10}"                               # benchmark: timed iterations per column
MODE="${MODE:-per-column}"                         # benchmark: per-column | full-table
THREADS="${THREADS:-0}"                            # benchmark full-table: worker threads (0 = one per column)
VERIFY="${VERIFY:-1}"                              # 1: round-trip check in compress/decompress
KEEP_PARQUET="${KEEP_PARQUET:-0}"                  # 1: decompress also writes a Parquet copy
export SIMPATICO_JIT_STATS="${SIMPATICO_JIT_STATS:-1}"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  cat <<USAGE

Flags:
  --table NAME     lineitem | orders (any table with a $PLANS/NAME.txt). Default: lineitem
  --part N         part file index under \$DATA/NAME/part.N.parquet. Default: 0
  --input PATH     explicit Parquet file; the table is inferred from its parent directory
  --plan PATH      explicit plan file (default: \$PLANS/<table>.txt)
  --hpln PATH      .hpln to write (compress) / read (decompress). Default: \$OUT_DIR/<table>.part<N>.hpln
  -h, --help

Environment knobs (current values):
  DATA=$DATA
  PLANS=$PLANS
  OUT_DIR=$OUT_DIR
  SIMPATICO=${SIMPATICO:-<build $BUILD_DIR>}
  WARMUP=$WARMUP ITERS=$ITERS MODE=$MODE THREADS=$THREADS VERIFY=$VERIFY KEEP_PARQUET=$KEEP_PARQUET
USAGE
}

[ $# -ge 1 ] || { usage; exit 1; }
STAGE="$1"; shift
case "$STAGE" in
  compress|decompress|benchmark|all) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "ERROR: unknown stage '$STAGE' (compress|decompress|benchmark|all)"; usage; exit 1 ;;
esac

TABLE="" PART=0 INPUT="" PLAN="" HPLN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --table) TABLE="$2"; shift 2 ;;
    --part)  PART="$2";  shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --plan)  PLAN="$2";  shift 2 ;;
    --hpln)  HPLN="$2";  shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown flag '$1'"; usage; exit 1 ;;
  esac
done

# ── Resolve input / table / plan ──────────────────────────────────────────────
if [ -n "$INPUT" ]; then
  [ -n "$TABLE" ] || TABLE="$(basename "$(dirname "$INPUT")")"
  PART_TAG="$(basename "$INPUT" .parquet)"
else
  TABLE="${TABLE:-lineitem}"
  INPUT="$DATA/$TABLE/part.$PART.parquet"
  PART_TAG="part.$PART"
fi
PLAN="${PLAN:-$PLANS/$TABLE.txt}"
HPLN="${HPLN:-$OUT_DIR/$TABLE.$PART_TAG.hpln}"

[ -f "$INPUT" ] || { echo "ERROR: no input Parquet at $INPUT (set DATA= or --input)"; exit 1; }
[ -f "$PLAN" ]  || { echo "ERROR: no plan file at $PLAN (tables with plans: $(ls "$PLANS" | sed 's/\.txt$//' | tr '\n' ' '))"; exit 1; }
mkdir -p "$OUT_DIR"

# ── Locate or build the CLI ───────────────────────────────────────────────────
if [ -z "$SIMPATICO" ]; then
  SIMPATICO="$BUILD_DIR/simpatico"
  if [ ! -x "$SIMPATICO" ]; then
    echo "building simpatico CLI into $BUILD_DIR (one-time, several minutes)"
    cmake -S "$REPO/src/compression/simpatico_codegen" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --parallel --target simpatico_cli
  fi
fi
[ -x "$SIMPATICO" ] || { echo "ERROR: simpatico CLI not found at $SIMPATICO"; exit 1; }

# The CLI matches plan blocks to Parquet columns positionally and refuses a count mismatch;
# fail early with the plan-side count so a wrong --plan is obvious before the 3 GB read.
PLAN_BLOCKS=$(( $(grep -c '^---' "$PLAN") + 1 ))

# Shared-box courtesy: other processes on the GPU skew every throughput number.
if command -v nvidia-smi >/dev/null 2>&1; then
  OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "$OTHER" -eq 0 ] || echo "WARNING: $OTHER other compute process(es) on the GPU; timings will be noisy"
fi

echo "stage     : $STAGE"
echo "table     : $TABLE ($PART_TAG)"
echo "input     : $INPUT ($(du -h "$INPUT" | cut -f1))"
echo "plan      : $PLAN ($PLAN_BLOCKS column blocks)"
echo "hpln      : $HPLN"
echo "simpatico : $SIMPATICO"
echo "jit cache : ${SIMPATICO_JIT_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/simpatico/jit}"
echo

T_ALL=$(date +%s.%N)
run_stage() {  # run_stage <label> <cmd...>: echo the command, run it, print its wall time
  local label="$1"; shift
  echo "==> $label"
  printf '   '; printf ' %q' "$@"; echo
  local t0; t0=$(date +%s.%N)
  "$@"
  local rc=$?
  printf '<== %s: %.1f s wall (exit %d)\n\n' "$label" "$(echo "$(date +%s.%N) - $t0" | bc)" "$rc"
  return $rc
}

VERIFY_FLAG=(); [ "$VERIFY" = 1 ] && VERIFY_FLAG=(--verify)

if [ "$STAGE" = compress ] || [ "$STAGE" = all ]; then
  run_stage "compress" "$SIMPATICO" compress --input "$INPUT" --plan "$PLAN" --out "$HPLN" "${VERIFY_FLAG[@]}"
  ls -l "$HPLN" | awk '{print "    hpln bytes:", $5}'; echo
fi

if [ "$STAGE" = decompress ] || [ "$STAGE" = all ]; then
  [ -f "$HPLN" ] || { echo "ERROR: no .hpln at $HPLN — run the compress stage first"; exit 1; }
  ARGS=(decompress --input "$HPLN")
  [ "$KEEP_PARQUET" = 1 ] && ARGS+=(--out "${HPLN%.hpln}.decompressed.parquet")
  [ "$VERIFY" = 1 ] && ARGS+=(--verify "$INPUT")
  run_stage "decompress" "$SIMPATICO" "${ARGS[@]}"
fi

if [ "$STAGE" = benchmark ] || [ "$STAGE" = all ]; then
  CSV="$OUT_DIR/$TABLE.$PART_TAG.$MODE.csv"
  ARGS=(benchmark --input "$INPUT" --plan "$PLAN" --mode "$MODE" --warmup "$WARMUP" --iters "$ITERS" --csv-out "$CSV")
  [ "$THREADS" -gt 0 ] && ARGS+=(--threads "$THREADS")
  run_stage "benchmark" "$SIMPATICO" "${ARGS[@]}"
  echo "    csv: $CSV"; echo
fi

printf 'total wall: %.1f s\n' "$(echo "$(date +%s.%N) - $T_ALL" | bc)"
