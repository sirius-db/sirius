#!/usr/bin/env bash
# Head-to-head TPC-H benchmark: upstream main vs PR #2468 (EXISTS MIN/MAX rewrite) vs
# feat/semi-anti-build-left (RIGHT_SEMI/RIGHT_ANTI built on the kept side), each under
# delim_join_as_cte = true and false. See README.md in this directory.
#
# usage: bench/q21-compare/run.sh --sf <N> [--data <dir|s3://bucket/dir>] [options]
#   --sf N             TPC-H scale factor (required)
#   --data PATH        TPC-H parquet root (<PATH>/<table>/*.parquet), local or s3://. Without it,
#                      data is generated into <work>/data/tpch_sf<N>.
#   --work DIR         checkouts, builds, data and results (default: ./q21-compare-work)
#   --config YAML      Sirius config (default: sirius-small-gpu.yaml next to this script)
#   --pin MODE         gpu | host | none: where performance_test.py pins the tables (default none)
#   --gpu-limit BYTES  cap Sirius' GPU usage (memory.gpu.usage_limit_bytes), e.g. 32GB
#   --branches LIST    subset of "main pr2468 kept" (default: all three)
#   --skip-full        only run q21
#   --skip-build       reuse existing builds
#   --iterations N     iterations per query (default 3; the summary takes their median)
#   --jobs N           build parallelism (default: half the cores)
# environment:
#   EXTRA_PRE_SQL      SQL run before each query, ahead of the delim_join_as_cte setting
#   KEPT_COMMIT        commit of feat/semi-anti-build-left to measure (default: the fetched tip)
#   SIRIUS_PIN_TIER_<TABLE>=gpu|host  per-table pin tier, passed through to performance_test.py
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(git -C "$HERE" rev-parse --show-toplevel)

# name | fetch URL | ref | commit the published numbers were measured on
BRANCH_SPECS=(
  "main|https://github.com/sirius-db/sirius.git|main|e3935a660"
  "pr2468|https://github.com/sirius-db/sirius.git|refs/pull/2468/head|3a7c5712f"
  "kept|https://github.com/joosthooz/sirius.git|feat/semi-anti-build-left|${KEPT_COMMIT:-FETCH_HEAD}"
)

SF=""
DATA=""
WORK="$PWD/q21-compare-work"
CONFIG="$HERE/sirius-small-gpu.yaml"
PIN=none
GPU_LIMIT=""
BRANCHES="main pr2468 kept"
SKIP_FULL=0
SKIP_BUILD=0
ITERATIONS=3
JOBS=$(($(nproc) / 2))
while [[ $# -gt 0 ]]; do
  case $1 in
    --sf) SF=$2; shift 2 ;;
    --data) DATA=$2; shift 2 ;;
    --work) WORK=$2; shift 2 ;;
    --config) CONFIG=$2; shift 2 ;;
    --pin) PIN=$2; shift 2 ;;
    --gpu-limit) GPU_LIMIT=$2; shift 2 ;;
    --branches) BRANCHES=$2; shift 2 ;;
    --skip-full) SKIP_FULL=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --iterations) ITERATIONS=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n $SF ]] || { echo "--sf is required" >&2; exit 2; }
[[ $PIN =~ ^(gpu|host|none)$ ]] || { echo "--pin must be gpu, host or none" >&2; exit 2; }
mkdir -p "$WORK/src" "$WORK/results"
WORK=$(cd "$WORK" && pwd)

log() { echo "[$(date +%H:%M:%S)] $*"; }

checkout_and_build() {
  local name=$1 url=$2 ref=$3 commit=$4 dir=$WORK/src/$1
  if [[ ! -d $dir ]]; then
    log "$name: fetching $ref from $url"
    git -C "$SRC" fetch --no-tags "$url" "$ref"
    [[ $commit == FETCH_HEAD ]] && commit=$(git -C "$SRC" rev-parse FETCH_HEAD)
    git -C "$SRC" worktree add --detach "$dir" "$commit"
    git -C "$dir" submodule update --init --recursive
  fi
  log "$name: $(git -C "$dir" log --oneline -1)"
  if [[ $SKIP_BUILD == 0 || ! -f $dir/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension ]]; then
    log "$name: building with $JOBS jobs (log: $WORK/build_$name.log)"
    (cd "$dir" && CMAKE_BUILD_PARALLEL_LEVEL=$JOBS pixi run make) >"$WORK/build_$name.log" 2>&1
  fi
}

generate_data() {
  local out=$WORK/data/tpch_sf$SF
  if [[ -f $out/.done ]]; then DATA=$out; return; fi
  mkdir -p "$out"
  if command -v tpchgen-cli >/dev/null; then
    log "generating SF$SF with tpchgen-cli into $out"
    tpchgen-cli --scale-factor "$SF" --format parquet --parts "$((SF > 10 ? SF / 10 : 1))" \
      --output-dir "$out"
    # tpchgen-cli writes <table>/<table>.<part>.parquet with --parts, or <table>.parquet without.
    for f in "$out"/*.parquet; do
      [[ -e $f ]] || continue
      t=$(basename "$f" .parquet); mkdir -p "$out/$t"; mv "$f" "$out/$t/"
    done
  else
    log "generating SF$SF with DuckDB dbgen into $out (slow above SF100; prefer tpchgen-cli)"
    (cd "$WORK/src/main" && pixi run python - "$SF" "$out") <<'EOF'
import sys, duckdb
sf, out = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch;")
con.execute(f"CALL dbgen(sf={sf});")
for t in ["lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region"]:
    con.execute(f"COPY {t} TO '{out}/{t}' (FORMAT PARQUET, PER_THREAD_OUTPUT true);")
EOF
  fi
  touch "$out/.done"
  DATA=$out
}

effective_config() {
  local cfg=$WORK/config_effective.yaml
  if [[ -z $GPU_LIMIT ]]; then cp "$CONFIG" "$cfg"; echo "$cfg"; return; fi
  pixi run --manifest-path "$WORK/src/main/pixi.toml" python - "$CONFIG" "$cfg" "$GPU_LIMIT" <<'EOF'
import sys, yaml
src, dst, limit = sys.argv[1:]
cfg = yaml.safe_load(open(src)) or {}
mem = cfg.setdefault("sirius", {}).setdefault("memory", {}).setdefault("gpu", {})
mem.pop("usage_limit_fraction", None)
mem["usage_limit_bytes"] = limit
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
EOF
  echo "$cfg"
}

run_suite() {
  local name=$1 cte=$2 queries=$3 label=$4 dir=$WORK/src/$1 cfg=$5
  local tag="${name}_cte${cte}_${label}_pin${PIN}${GPU_LIMIT:+_gpu$GPU_LIMIT}"
  log "run $tag"
  export SIRIUS_PRE_SQL="${EXTRA_PRE_SQL:-} SET delim_join_as_cte = $cte;"
  (cd "$dir" && pixi run python test/tpch_performance/performance_test.py \
      --input "$DATA" --scale-factor "$SF" --iterations "$ITERATIONS" --engine gpu \
      --pin "$PIN" --queries "$queries" --config "$cfg" \
      --output "$WORK/results" --name "$tag") >"$WORK/results/$tag.log" 2>&1 ||
    log "  $tag exited nonzero (see $WORK/results/$tag.log)"
}

for spec in "${BRANCH_SPECS[@]}"; do
  IFS='|' read -r name url ref commit <<<"$spec"
  [[ " $BRANCHES " == *" $name "* ]] || continue
  checkout_and_build "$name" "$url" "$ref" "$commit"
done
[[ -n $DATA ]] || generate_data
CFG=$(effective_config)
log "data: $DATA, config: $CFG, pin: $PIN"

for name in $BRANCHES; do
  for cte in true false; do
    run_suite "$name" "$cte" 21 q21 "$CFG"
    [[ $SKIP_FULL == 1 ]] || run_suite "$name" "$cte" 1-22 full "$CFG"
  done
done

python3 "$HERE/summarize.py" "$WORK/results" | tee "$WORK/results/summary.md"
