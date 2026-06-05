#!/usr/bin/env bash
# =============================================================================
# TPC-H SF1000 multi-GPU benchmark driver for Super Sirius.
#
# Runs the 22 TPC-H queries across a set of (access-mode, GPU-count) scenarios
# and captures per-query timings, statuses, and full logs so the run is
# reproducible. Uses the statically-linked DuckDB CLI (build/release/duckdb)
# with the Sirius extension built in — the transparent `SET gpu_execution=true`
# path. Each query runs in its OWN CLI process so a hard failure (OOM / illegal
# memory access) in one query cannot poison the GPU context for the rest.
#
# Scenarios (default = all 7):
#   disk_1gpu disk_2gpu disk_4gpu   read from /scratch, no pinning
#   host_1gpu host_2gpu host_4gpu   pin referenced columns in HOST tier
#   gpu_4gpu                        pin referenced columns in GPU tier
#
# NOTE: this run depends on the TEMP(GPFS) patch in src/io/uring/uring_reactor.cpp
# (plain io_uring reads instead of fixed-buffer reads) so the Sirius datasource
# can read parquet directly from the GPFS-mounted /scratch. Rebuild after
# reverting that patch if you move the data to a local NVMe filesystem.
#
# Usage:
#   ./run_benchmarks.sh                       # all scenarios, all queries
#   QUERIES="1,6,14" ./run_benchmarks.sh      # subset of queries
#   SCENARIOS="host_4gpu gpu_4gpu" ./run_benchmarks.sh
#   ITERATIONS=3 DATA=/path/to/sf1000 ./run_benchmarks.sh
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# ---- knobs (override via env) ----------------------------------------------
DATA="${DATA:-/scratch/prestouser/tpch-rs-float/scale-1000}"
DUCKDB="${DUCKDB:-$REPO/build/release/duckdb}"
PYTHON="${PYTHON:-$REPO/.pixi/envs/default/bin/python}"
GEN="$HERE/gen_query_sql.py"
CONFIG_DIR="$HERE/configs"
ITERATIONS="${ITERATIONS:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HERE/results/$TS}"
RETRIES="${RETRIES:-1}"   # extra attempts on transient GPU-init failure
# Per-query wall-clock budget (pin population + all iterations). A query that
# exceeds this is SIGKILLed and recorded as 'timeout' so a deadlock (e.g. the
# OOM-during-multi-table-pin futex hang seen on 1 GPU) can't stall the run.
QUERY_TIMEOUT="${QUERY_TIMEOUT:-600}"

DEFAULT_SCENARIOS="disk_1gpu disk_2gpu disk_4gpu host_1gpu host_2gpu host_4gpu gpu_4gpu"
SCENARIOS="${SCENARIOS:-$DEFAULT_SCENARIOS}"

# scenario -> "config_yaml tier gpus"
scenario_spec() {
  case "$1" in
    disk_1gpu) echo "sirius_1gpu.yaml none 1" ;;
    disk_2gpu) echo "sirius_2gpu.yaml none 2" ;;
    disk_4gpu) echo "sirius_4gpu.yaml none 4" ;;
    host_1gpu) echo "sirius_1gpu.yaml host 1" ;;
    host_2gpu) echo "sirius_2gpu.yaml host 2" ;;
    host_4gpu) echo "sirius_4gpu.yaml host 4" ;;
    gpu_4gpu)  echo "sirius_4gpu.yaml gpu  4" ;;
    *) echo "" ;;
  esac
}

# ---- query list expansion ("1,6-8,14" -> "1 6 7 8 14") ---------------------
expand_queries() {
  local spec="${1:-1-22}" out=""
  IFS=',' read -ra parts <<< "$spec"
  for p in "${parts[@]}"; do
    if [[ "$p" == *-* ]]; then
      local a="${p%-*}" b="${p#*-}"
      for ((i=a; i<=b; i++)); do out+="$i "; done
    else
      out+="$p "
    fi
  done
  echo "$out"
}
QLIST="$(expand_queries "${QUERIES:-1-22}")"

# ---- preflight -------------------------------------------------------------
[[ -x "$DUCKDB" ]] || { echo "ERROR: duckdb CLI not found/executable: $DUCKDB" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "ERROR: python not found: $PYTHON" >&2; exit 1; }
[[ -d "$DATA"   ]] || { echo "ERROR: data dir not found: $DATA" >&2; exit 1; }
"$PYTHON" -c "import sys" || { echo "ERROR: python broken" >&2; exit 1; }

mkdir -p "$OUT"
CSV="$OUT/runtimes.csv"
echo "scenario,access,gpus,query,iteration,runtime_s,status" > "$CSV"
SUMMARY="$OUT/summary.txt"

# metadata
{
  echo "date:        $(date -Iseconds)"
  echo "commit:      $(cd "$REPO" && git rev-parse HEAD 2>/dev/null)"
  echo "branch:      $(cd "$REPO" && git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "data:        $DATA"
  echo "duckdb:      $DUCKDB"
  echo "iterations:  $ITERATIONS"
  echo "scenarios:   $SCENARIOS"
  echo "queries:     $QLIST"
  echo "reactor_patch: $(cd "$REPO" && git diff --stat -- src/io/uring/uring_reactor.cpp 2>/dev/null | tail -1)"
} | tee "$OUT/metadata.txt"

echo "=== output dir: $OUT ===" | tee "$SUMMARY"

# ---- run a single (scenario, query) in its own CLI process -----------------
# parses every "Run Time (s): real X" line -> iterations 0..N-1
run_one() {
  local scenario="$1" cfg="$2" tier="$3" gpus="$4" q="$5"
  local sdir="$OUT/$scenario"
  local qlog="$sdir/q${q}.log"
  local qsql="$sdir/q${q}.sql"
  mkdir -p "$sdir/log"

  "$PYTHON" "$GEN" --data "$DATA" --query "$q" --tier "$tier" --iterations "$ITERATIONS" > "$qsql" 2>"$sdir/q${q}.generr" || {
    echo "  q$q: SQL-GEN FAILED ($(tail -1 "$sdir/q${q}.generr"))"
    for ((it=0; it<ITERATIONS; it++)); do
      echo "$scenario,$tier,$gpus,$q,$it,,gen_error" >> "$CSV"; done
    return
  }

  local attempt=0 ok=0 timed_out=0 rc=0
  while (( attempt <= RETRIES )); do
    # SIGKILL on timeout: a futex-deadlocked CUDA process ignores SIGTERM.
    SIRIUS_CONFIG_FILE="$CONFIG_DIR/$cfg" SIRIUS_LOG_DIR="$sdir/log" \
      timeout -s KILL "${QUERY_TIMEOUT}s" \
      "$DUCKDB" -unsigned -init /dev/null < "$qsql" > "$qlog" 2>&1
    rc=$?
    if (( rc == 124 || rc == 137 )); then   # timeout: killed by SIGTERM(124)/SIGKILL(137)
      timed_out=1; ok=1
      echo "  q$q: TIMEOUT after ${QUERY_TIMEOUT}s — killed (likely OOM-deadlock)" | tee -a "$SUMMARY"
      break
    fi
    # retry only on transient GPU pool init OOM at extension load
    if grep -q 'Initialization function .* threw an exception.*out_of_memory\|cuda_async_view_memory_resource' "$qlog"; then
      attempt=$((attempt+1))
      echo "  q$q: transient GPU-init OOM, retry $attempt/$RETRIES"
      sleep 3
      continue
    fi
    ok=1; break
  done
  sleep 2   # let the GPU context fully release before the next process

  # status classification
  local status="ok"
  if (( timed_out )); then
    status="timeout"
  elif (( ! ok )); then
    status="init_oom"
  elif grep -qiE 'illegal memory access|rmm::bad_alloc|out of memory|out_of_memory|RMM failure|misaligned address|unspecified launch failure|invalid device pointer|thrust::system_error|CUDA error:|cudaError' "$qlog"; then
    status="cuda_error"
  elif grep -qiE 'Error in SiriusExecuteQuery|Error in SiriusGeneratePhysicalPlan|fallback to DuckDB' "$qlog"; then
    status="fallback"
  fi

  # extract per-iteration runtimes (k-th "Run Time" line == iter k)
  mapfile -t times < <(grep -oE 'Run Time \(s\): real [0-9.]+' "$qlog" | grep -oE '[0-9.]+$')
  local n=${#times[@]}
  for ((it=0; it<ITERATIONS; it++)); do
    if (( it < n )); then
      echo "$scenario,$tier,$gpus,$q,$it,${times[$it]},$status" >> "$CSV"
    else
      echo "$scenario,$tier,$gpus,$q,$it,,${status/ok/missing}" >> "$CSV"
    fi
  done
  printf "  q%-2s %-8s gpus=%s  times=[%s]  status=%s\n" \
    "$q" "$tier" "$gpus" "$(IFS=,; echo "${times[*]:-}")" "$status" | tee -a "$SUMMARY"
}

# ---- main loop -------------------------------------------------------------
START=$(date +%s)
for scenario in $SCENARIOS; do
  spec="$(scenario_spec "$scenario")"
  if [[ -z "$spec" ]]; then echo "WARN: unknown scenario '$scenario', skipping" | tee -a "$SUMMARY"; continue; fi
  read -r cfg tier gpus <<< "$spec"
  [[ -f "$CONFIG_DIR/$cfg" ]] || { echo "ERROR: missing config $CONFIG_DIR/$cfg" | tee -a "$SUMMARY"; continue; }
  echo "" | tee -a "$SUMMARY"
  echo "=== scenario: $scenario (config=$cfg tier=$tier gpus=$gpus) $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
  for q in $QLIST; do
    run_one "$scenario" "$cfg" "$tier" "$gpus" "$q"
  done
done
END=$(date +%s)

echo "" | tee -a "$SUMMARY"
echo "=== done in $((END-START))s. results: $CSV ===" | tee -a "$SUMMARY"

# ---- compact pivot (min runtime per scenario/query) ------------------------
"$PYTHON" - "$CSV" "$OUT/pivot_min.csv" <<'PY' 2>/dev/null || true
import csv, sys, collections
src, dst = sys.argv[1], sys.argv[2]
best = collections.defaultdict(dict)         # query -> scenario -> min time
scenarios = []
with open(src) as f:
    for r in csv.DictReader(f):
        if not r["runtime_s"]:
            continue
        t = float(r["runtime_s"]); sc = r["scenario"]; q = int(r["query"])
        if sc not in scenarios: scenarios.append(sc)
        cur = best[q].get(sc)
        best[q][sc] = t if cur is None else min(cur, t)
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["query"] + scenarios)
    for q in sorted(best):
        w.writerow([q] + [f"{best[q].get(sc,''):.4f}" if best[q].get(sc) is not None else "" for sc in scenarios])
print("wrote", dst)
PY
echo "pivot (min runtime per query/scenario): $OUT/pivot_min.csv"
