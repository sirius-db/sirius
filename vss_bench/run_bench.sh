#!/usr/bin/env bash
# Run the VSS benchmark battery under GPU (gpu_execution=true) and CPU (false).
# Per mode: one discarded WARM-UP pass, then ITERS measured passes, then a per-query
# average (warm-up excluded) appended to results_<mode>.txt.
#
#   ./vss_bench/run_bench.sh [DB_FILE]
# Env:
#   DUCKDB  path to the Sirius-built duckdb binary (default: ../build/release/duckdb)
#   ITERS   measured iterations per mode (default: 10)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUCKDB="${DUCKDB:-$SCRIPT_DIR/../build/release/duckdb}"
DB="${1:-partsupp_laion.duckdb}"
ITERS="${ITERS:-10}"
BENCH="$SCRIPT_DIR/02_bench.sql"

[ -f "$DB" ]      || { echo "DB '$DB' not found; generate it with 01_create_partsupp_laion.sql first."; exit 1; }
[ -x "$DUCKDB" ]  || { echo "duckdb binary '$DUCKDB' not found/executable (set DUCKDB=...)."; exit 1; }
[ -f "$BENCH" ]   || { echo "bench file '$BENCH' not found."; exit 1; }

for MODE in gpu cpu; do
  if [ "$MODE" = gpu ]; then SETLINE="SET gpu_execution=true;"; else SETLINE="SET gpu_execution=false;"; fi

  TMP="$(mktemp)"
  {
    echo "$SETLINE"
    echo ".timer on"
    echo ".print __WARMUP_BEGIN__"      # one discarded pass to warm caches / GPU ctx / RMM pool
    cat "$BENCH"
    echo ".print __WARMUP_END__"
    for ((k=1; k<=ITERS; k++)); do      # measured passes
      echo ".print __ITER__ $k"
      cat "$BENCH"
    done
  } > "$TMP"

  echo "########## MODE=$MODE  ITERS=$ITERS (+1 warm-up) ##########"
  OUT="$SCRIPT_DIR/results_${MODE}.txt"
  "$DUCKDB" -unsigned "$DB" -f "$TMP" 2>&1 | tee "$OUT"
  rm -f "$TMP"

  # Per-query average over the measured passes (warm-up region ignored).
  awk -v MODE="$MODE" '
    /__WARMUP_BEGIN__/ { warm=1; next }
    /__WARMUP_END__/   { warm=0; next }
    /__QUERY__\|/ {
      split($0, a, "|"); cur=a[2];
      if (!(cur in seen)) { seen[cur]=1; order[++m]=cur; desc[cur]=a[3] }
      next
    }
    warm { next }
    /Run Time \(s\): real/ {
      for (i=1; i<=NF; i++) if ($i=="real") { t=$(i+1)+0; break }
      if (cur != "") {
        sum[cur]+=t; cnt[cur]++;
        if (mn[cur]=="" || t<mn[cur]) mn[cur]=t
      }
    }
    END {
      printf "\n===== SUMMARY  mode=%s  (warm-up excluded) =====\n", MODE
      printf "%-4s %-30s %5s %10s %10s\n", "qry", "description", "runs", "avg_s", "min_s"
      for (j=1; j<=m; j++) { q=order[j];
        if (cnt[q]>0) printf "%-4s %-30s %5d %10.4f %10.4f\n", q, desc[q], cnt[q], sum[q]/cnt[q], mn[q] }
    }
  ' "$OUT" | tee -a "$OUT"
done

echo
echo "Done. Full logs + per-query SUMMARY are in:"
echo "  $SCRIPT_DIR/results_gpu.txt"
echo "  $SCRIPT_DIR/results_cpu.txt"
