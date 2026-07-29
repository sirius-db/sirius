#!/usr/bin/env bash
# Full TPC-H sweep at SF100, spill compression off vs on.
#
# Standalone (performance_test.py needs passwordless sudo for drop_os_cache).
# Runs all 22 queries in ONE DuckDB session per arm, so the plan register and
# per-edge spill state persist across queries the way they would in a real
# workload. Per-query times come from the CLI's own timer.
set -uo pipefail

SF_DIR=${SF_DIR:-test_datasets/tpch_parquet_sf100}
ITERS=${ITERS:-1}
QUERIES=${QUERIES:-$(seq 1 22)}
OUT=./spill_sweep
mkdir -p "$OUT"

views() {
  for t in lineitem orders customer part partsupp supplier nation region; do
    echo "CREATE VIEW ${t} AS SELECT * FROM read_parquet('${SF_DIR}/${t}/*.parquet');"
  done
}

for ARM in off on; do
  LOGDIR="$OUT/${ARM}_logs"; rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
  SQL="SET sirius_log_backend='spdlog'; SET sirius_log_dir='${LOGDIR}'; SET sirius_log_level='debug';
$(views)"
  for q in $QUERIES; do
    QSQL=$(tr '\n' ' ' < "test/tpch_performance/tpch_queries/orig/q${q}.sql" | sed "s/'/''/g")
    for _ in $(seq 1 "$ITERS"); do
      SQL="${SQL}
select 'QMARK' as m, ${q} as q, epoch_ms(current_timestamp) as t;
FROM gpu_execution('${QSQL}');
select 'QDONE' as m, ${q} as q, epoch_ms(current_timestamp) as t;"
    done
  done

  echo "=== arm=${ARM}"
  START=$(date +%s.%N)
  SIRIUS_CONFIG_FILE="$(pwd)/test/tpch_performance/spill_compression_${ARM}.yaml" \
    timeout 5400 ./build/release/duckdb -c "$SQL" > "$OUT/${ARM}.out" 2>&1
  RC=$?
  END=$(date +%s.%N)
  echo "  exit=${RC} total_wall=$(echo "$END - $START" | bc)s"

  S=$(ls "$LOGDIR"/sirius_*.log 2>/dev/null | head -1)
  if [[ -n "${S}" ]]; then
    printf "  %-26s %s\n" "seeded-from-lineage" "$(grep -c 'seeded' "$S")"
    printf "  %-26s %s\n" "explored"            "$(grep -c 'explored spill plans' "$S")"
    printf "  %-26s %s\n" "compressed spills"   "$(grep -c 'compressed host' "$S")"
    printf "  %-26s %s\n" "declined (fallback)" "$(grep -c 'compressed spill declined' "$S")"
    printf "  %-26s %s\n" "OOM declines"        "$(grep -c 'declined (std::bad_alloc' "$S")"
  fi
done

# Per-query times: the CLI prints "Run Time (s): real N" after each statement;
# the marker select before each query lets us attribute them.
echo
echo "=== per-query real seconds (off / on)"
python3 - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
def times(f):
    txt = pathlib.Path(f).read_text(errors="ignore")
    # Rows print as | QMARK | 3 | 1690000000000 |
    ev = re.findall(r"[|│]\s*(QMARK|QDONE)\s*[|│]\s*(\d+)\s*[|│]\s*(\d+)\s*[|│]", txt)
    start, res = {}, {}
    for kind, q, t in ev:
        q, t = int(q), int(t)
        if kind == "QMARK": start[q] = t
        elif q in start:    res[q] = (t - start[q]) / 1000.0
    return res
a, b = times(out/"off.out"), times(out/"on.out")
print(f"{'q':>3} {'off':>8} {'on':>8} {'ratio':>7}")
ta = tb = 0.0
for q in sorted(set(a) | set(b)):
    x, y = a.get(q), b.get(q)
    if x: ta += x
    if y: tb += y
    r = f"{y/x:.2f}x" if x and y and x > 0 else "-"
    xs = f"{x:.2f}" if x else "-"
    ys = f"{y:.2f}" if y else "-"
    print(f"{q:>3} {xs:>8} {ys:>8} {r:>7}")
print(f"{'sum':>3} {ta:>8.2f} {tb:>8.2f} {(tb/ta if ta else 0):>6.2f}x")
PY
