#!/usr/bin/env bash
# Full TPC-H sweep at SF100, spill compression off vs on.
#
# Standalone (performance_test.py needs passwordless sudo for drop_os_cache).
# Runs all 22 queries in ONE DuckDB session per arm, so the plan register and
# per-edge spill state persist across queries the way they would in a real
# workload. Per-query times come from the CLI's own timer.
set -uo pipefail

SF_DIR=${SF_DIR:-test_datasets/tpch_parquet_sf100}
ITERS=${ITERS:-3}
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
    """All per-iteration durations per query, in order."""
    txt = pathlib.Path(f).read_text(errors="ignore")
    ev = re.findall(r"[|\u2502]\s*(QMARK|QDONE)\s*[|\u2502]\s*(\d+)\s*[|\u2502]\s*(\d+)\s*[|\u2502]", txt)
    start, res = {}, {}
    for kind, q, t in ev:
        q, t = int(q), int(t)
        if kind == "QMARK":
            start[q] = t
        elif q in start:
            res.setdefault(q, []).append((t - start[q]) / 1000.0)
            del start[q]
    return res

a, b = times(out / "off.out"), times(out / "on.out")

# Steady state: fastest iteration, so page-cache warmth is equal across arms.
# First iteration is reported alongside because that is where an edge pays its
# one-off plan setup — the cost the bitpack default exists to remove.
print(f"{'q':>3} {'off_min':>8} {'on_min':>8} {'ratio':>7}   {'off_1st':>8} {'on_1st':>8} {'ratio':>7}")
tam = tbm = ta1 = tb1 = 0.0
for q in sorted(set(a) | set(b)):
    xs, ys = a.get(q, []), b.get(q, [])
    xm, ym = (min(xs) if xs else None), (min(ys) if ys else None)
    x1, y1 = (xs[0] if xs else None), (ys[0] if ys else None)
    for v, acc in ((xm, 'tam'), (ym, 'tbm'), (x1, 'ta1'), (y1, 'tb1')):
        pass
    if xm: tam += xm
    if ym: tbm += ym
    if x1: ta1 += x1
    if y1: tb1 += y1
    rm = f"{ym/xm:.2f}x" if xm and ym else "-"
    r1 = f"{y1/x1:.2f}x" if x1 and y1 else "-"
    f = lambda v: f"{v:.2f}" if v else "-"
    print(f"{q:>3} {f(xm):>8} {f(ym):>8} {rm:>7}   {f(x1):>8} {f(y1):>8} {r1:>7}")
print(f"{'sum':>3} {tam:>8.2f} {tbm:>8.2f} {(tbm/tam if tam else 0):>6.2f}x   "
      f"{ta1:>8.2f} {tb1:>8.2f} {(tb1/ta1 if ta1 else 0):>6.2f}x")

# Spread across iterations tells us whether the min is stable or noisy.
import statistics
for name, d in (("off", a), ("on", b)):
    sp = [max(v) / min(v) for v in d.values() if len(v) > 1 and min(v) > 0]
    if sp:
        print(f"  {name}: per-query max/min spread  median {statistics.median(sp):.2f}x  "
              f"worst {max(sp):.2f}x")
PY
