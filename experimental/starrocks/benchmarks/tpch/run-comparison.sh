#!/usr/bin/env bash
# End-to-end A-vs-B TPC-H comparison: sweeps engine A (Sirius GPU CNs), then engine B
# (stock StarRocks BEs), then produces the medians table + plot. One engine runs at a
# time -- they share the FE port and the host CPUs.
#
# Usage: TPCH_DATA=/path/to/tpch_sf1 ./run-comparison.sh [out_dir] [runs]
#
# Environment:
#   TPCH_DATA   directory holding <table>/*.parquet (required)
#   B_DIR       engine B layout dir (default ~/starrocks-bench; created by setup-engine-b.sh)
#   JAVA_HOME   JDK 17+ for engine B's FE (engine A brings its own via pixi)
#   SKIP_A / SKIP_B  set to 1 to reuse an existing timings.csv for that engine
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SR=$(cd "$HERE/../.." && pwd)             # experimental/starrocks
OUT=${1:-/tmp/tpch-comparison}
RUNS=${2:-3}
: "${TPCH_DATA:?set TPCH_DATA to the directory holding <table>/*.parquet}"
B=${B_DIR:-$HOME/starrocks-bench}
mkdir -p "$OUT"

alive_count() {  # rows whose Alive column is exactly "true" (see bench.sh for why)
  mysql --host 127.0.0.1 --port 9030 --user root --batch -e "$1" 2>/dev/null | awk -F'\t' '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "Alive") c = i; next }
    c && $c == "true" { n++ }
    END { print n + 0 }'
}

alive() {  # wait until N backends answer on 9030
  for _ in $(seq 1 150); do
    n=$(alive_count "SHOW COMPUTE NODES;")
    b=$(alive_count "SHOW BACKENDS;")
    [ $((n + b)) -ge "$1" ] && return 0
    sleep 2
  done
  echo "cluster did not come up" >&2
  return 1
}

engines_down() {
  pkill -f '[s]irius-starrocks-cn' 2>/dev/null || true
  pkill -f '[s]tarrocks_be' 2>/dev/null || true
  pkill -f '[S]tarRocksFE' 2>/dev/null || true
  sleep 5
}

if [ "${SKIP_A:-0}" != "1" ]; then
  echo "== engine A: Sirius GPU CNs =="
  engines_down
  (cd "$SR" && nohup pixi run cluster2 > "$OUT/cluster2.log" 2>&1 &)
  alive 2
  TPCH_DATA=$TPCH_DATA RESTART_CMD="pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'; sleep 5; (cd $SR && nohup pixi run cluster2 >> $OUT/cluster2.log 2>&1 &); sleep 20" \
    bash "$HERE/bench.sh" "$OUT/A/timings.csv" "$RUNS"
  engines_down
fi

if [ "${SKIP_B:-0}" != "1" ]; then
  echo "== engine B: stock StarRocks BEs =="
  [ -d "$B/fe" ] || "$HERE/setup-engine-b.sh"
  engines_down
  "$B/fe/bin/start_fe.sh" --daemon
  "$B/be1/bin/start_be.sh" --daemon
  "$B/be2/bin/start_be.sh" --daemon
  alive 1   # FE first
  mysql --host 127.0.0.1 --port 9030 --user root -e \
    'ALTER SYSTEM ADD BACKEND "127.0.0.1:9050"; ALTER SYSTEM ADD BACKEND "127.0.0.1:9052";' 2>/dev/null || true
  alive 2
  TPCH_DATA=$TPCH_DATA bash "$HERE/bench.sh" "$OUT/B/timings.csv" "$RUNS"
  engines_down
fi

echo "== analyze =="
python3 "$HERE/analyze.py" "$OUT/A/timings.csv" "$OUT/B/timings.csv" "$OUT/results.md" "$OUT/tpch_a_vs_b.png"
echo "table: $OUT/results.md"
echo "plot:  $OUT/tpch_a_vs_b.png"
