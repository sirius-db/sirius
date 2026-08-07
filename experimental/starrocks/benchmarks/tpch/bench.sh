#!/usr/bin/env bash
# TPC-H timing sweep against whatever FE answers on $FE_PORT.
#
# Usage: bench.sh <out_csv> [runs] [q01 q02 ...]
#   out_csv   where per-run timings land (CSV: query,run,status,ms,rows)
#   runs      timed repetitions per query after 1 discarded warm-up (default 3)
#   qNN...    subset of queries; default all 22
#
# Environment:
#   TPCH_DATA          directory holding <table>/*.parquet (substituted into the
#                      queries' FILES() paths; required)
#   FE_PORT            FE MySQL port (default 9030)
#   QUERY_TIMEOUT      per-run client timeout in seconds (default 60; SF1 passes
#                      finish in well under 2s, so anything near this is a hang)
#   RESTART_CMD        command that fully restarts the cluster. The CN has no
#                      cancel_plan_fragment yet, so a hung or failed query strands
#                      its fragments and eventually starves the CNs ("No available
#                      backends") -- without a restart every later measurement is
#                      invalid. Leave empty only for engines that clean up after
#                      themselves (e.g. stock StarRocks BEs).
#
# A refusal (ERROR on the first output line) is recorded once and not retried;
# a timeout/empty result is recorded as a wedge. Both trigger $RESTART_CMD.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_CSV=${1:?usage: bench.sh <out_csv> [runs] [q01 q02 ...]}
RUNS=${2:-3}
shift $(( $# >= 2 ? 2 : 1 ))
QUERIES=("$@")
[ ${#QUERIES[@]} -eq 0 ] && QUERIES=($(cd "$HERE/queries" && ls q*.sql | sed 's/\.sql$//'))

TPCH_DATA=${TPCH_DATA:?set TPCH_DATA to the directory holding <table>/*.parquet}
FE_PORT=${FE_PORT:-9030}
QUERY_TIMEOUT=${QUERY_TIMEOUT:-60}
RESTART_CMD=${RESTART_CMD:-}
MYSQL="mysql --host 127.0.0.1 --port $FE_PORT --user root --batch --connect-timeout=5"
OUT=$(dirname "$OUT_CSV")
mkdir -p "$OUT"

wait_alive() {
  for _ in $(seq 1 150); do
    n=$($MYSQL -N -e "SHOW COMPUTE NODES;" 2>/dev/null | grep -c true)
    b=$($MYSQL -N -e "SHOW BACKENDS;" 2>/dev/null | grep -c true)
    [ $((${n:-0} + ${b:-0})) -ge 1 ] && return 0
    sleep 2
  done
  return 1
}

restart_cluster() {
  [ -z "$RESTART_CMD" ] && return 0
  echo "restarting cluster..."
  eval "$RESTART_CMD"
  wait_alive
}

wait_alive || { echo "no cluster on port $FE_PORT"; exit 1; }
echo "query,run,status,ms,rows" > "$OUT_CSV"

for q in "${QUERIES[@]}"; do
  Q=$(sed "s|__TPCH_DATA__|$TPCH_DATA|g" "$HERE/queries/$q.sql")
  for r in $(seq 0 "$RUNS"); do   # run 0 = warm-up, discarded
    f=$OUT/$q.r$r.out
    t0=$(date +%s%3N)
    timeout "$QUERY_TIMEOUT" $MYSQL -e "${Q}" > "$f" 2>&1
    rc=$?
    t1=$(date +%s%3N)
    ms=$((t1 - t0))
    if [ $rc -eq 0 ] && [ -s "$f" ] && ! head -1 "$f" | grep -q ERROR; then
      rows=$(($(wc -l < "$f") - 1))
      [ "$r" -gt 0 ] && echo "$q,$r,pass,$ms,$rows" >> "$OUT_CSV"
      echo "$q r$r pass ${ms}ms rows=$rows"
    elif head -1 "$f" 2>/dev/null | grep -q ERROR; then
      echo "$q,$r,refused,$ms,0" >> "$OUT_CSV"
      echo "$q r$r REFUSED: $(head -c 160 "$f" | tr '\n' ' ')"
      restart_cluster || { echo "cluster did not recover"; exit 1; }
      break
    else
      echo "$q,$r,wedge,$ms,0" >> "$OUT_CSV"
      echo "$q r$r WEDGE/TIMEOUT (rc=$rc)"
      restart_cluster || { echo "cluster did not recover"; exit 1; }
      break
    fi
  done
done
echo "== bench complete: $OUT_CSV =="
