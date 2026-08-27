#!/usr/bin/env bash
# TPC-H timing sweep against whatever FE answers on $FE_PORT.
#
# Usage: bench.sh [--cold|--cold-restart] <out_csv> [runs] [q01 q02 ...]
#   out_csv   where per-run timings land (CSV: query,run,phase,status,ms,rows)
#   runs      timed repetitions per query after the warm-up (default 3)
#   qNN...    subset of queries; default all 22
#
# Cold-start mode (--cold, or COLD=1; the flag is optional so existing positional
# callers keep working unchanged):
#   Run 0 -- the first execution of a query on the current cluster -- is discarded as
#   a warm-up by default, and it is the only run that exercises first contact (lazy
#   nixl session setup, plan-cache misses, first-touch allocation). --cold RECORDS it
#   tagged phase=cold; runs 1..N stay phase=warm, which keeps the cold outlier out of
#   the warm medians while both land in one file.
#   Cold runs are cut at $COLD_TIMEOUT (default 180 s) rather than $QUERY_TIMEOUT: a
#   first-contact stall can run past a minute, and a 30 s cut records it as a generic
#   wedge carrying no latency information.
#   A failing COLD run does NOT trigger $RESTART_CMD -- a restart would only make the
#   next run cold again -- so the warm runs continue on the same cluster and a
#   cold-only failure is visible as such. A failing WARM run keeps the old behaviour
#   (record, restart, skip the rest of that query).
#
# --cold-restart implies --cold and additionally runs $RESTART_CMD before every
#   query, so every run 0 is a true cold-cluster first contact instead of only the
#   sweep's first query. Requires $RESTART_CMD.
#
# Environment:
#   TPCH_DATA          directory holding <table>/*.parquet (substituted into the
#                      queries' FILES() paths; required)
#   FE_PORT            FE MySQL port (default 9030)
#   QUERY_TIMEOUT      per-run client timeout in seconds for warm runs (default 30;
#                      SF1 passes finish in well under 2s, so anything near this is
#                      a hang)
#   COLD_TIMEOUT       per-run client timeout for run 0 (default 180)
#   COLD               set to 1 for --cold without touching the argv
#   MIN_BACKENDS       alive nodes wait_alive requires before proceeding (default 2).
#                      Treated as the EXPECTED size of the cluster, not a floor: if
#                      more nodes than this are alive the sweep aborts, because a
#                      threshold below the real topology can be satisfied while nodes
#                      are still joining and the sweep then measures a half-booted
#                      cluster. Set it to the real node count.
#   ALLOW_EXTRA_BACKENDS  set to 1 to downgrade that abort to a warning
#   RESTART_CMD        command that fully restarts the cluster. The CN has no
#                      cancel_plan_fragment yet, so a hung or failed query strands
#                      its fragments and eventually starves the CNs ("No available
#                      backends") -- without a restart every later measurement is
#                      invalid. Leave empty only for engines that clean up after
#                      themselves (e.g. stock StarRocks BEs).
#
# A refusal (ERROR on the first output line) is recorded once and not retried;
# a timeout/empty result is recorded as a wedge.
#
# NOTE: this script times and counts rows only -- it does not check answers. The
# row counts it records are what analyze.py compares across engines; a query that
# returns the wrong number of rows is caught there, not here.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

COLD=${COLD:-0}
COLD_RESTART=${COLD_RESTART:-0}
while [ $# -gt 0 ]; do
  case $1 in
    --cold)         COLD=1; shift ;;
    --cold-restart) COLD=1; COLD_RESTART=1; shift ;;
    --)             shift; break ;;
    -*)             echo "unknown option: $1" >&2; exit 2 ;;
    *)              break ;;
  esac
done

OUT_CSV=${1:?usage: bench.sh [--cold|--cold-restart] <out_csv> [runs] [q01 q02 ...]}
RUNS=${2:-3}
shift $(( $# >= 2 ? 2 : 1 ))
QUERIES=("$@")
[ ${#QUERIES[@]} -eq 0 ] && QUERIES=($(cd "$HERE/queries" && ls q*.sql | sed 's/\.sql$//'))

TPCH_DATA=${TPCH_DATA:?set TPCH_DATA to the directory holding <table>/*.parquet}
FE_PORT=${FE_PORT:-9030}
QUERY_TIMEOUT=${QUERY_TIMEOUT:-30}
COLD_TIMEOUT=${COLD_TIMEOUT:-180}
RESTART_CMD=${RESTART_CMD:-}
MIN_BACKENDS=${MIN_BACKENDS:-2}
ALLOW_EXTRA_BACKENDS=${ALLOW_EXTRA_BACKENDS:-0}
MYSQL="mysql --host 127.0.0.1 --port $FE_PORT --user root --batch --connect-timeout=5"
OUT=$(dirname "$OUT_CSV")
mkdir -p "$OUT"

if [ "$COLD_RESTART" = 1 ] && [ -z "$RESTART_CMD" ]; then
  echo "--cold-restart needs RESTART_CMD to make each run 0 cold" >&2
  exit 2
fi

# Number of rows whose Alive column is exactly "true".
#
# The old form -- `SHOW ... | grep -c true` -- matched the whole row, so any other
# true-valued column (HasStoragePath, SystemDecommissioned) counted a node as alive
# and $MIN_BACKENDS could be satisfied by a cluster that had not finished booting.
#
# Alive is column 9 of both statements: SHOW emits the ProcDir title lists verbatim
# (ShowExecutor.visitShowComputeNodes / .visitShowBackendsStatement -> ...ProcDir
# .getMetadata()), and the first nine titles of ComputeNodeProcDir.TITLE_NAMES and
# of BackendsProcDir.TITLE_NAMES are identical: <Id>, IP, HeartbeatPort, BePort,
# HttpPort, BrpcPort, LastStartTime, LastHeartbeat, Alive. (The shared-data variants
# only append columns, so the index is stable there too.) The index is resolved from
# the header row anyway, so a future column insertion cannot silently re-break this
# the way the grep did; an unrecognised header yields 0 and the gate fails closed.
alive_count() {
  $MYSQL -e "$1" 2>/dev/null | awk -F'\t' '
    NR == 1 { for (i = 1; i <= NF; i++) if ($i == "Alive") c = i; next }
    c && $c == "true" { n++ }
    END { print n + 0 }'
}

# Waits for >= $MIN_BACKENDS alive nodes AND for that count to hold across two
# consecutive polls -- a single poll crossing the threshold can be a cluster that is
# still adding nodes. Publishes the settled counts in $ALIVE_CN/$ALIVE_BE/$ALIVE_TOTAL.
ALIVE_CN=0; ALIVE_BE=0; ALIVE_TOTAL=0
wait_alive() {
  local n=0 b=0 total=0 prev=-1
  for _ in $(seq 1 150); do
    n=$(alive_count "SHOW COMPUTE NODES;")
    b=$(alive_count "SHOW BACKENDS;")
    total=$((n + b))
    ALIVE_CN=$n; ALIVE_BE=$b; ALIVE_TOTAL=$total
    [ "$total" -ge "$MIN_BACKENDS" ] && [ "$total" -eq "$prev" ] && return 0
    prev=$total
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

wait_alive || {
  echo "no cluster on port $FE_PORT (alive: $ALIVE_CN CN + $ALIVE_BE BE, need $MIN_BACKENDS)"
  exit 1
}
if [ "$ALIVE_TOTAL" -gt "$MIN_BACKENDS" ]; then
  echo "MIN_BACKENDS=$MIN_BACKENDS but $ALIVE_TOTAL nodes are alive ($ALIVE_CN CN + $ALIVE_BE BE)." >&2
  echo "  MIN_BACKENDS is the gate that stops a sweep from starting against a partially" >&2
  echo "  booted cluster; set below the real topology it can be satisfied while nodes are" >&2
  echo "  still joining, and the sweep then measures the wrong cluster." >&2
  if [ "$ALLOW_EXTRA_BACKENDS" = 1 ]; then
    echo "  ALLOW_EXTRA_BACKENDS=1 -- continuing anyway." >&2
  else
    echo "  Re-run with MIN_BACKENDS=$ALIVE_TOTAL (or ALLOW_EXTRA_BACKENDS=1 to override)." >&2
    exit 1
  fi
fi
echo "cluster: $ALIVE_CN alive compute nodes + $ALIVE_BE alive backends (MIN_BACKENDS=$MIN_BACKENDS)"
[ "$COLD" = 1 ] && echo "cold mode: run 0 is recorded (phase=cold, timeout ${COLD_TIMEOUT}s)"
[ "$COLD_RESTART" = 1 ] && echo "cold-restart mode: RESTART_CMD runs before every query"

echo "query,run,phase,status,ms,rows" > "$OUT_CSV"

for q in "${QUERIES[@]}"; do
  Q=$(sed "s|__TPCH_DATA__|$TPCH_DATA|g" "$HERE/queries/$q.sql")
  if [ "$COLD_RESTART" = 1 ]; then
    restart_cluster || { echo "cluster did not recover"; exit 1; }
  fi
  for r in $(seq 0 "$RUNS"); do   # run 0 = first contact: discarded unless --cold
    if [ "$r" -eq 0 ]; then phase=cold; tmo=$COLD_TIMEOUT; else phase=warm; tmo=$QUERY_TIMEOUT; fi
    f=$OUT/$q.r$r.out
    t0=$(date +%s%3N)
    timeout "$tmo" $MYSQL -e "${Q}" > "$f" 2>&1
    rc=$?
    t1=$(date +%s%3N)
    ms=$((t1 - t0))
    if [ $rc -eq 0 ] && [ -s "$f" ] && ! head -1 "$f" | grep -q ERROR; then
      rows=$(($(wc -l < "$f") - 1))
      if [ "$r" -gt 0 ] || [ "$COLD" = 1 ]; then
        echo "$q,$r,$phase,pass,$ms,$rows" >> "$OUT_CSV"
      fi
      echo "$q r$r $phase pass ${ms}ms rows=$rows"
      continue
    fi
    if head -1 "$f" 2>/dev/null | grep -q ERROR; then
      echo "$q,$r,$phase,refused,$ms,0" >> "$OUT_CSV"
      echo "$q r$r $phase REFUSED: $(head -c 160 "$f" | tr '\n' ' ')"
    else
      echo "$q,$r,$phase,wedge,$ms,0" >> "$OUT_CSV"
      echo "$q r$r $phase WEDGE/TIMEOUT (rc=$rc, cut at ${tmo}s)"
    fi
    if [ "$COLD" = 1 ] && [ "$phase" = cold ]; then
      # The cold failure IS the datapoint; keep going on the same cluster so the warm
      # runs show whether it was first-contact-only. A restart here would just produce
      # another cold failure.
      echo "  (cold failure recorded; continuing to warm runs on the same cluster)"
      continue
    fi
    restart_cluster || { echo "cluster did not recover"; exit 1; }
    break
  done
done
echo "== bench complete: $OUT_CSV =="
