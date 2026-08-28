#!/usr/bin/env bash
# Full cluster restart for bench.sh's RESTART_CMD: kill everything, relaunch via
# up.sh, wait for NUM_CNS alive CNs, re-apply FE settings, and — for the pinned
# arm (PIN_AFTER_RESTART=1) — re-pin, since pins are in-process state and do not
# survive a CN restart.
#
# Env: NUM_CNS (default 2), LOG (cluster log path, default /tmp/pinned-cluster.log),
#      PIN_AFTER_RESTART=1 to re-run pin-all.sh after bring-up.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SR_DIR="$(cd "$HERE/../.." && pwd)"
NUM_CNS=${NUM_CNS:-2}
LOG=${LOG:-/tmp/pinned-cluster.log}

pkill -f '[s]irius-starrocks-cn' 2>/dev/null
pkill -f '[S]tarRocksFE' 2>/dev/null
for _ in $(seq 1 30); do
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break
  sleep 2
done
sleep 5
nohup "$HERE/up.sh" >> "$LOG" 2>&1 &

cd "$SR_DIR"
n=0
for _ in $(seq 1 120); do
  n=$(pixi run -e client bash -c "mysql -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES;'" 2>/dev/null \
      | awk -F'\t' '$9=="true"' | wc -l)
  [ "$n" -ge "$NUM_CNS" ] && break
  sleep 5
done
[ "$n" -ge "$NUM_CNS" ] || { echo "restart: only $n/$NUM_CNS CNs alive" >&2; exit 1; }

pixi run -e client bash -c "mysql -h127.0.0.1 -P9030 -uroot -e '
  ADMIN SET FRONTEND CONFIG (\"files_query_whole_file_ranges\" = \"true\");
  SET GLOBAL query_timeout = 1800;'"

if [ "${PIN_AFTER_RESTART:-0}" = "1" ]; then "$HERE/pin-all.sh"; fi
