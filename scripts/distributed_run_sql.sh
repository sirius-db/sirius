#!/usr/bin/env bash
# Runs a SQL file on N ranks on this host, rank i on GPU i, and prints rank 0's output (the
# result). Every rank runs the whole file: define tables over every file of each table, the same
# on every rank. Each statement is one distributed query; each rank scans its own byte range of
# each table.
# One host only. Across hosts, start each rank on its own host with the same
# sirius_exchange_peers list and its own sirius_exchange_rank.
#   pixi run scripts/distributed_run_sql.sh <ranks> <file.sql> [base port=27000] [log dir=/tmp]
set -euo pipefail
ranks=$1
sql=$2
port=${3:-27000}
logs=${4:-/tmp}
here=$(cd "$(dirname "$0")" && pwd)
source "$here/launcher_env.sh"
build="$here/../sirius-duckdb/build/release"
gpus=$(nvidia-smi -L | wc -l)
if ((ranks > gpus)); then echo "$ranks ranks need $ranks GPUs; this host has $gpus" >&2; exit 2; fi

peers=""
for ((i = 0; i < ranks; i++)); do peers+="${peers:+,}127.0.0.1:$((port + i))"; done

# A crashed rank leaves its receive mirror in /dev/shm: remove those whose process is gone.
for f in /dev/shm/sirius-exchange-*; do
  [[ -e $f ]] || continue
  pid=${f#/dev/shm/sirius-exchange-}
  [[ -d /proc/${pid%%-*} ]] || rm -f "$f"
done

pids=()
# Interrupted: stop the ranks too (timeout passes TERM on to each).
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
for ((i = 0; i < ranks; i++)); do
  {
    echo "LOAD '$build/extension/sirius/sirius.duckdb_extension';"
    # A rank falling back to the CPU alone would return a wrong answer: fail instead.
    echo "SET enable_duckdb_fallback = false;"
    echo "SET sirius_exchange_peers = '$peers';"
    echo "SET sirius_exchange_rank = $i;"
    cat "$sql"
  } | CUDA_VISIBLE_DEVICES=$i timeout -s KILL "${EXCHANGE_TIMEOUT:-1800}" "$build/duckdb" -unsigned -bail \
    > "$logs/distributed_rank_$i.log" 2>&1 &
  pids+=($!)
done
status=0
for ((i = 0; i < ranks; i++)); do
  if ! wait "${pids[$i]}"; then status=1; echo "rank $i failed: $logs/distributed_rank_$i.log"; fi
done
# Drop NIXL's start-up log lines: its info lines, and the plugin directory it looks for although its
# UCX backend is built in.
grep -v -E '^I[0-9]{4} |nixl_plugin_manager.*Error accessing directory|absl::InitializeLog' \
  "$logs/distributed_rank_0.log"
exit $status
