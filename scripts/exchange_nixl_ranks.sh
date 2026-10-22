#!/usr/bin/env bash
# Runs the exchange's one-rank-per-process NIXL test: N ranks on this host, rank i on GPU i.
#   scripts/exchange_nixl_ranks.sh [ranks=2] [base port=27000]
# Needs a build configured with -DSIRIUS_WITH_NIXL=ON; sources scripts/nixl_env.sh.
set -euo pipefail
ranks=${1:-2}
port=${2:-27000}
here=$(cd "$(dirname "$0")" && pwd)
source "$here/nixl_env.sh"
test_bin="$here/../build/release/extension/sirius/test/cpp/sirius_unittest"

peers=""
for ((i = 0; i < ranks; i++)); do peers+="${peers:+,}127.0.0.1:$((port + i))"; done

pids=()
for ((i = 0; i < ranks; i++)); do
  CUDA_VISIBLE_DEVICES=$i SIRIUS_EXCHANGE_RANK=$i SIRIUS_EXCHANGE_PEERS=$peers \
    timeout -s KILL 300 "$test_bin" "[nixl_multiprocess]" > "/tmp/exchange_nixl_rank_$i.log" 2>&1 &
  pids+=($!)
done
status=0
for ((i = 0; i < ranks; i++)); do
  if wait "${pids[$i]}"; then echo "rank $i: passed"; else status=1; echo "rank $i: FAILED (/tmp/exchange_nixl_rank_$i.log)"; fi
done
exit $status
