#!/usr/bin/env bash
# Runs the exchange's one-rank-per-process NIXL test: N ranks on this host, rank i on GPU i, or on
# the i-th GPU of GPUS (e.g. GPUS=0,2). TEST picks the Catch2 tag (default [nixl_multiprocess]).
#   scripts/distributed_test_nixl.sh [ranks=2] [base port=27000]
# Needs a build configured with -DSIRIUS_WITH_NIXL=ON; sources scripts/nixl_env.sh.
set -euo pipefail
ranks=${1:-2}
port=${2:-27000}
here=$(cd "$(dirname "$0")" && pwd)
source "$here/nixl_env.sh"
test_bin="$here/../build/release/test/cpp/sirius_unittest"

peers=""
for ((i = 0; i < ranks; i++)); do peers+="${peers:+,}127.0.0.1:$((port + i))"; done

IFS=, read -r -a gpus <<< "${GPUS:-$(seq -s, 0 $((ranks - 1)))}"
pids=()
for ((i = 0; i < ranks; i++)); do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} SIRIUS_EXCHANGE_RANK=$i SIRIUS_EXCHANGE_PEERS=$peers \
    timeout -s KILL 300 "$test_bin" "${TEST:-[nixl_multiprocess]}" > "/tmp/distributed_nixl_rank_$i.log" 2>&1 &
  pids+=($!)
done
status=0
for ((i = 0; i < ranks; i++)); do
  if wait "${pids[$i]}"; then echo "rank $i: passed"; else status=1; echo "rank $i: FAILED (/tmp/distributed_nixl_rank_$i.log)"; fi
done
exit $status
