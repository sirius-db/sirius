#!/usr/bin/env bash
# Timings across iterations for all 22 TPC-H queries, every result checked against DuckDB.
# Each iteration starts a fresh cluster. Writes runtimes.csv and summary.md under
# $RUN_ROOT_BASE/bench/<timestamp>_<label>/.
#   bench.sh                                   SF1000, 3 iterations
#   bench.sh --sf 3000 --iterations 1          SF3000 (set HOST_BYTES=$((256 << 30)))
#   PIN_COMPRESSION=1 bench.sh --pin --label pinned
#                                              lineitem + orders pinned in host memory on every CN
# Extra arguments go to the tools repo's scripts/bench.sh (--queries, --label, --fresh-per-query, ...).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

args=("$@")
[[ " $* " == *" --queries "* ]] || args+=(--queries "$ALL22")
[[ " $* " == *" --sf "* ]] || args+=(--sf 1000)
[[ " $* " == *" --iterations "* ]] || args+=(--iterations 3)
exec "$TOOLS_REPO/scripts/bench.sh" "${args[@]}"
