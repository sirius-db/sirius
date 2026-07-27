#!/usr/bin/env bash
# Generate per-stream TPC-H query sets with qgen, the reference substitution
# parameter generator.
#
# Produces, per stream n = 0..<num_streams>:
#   stream<n>.sql   the 22 queries in stream n's permutation, with stream n's
#                   own substitution parameters
#
# Stream 0 is the power run; streams 1..N are the throughput query streams, so
# generate at least <streams> streams.
#
# Usage:
#   ./test/tpch_performance/generate_tpch_queries.sh [--dbgen-dir <dir>] [--output-dir <dir>] [--seed <n>] <SF> <num_streams>
#
# Defaults:
#   --dbgen-dir   <project>/test_datasets/tpch-dbgen  (unzipped/built from tpch-dbgen.zip if missing)
#   --output-dir  <project>/test_datasets/tpch_queries_sf<SF>
#
# Without --seed, qgen draws a time-based seed per stream, which is what an
# official run wants. --seed <n> makes the draws reproducible: stream i uses
# seed n + i. One seed shared by every stream would give every stream the same
# parameters, because -p only permutes the query order.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=dbgen_bootstrap.sh
. "$SCRIPT_DIR/dbgen_bootstrap.sh"

DBGEN_DIR=""
OUTPUT_DIR=""
SEED=""
while [ $# -gt 0 ]; do
    case "${1:-}" in
        --dbgen-dir) DBGEN_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--dbgen-dir <dir>] [--output-dir <dir>] [--seed <n>] <SF> <num_streams>"
    echo "Example: $0 1 2   # SF1, streams 0..2 -> test_datasets/tpch_queries_sf1/"
    exit 1
fi

SF="$1"
NUM_STREAMS="$2"

DBGEN_DIR="${DBGEN_DIR:-$PROJECT_DIR/test_datasets/tpch-dbgen}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/test_datasets/tpch_queries_sf${SF}}"

ensure_tpch_tools "$DBGEN_DIR" "$PROJECT_DIR" qgen

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "Generating query streams 0..$NUM_STREAMS at SF$SF -> $OUTPUT_DIR"
for ((n = 0; n <= NUM_STREAMS; n++)); do
    args=(-c -s "$SF" -p "$n")
    if [ -n "$SEED" ]; then
        args+=(-r "$((SEED + n))")
    fi
    # qgen reads dists.dss from the cwd and the query templates from DSS_QUERY.
    (cd "$DBGEN_DIR" && DSS_QUERY=queries ./qgen "${args[@]}") \
        > "$OUTPUT_DIR/stream${n}.sql"
done

STATUS=0
for ((n = 0; n <= NUM_STREAMS; n++)); do
    f="$OUTPUT_DIR/stream${n}.sql"
    found=$(grep -coE '\(Q[0-9]+\)' "$f" || true)
    if [ "$found" -ne 22 ]; then
        echo "ERROR: $f holds $found queries, expected 22"
        STATUS=1
    fi
done
if [ "$STATUS" -ne 0 ]; then
    exit "$STATUS"
fi

echo ""
echo "Per-stream RNG seeds:"
for ((n = 0; n <= NUM_STREAMS; n++)); do
    printf '  stream %-3s %s\n' "$n" \
        "$(grep -m1 -oE 'using [0-9]+ as a seed' "$OUTPUT_DIR/stream${n}.sql")"
done
echo "Done."
