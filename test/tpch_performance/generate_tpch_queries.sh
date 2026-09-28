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
# Seeding follows TPC-H spec clause 2.1.3.3: stream n is generated with
# qgen -r <seed0 + n>, so the power stream (0) runs on seed0 and every
# throughput stream gets its own substitution parameters. qgen's -p only
# permutes the query order — it never varies the parameters — and without -r
# qgen falls back to time(NULL), whose one-second granularity hands every
# stream generated in the same second identical parameters.
#
# --seed <seed0> sets seed0 explicitly; an official run passes the end of the
# database load formatted mmddhhmmss (clause 2.1.3.3) and discloses it.
# Without --seed, seed0 defaults to the current time in that format: the
# streams are still spec-shaped (distinct, consecutive seeds), but seed0 is
# not tied to the load end, so pass --seed for an official-style run.

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

if [ -n "$SEED" ] && ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --seed must be a non-negative integer (mmddhhmmss for an official run), got '$SEED'"
    exit 1
fi
if [ -z "$SEED" ]; then
    RAW_SEED="$(date +%m%d%H%M%S)"
    SEED="$((10#$RAW_SEED))"
    echo "No --seed given; seed0=$SEED (current time $RAW_SEED, mmddhhmmss)."
    echo "An official run passes --seed <load-end timestamp mmddhhmmss> (TPC-H clause 2.1.3.3)."
else
    SEED="$((10#$SEED))"
fi

DBGEN_DIR="${DBGEN_DIR:-$PROJECT_DIR/test_datasets/tpch-dbgen}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/test_datasets/tpch_queries_sf${SF}}"

ensure_tpch_tools "$DBGEN_DIR" "$PROJECT_DIR" qgen

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "Generating query streams 0..$NUM_STREAMS at SF$SF -> $OUTPUT_DIR"
for ((n = 0; n <= NUM_STREAMS; n++)); do
    # Clause 2.1.3.3: stream n runs on seed0 + n. Never rely on qgen's no-seed
    # fallback — time(NULL) collides across streams generated in the same second.
    args=(-c -s "$SF" -p "$n" -r "$((SEED + n))")
    # qgen reads dists.dss from the cwd and the query templates from DSS_QUERY.
    # The templates in the dbgen root are the corrected ones; queries/ holds the
    # older variants (view-based q15, ANSI "day (3)" in q1).
    (cd "$DBGEN_DIR" && DSS_QUERY=. ./qgen "${args[@]}") \
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
