#!/usr/bin/env bash
# Generate TPC-H refresh datasets (RF1 inserts / RF2 deletes) with the classic dbgen -U.
#
# Produces, per update set n = 1..<num_sets>:
#   orders.tbl.u<n>     RF1: new orders rows          (~0.1% of SF x 1.5M per set)
#   lineitem.tbl.u<n>   RF1: their lineitem rows
#   delete.<n>          RF2: o_orderkey values to delete
#
# tpch_power_throughput.py consumes one set for the power run and one set per
# throughput stream, so generate at least <streams> + 1 sets.
#
# Usage:
#   ./test/tpch_performance/generate_tpch_refresh.sh [--dbgen-dir <dir>] [--output-dir <dir>] <SF> <num_sets>
#
# Defaults:
#   --dbgen-dir   <project>/test_datasets/tpch-dbgen  (unzipped/built from tpch-dbgen.zip if missing)
#   --output-dir  <project>/test_datasets/tpch_refresh_sf<SF>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=dbgen_bootstrap.sh
. "$SCRIPT_DIR/dbgen_bootstrap.sh"

DBGEN_DIR=""
OUTPUT_DIR=""
while [ "${1:-}" = "--dbgen-dir" ] || [ "${1:-}" = "--output-dir" ]; do
    if [ "$1" = "--dbgen-dir" ]; then
        DBGEN_DIR="$2"
        shift 2
    else
        OUTPUT_DIR="$2"
        shift 2
    fi
done

if [ $# -ne 2 ]; then
    echo "Usage: $0 [--dbgen-dir <dir>] [--output-dir <dir>] <SF> <num_sets>"
    echo "Example: $0 1 5   # SF1, update sets 1..5 -> test_datasets/tpch_refresh_sf1/"
    exit 1
fi

SF="$1"
NUM_SETS="$2"

DBGEN_DIR="${DBGEN_DIR:-$PROJECT_DIR/test_datasets/tpch-dbgen}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/test_datasets/tpch_refresh_sf${SF}}"

ensure_tpch_tools "$DBGEN_DIR" "$PROJECT_DIR" dbgen

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "Generating $NUM_SETS TPC-H update set(s) at SF$SF -> $OUTPUT_DIR"
# dbgen resolves dists.dss relative to cwd; DSS_PATH redirects the output files.
# dbgen builds each output path into a 128-byte stack buffer (print.c: char
# upath[128]), so a long OUTPUT_DIR (e.g. under a git worktree) is a buffer
# overflow and an abort. Generate into a short scratch dir and move the files.
SCRATCH_DIR="$(mktemp -d /tmp/tpch_rf.XXXXXX)"
trap 'rm -rf "$SCRATCH_DIR"' EXIT
(cd "$DBGEN_DIR" && DSS_PATH="$SCRATCH_DIR" ./dbgen -f -q -s "$SF" -U "$NUM_SETS")
mv "$SCRATCH_DIR"/* "$OUTPUT_DIR/"

STATUS=0
for ((n = 1; n <= NUM_SETS; n++)); do
    for f in "orders.tbl.u$n" "lineitem.tbl.u$n" "delete.$n"; do
        if [ ! -s "$OUTPUT_DIR/$f" ]; then
            echo "ERROR: expected refresh file missing or empty: $OUTPUT_DIR/$f"
            STATUS=1
        fi
    done
done
[ "$STATUS" -ne 0 ] && exit "$STATUS"

echo ""
echo "Update set row counts:"
for ((n = 1; n <= NUM_SETS; n++)); do
    printf '  set %-3s orders=%-8s lineitem=%-8s delete_keys=%s\n' "$n" \
        "$(wc -l < "$OUTPUT_DIR/orders.tbl.u$n")" \
        "$(wc -l < "$OUTPUT_DIR/lineitem.tbl.u$n")" \
        "$(wc -l < "$OUTPUT_DIR/delete.$n")"
done
echo "Done."
