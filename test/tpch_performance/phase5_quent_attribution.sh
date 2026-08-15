#!/usr/bin/env bash
# Phase-5 Quent attribution pass (measurement plan section 4.5): run the LIMIT queries in BOTH
# arms (flag-off / flag-on via SET enable_top_n_dynamic_filter) with telemetry enabled, labeling
# every execution phase5_<cell>_<qN>_<arm>_<iter> through sirius_set_query_label so Quent events
# attribute back to a specific (query, arm, iteration).
#
# This pass is NEVER a timing cell: telemetry's write cost on the timed path is unmeasured, so
# every quotable number comes from performance_test.py --mode ab with telemetry off. This pass
# exists only to triage a flagged regression (which operator's compute/wall grew, off vs on) and
# for the Q18/Q2 mechanism narrative. Extract the off/on comparison afterwards with
# phase5_quent_extract.py while `pixi run quent` serves the telemetry directory.
#
# Usage:
#   ./phase5_quent_attribution.sh [options] <scale_factor> [query_numbers...]
#
# If no query numbers are given, the LIMIT queries (2 3 10 18 21) are run.
#
# Options:
#   --cell <name>          Cell label component (default: host_pinned)
#   --config <path>        Sirius YAML with telemetry enabled (default: phase5_quent.yaml)
#   --parquet-dir <path>   TPC-H parquet directory
#                          (default: $PROJECT_DIR/test_datasets/tpch_parquet_sf<SF>)
#   --iterations <N>       Iterations per (query, arm) (default: 2)
#   --pin <tier>           Pin per query into 'host' or 'gpu' tier via pin_table, or 'none'
#                          (default: none)
#
# Example (SF1000 pinned-cell attribution):
#   ./phase5_quent_attribution.sh --cell host_pinned --pin host \
#       --parquet-dir /localhome/local-kkristensen/tpch_parquet_sf1000 1000

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/orig"
SIRIUS_CONFIG="$SCRIPT_DIR/phase5_quent.yaml"

CELL="host_pinned"
PARQUET_DIR=""
NUM_ITERATIONS=2
PIN_TIER="none"
while [ $# -gt 0 ]; do
    case "${1:-}" in
        --cell)         CELL="$2";           shift 2 ;;
        --config)       SIRIUS_CONFIG="$2";  shift 2 ;;
        --parquet-dir)  PARQUET_DIR="$2";    shift 2 ;;
        --iterations)   NUM_ITERATIONS="$2"; shift 2 ;;
        --pin)          PIN_TIER="$2";       shift 2 ;;
        *) break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--cell <name>] [--config <path>] [--parquet-dir <path>] [--iterations <N>] [--pin host|gpu|none] <scale_factor> [query_numbers...]"
    exit 1
fi

SF="$1"; shift
QUERIES=("$@")
if [ ${#QUERIES[@]} -eq 0 ]; then
    # The LIMIT queries: the Top-N dynamic filter's whole surface.
    QUERIES=(2 3 10 18 21)
fi

if [ -z "$PARQUET_DIR" ]; then
    PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
fi

[ -x "$DUCKDB" ]        || { echo "ERROR: DuckDB binary not found at $DUCKDB"; exit 1; }
[ -f "$SIRIUS_CONFIG" ] || { echo "ERROR: Sirius config not found: $SIRIUS_CONFIG"; exit 1; }
[ -d "$PARQUET_DIR" ]   || { echo "ERROR: Parquet directory not found: $PARQUET_DIR"; exit 1; }

# Build CREATE VIEW statements for all TPC-H tables.
TPCH_TABLES=(customer lineitem nation orders part partsupp region supplier)
VIEW_SQL=""
for TABLE_NAME in "${TPCH_TABLES[@]}"; do
    FILES=()
    for f in "$PARQUET_DIR/${TABLE_NAME}.parquet" \
             "$PARQUET_DIR/${TABLE_NAME}_"*.parquet \
             "$PARQUET_DIR/${TABLE_NAME}/"*.parquet; do
        [ -f "$f" ] && FILES+=("'$f'")
    done
    FILE_LIST=$(IFS=,; echo "${FILES[*]}")
    VIEW_SQL+="CREATE VIEW ${TABLE_NAME} AS SELECT * FROM read_parquet([${FILE_LIST}]);"$'\n'
done

# One SQL file: views, then per query { pin, per iteration { off arm, on arm }, unpin }. Arms
# interleave within an iteration so slow host drift is shared, mirroring the AB cells this pass
# explains.
TEMP_SQL=$(mktemp /tmp/phase5_quent_XXXXXX.sql)
{
    printf '%s\n' "$VIEW_SQL"
    printf "SET gpu_execution = true;\n"
    printf ".timer on\n"
    for q in "${QUERIES[@]}"; do
        QUERY_FILE="$QUERY_DIR/q${q}.sql"
        if [ ! -f "$QUERY_FILE" ]; then
            echo "WARNING: Query file not found: $QUERY_FILE, skipping Q${q}" >&2
            continue
        fi
        if [ "$PIN_TIER" != "none" ]; then
            SIRIUS_PIN_TIER="$PIN_TIER" python3 "$SCRIPT_DIR/tpch_pin_columns.py" pin "$q" "$PARQUET_DIR" || exit 1
        fi
        for ((iter = 1; iter <= NUM_ITERATIONS; iter++)); do
            for arm in off on; do
                flag=false; [ "$arm" = "on" ] && flag=true
                printf ".print =============== Q%d iter %d arm %s ===============\n" "$q" "$iter" "$arm"
                printf "SET enable_top_n_dynamic_filter = %s;\n" "$flag"
                printf "CALL sirius_set_query_label('phase5_%s_q%d_%s_%d');\n" \
                    "$CELL" "$q" "$arm" "$iter"
                cat "$QUERY_FILE"
                printf '\n'
            done
        done
        if [ "$PIN_TIER" != "none" ]; then
            python3 "$SCRIPT_DIR/tpch_pin_columns.py" unpin "$q" || exit 1
        fi
    done
} > "$TEMP_SQL"

echo "Phase-5 Quent attribution pass (SF${SF}, cell=${CELL})"
echo "  Parquet dir:  $PARQUET_DIR"
echo "  Queries:      ${QUERIES[*]}"
echo "  Iterations:   $NUM_ITERATIONS per arm"
echo "  Pin tier:     $PIN_TIER"
echo "  Config:       $SIRIUS_CONFIG (telemetry must be enabled)"
echo "  Labels:       phase5_${CELL}_q<N>_<off|on>_<iter>"
echo "=========================================="

SIRIUS_DISABLE=0 SIRIUS_CONFIG_FILE="$SIRIUS_CONFIG" "$DUCKDB" -f "$TEMP_SQL"
EXIT=$?
rm -f "$TEMP_SQL"

echo ""
echo "Telemetry written to sirius.telemetry.output_directory from the config."
echo "Extract the off/on comparison with: pixi run python test/tpch_performance/phase5_quent_extract.py --cell ${CELL}"
exit "$EXIT"
