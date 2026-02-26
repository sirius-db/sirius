#!/usr/bin/env bash
# =============================================================================
# Thread sweep benchmark: Sirius-only on 2M RG parquet
#
# Runs the Sirius queries with different thread configurations to find
# the optimal settings. Only runs Sirius (no DuckDB).
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="$PROJECT_DIR/build/release/duckdb"
QUERY_DIR="$PROJECT_DIR/test/tpch_performance/tpch_queries/gpu"
CONFIG_FILE="$PROJECT_DIR/test/cpp/integration/integration.cfg"

export SIRIUS_CONFIG_FILE="$CONFIG_FILE"

SF="100_rg2m"
QUERIES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22)
PARQUET_DIR="$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}"
OUTPUT_DIR="$PROJECT_DIR/benchmark_results_thread_sweep"
mkdir -p "$OUTPUT_DIR"

# Build view SQL
TPCH_TABLES=(customer lineitem nation orders part partsupp region supplier)
VIEW_SQL=""
for T in "${TPCH_TABLES[@]}"; do
    FILES=()
    for f in "$PARQUET_DIR/${T}.parquet" "$PARQUET_DIR/${T}_"*.parquet; do
        [ -f "$f" ] && FILES+=("'$f'")
    done
    FILE_LIST=$(IFS=,; echo "${FILES[*]}")
    VIEW_SQL+="CREATE VIEW ${T} AS SELECT * FROM read_parquet([${FILE_LIST}]);"$'\n'
done

# update_config <pipeline> <scan> <task_creator>
update_config() {
    local pipeline=$1 scan=$2 task_creator=$3
    cat > "$CONFIG_FILE" << EOF
sirius = {
    topology = {
        num_gpus = 1;
    };
    memory = {
        gpu = {
            usage_limit_fraction = 0.9;
            reservation_limit_fraction = 1.0;
        }
        host = {
            capacity_bytes = 50000000000;
            initial_number_pools = 80;
            pool_size = 512;
            block_size = 1048576;
        };
    };
    executor = {
        pipeline = {
            num_threads = ${pipeline};
        };
        duckdb_scan = {
            num_threads = ${scan};
            cache = true;
        };
        task_creator = {
            num_threads = ${task_creator};
        };
        downgrade = {
            num_threads = 4;
        };
    };
};
EOF
}

# run_sirius_sweep <label> <pipeline> <scan> <task_creator>
run_sirius_sweep() {
    local label=$1 pipeline=$2 scan=$3 task_creator=$4
    local csv="$OUTPUT_DIR/${label}.csv"

    update_config "$pipeline" "$scan" "$task_creator"

    echo ""
    echo "============================================================"
    echo "  $label: pipeline=$pipeline scan=$scan task_creator=$task_creator"
    echo "============================================================"

    echo "query,cold,warm" > "$csv"

    for q in "${QUERIES[@]}"; do
        QUERY_FILE="$QUERY_DIR/q${q}.sql"
        [ ! -f "$QUERY_FILE" ] && continue

        TEMP_SQL=$(mktemp /tmp/tpch_sweep_q${q}_XXXXXX.sql)
        {
            printf '%s\n' "$VIEW_SQL"
            echo ".timer on"
            cat "$QUERY_FILE"
            cat "$QUERY_FILE"
        } > "$TEMP_SQL"

        OUTPUT=$("$DUCKDB" -f "$TEMP_SQL" 2>&1)
        rm -f "$TEMP_SQL"

        readarray -t TIMES < <(echo "$OUTPUT" | grep -oP 'Run Time \(s\): real \K[0-9]+\.[0-9]+')
        cold="${TIMES[0]:-N/A}"
        warm="${TIMES[1]:-N/A}"
        echo "  Q${q}: cold=${cold}s warm=${warm}s"
        echo "${q},${cold},${warm}" >> "$csv"
    done
}

echo "============================================================"
echo "  Thread Sweep Benchmark (SF${SF}, Sirius-only)"
echo "============================================================"

# Baseline: pipeline=8, scan=4, task_creator=4
run_sirius_sweep "baseline_p8_s4_t4" 8 4 4

# Vary pipeline threads (scan=4, task_creator=4)
run_sirius_sweep "pipeline_p4_s4_t4" 4 4 4
run_sirius_sweep "pipeline_p12_s4_t4" 12 4 4
run_sirius_sweep "pipeline_p16_s4_t4" 16 4 4

# Vary scan threads (pipeline=8, task_creator=4)
run_sirius_sweep "scan_p8_s2_t4" 8 2 4
run_sirius_sweep "scan_p8_s8_t4" 8 8 4

# Vary task creator threads (pipeline=8, scan=4)
run_sirius_sweep "taskcreator_p8_s4_t2" 8 4 2
run_sirius_sweep "taskcreator_p8_s4_t8" 8 4 8

# Restore baseline config
update_config 8 4 4

# ---- Print comparison table ----
echo ""
echo "============================================================"
echo "  Thread Sweep Summary (Sirius cold times)"
echo "============================================================"
echo ""

LABELS=(
    "baseline_p8_s4_t4"
    "pipeline_p4_s4_t4"
    "pipeline_p12_s4_t4"
    "pipeline_p16_s4_t4"
    "scan_p8_s2_t4"
    "scan_p8_s8_t4"
    "taskcreator_p8_s4_t2"
    "taskcreator_p8_s4_t8"
)
SHORT_NAMES=(
    "p8/s4/t4"
    "p4/s4/t4"
    "p12/s4/t4"
    "p16/s4/t4"
    "p8/s2/t4"
    "p8/s8/t4"
    "p8/s4/t2"
    "p8/s4/t8"
)

# Header
printf "%-5s" "Query"
for name in "${SHORT_NAMES[@]}"; do
    printf " | %11s" "$name"
done
echo ""

printf "%-5s" "-----"
for name in "${SHORT_NAMES[@]}"; do
    printf "-+-%-11s" "-----------"
done
echo ""

# Data rows
for q in "${QUERIES[@]}"; do
    printf "%-5s" "Q${q}"
    for label in "${LABELS[@]}"; do
        csv="$OUTPUT_DIR/${label}.csv"
        val=$(grep "^${q}," "$csv" 2>/dev/null | cut -d, -f2)
        if [ -n "$val" ] && [ "$val" != "N/A" ]; then
            printf " | %10ss" "$val"
        else
            printf " | %11s" "N/A"
        fi
    done
    echo ""
done

# Totals
printf "%-5s" "TOTAL"
for label in "${LABELS[@]}"; do
    csv="$OUTPUT_DIR/${label}.csv"
    total=$(tail -n +2 "$csv" 2>/dev/null | cut -d, -f2 | grep -v N/A | paste -sd+ | bc 2>/dev/null || echo "N/A")
    if [ -n "$total" ] && [ "$total" != "N/A" ]; then
        printf " | %10ss" "$(printf '%.1f' "$total")"
    else
        printf " | %11s" "N/A"
    fi
done
echo ""

echo ""
echo "CSVs saved to: $OUTPUT_DIR/"
echo "============================================================"
