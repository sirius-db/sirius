#!/usr/bin/env bash
# Thread sweep benchmark: Sirius-only on 2M RG parquet.
#
# The functions are intentionally sourceable so the configuration rendering and
# failure behavior can be tested without a GPU or DuckDB build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DUCKDB="${SWEEP_DUCKDB:-$PROJECT_DIR/build/release/duckdb}"
QUERY_DIR="${SWEEP_QUERY_DIR:-$PROJECT_DIR/test/tpch_performance/tpch_queries/orig}"
SF="${SWEEP_SF:-100_rg2m}"
PARQUET_DIR="${SWEEP_PARQUET_DIR:-$PROJECT_DIR/test_datasets/tpch_parquet_sf${SF}}"
if [ -n "${SWEEP_OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="$SWEEP_OUTPUT_DIR"
else
    OUTPUT_DIR="$PROJECT_DIR/benchmark_results_thread_sweep/run_$(date +%Y%m%d-%H%M%S)-$$"
fi

read -r -a QUERIES <<< "${SWEEP_QUERIES:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22}"
TPCH_TABLES=(customer lineitem nation orders part partsupp region supplier)
LABELS=(
    "baseline_p8_s4_t4"
    "pipeline_p4_s4_t4"
    "pipeline_p12_s4_t4"
    "pipeline_p16_s4_t4"
    "scan_p8_s3_t4"
    "scan_p8_s8_t4"
    "taskcreator_p8_s4_t2"
    "taskcreator_p8_s4_t8"
)
SHORT_NAMES=(
    "p8/s4/t4"
    "p4/s4/t4"
    "p12/s4/t4"
    "p16/s4/t4"
    "p8/s3/t4"
    "p8/s8/t4"
    "p8/s4/t2"
    "p8/s4/t8"
)

CONFIG_FILE=""
CURRENT_SQL=""
CURRENT_CSV=""
VIEW_SQL=""

die() {
    echo "ERROR: $*" >&2
    return 1
}

cleanup() {
    if [ -n "$CURRENT_SQL" ] && [ -f "$CURRENT_SQL" ]; then
        rm -f "$CURRENT_SQL"
    fi
    if [ -n "$CURRENT_CSV" ] && [ -f "$CURRENT_CSV" ]; then
        rm -f "$CURRENT_CSV"
    fi
    if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
        rm -f "$CONFIG_FILE"
    fi
}

require_positive_integer() {
    local name=$1 value=$2
    case "$value" in
        ''|*[!0-9]*) die "$name must be a positive integer, got '$value'"; return 1 ;;
        0) die "$name must be a positive integer, got '0'"; return 1 ;;
    esac
}

validate_config_values() {
    local pipeline=$1 scan=$2 task_creator=$3
    require_positive_integer "pipeline threads" "$pipeline" || return 1
    require_positive_integer "scan threads" "$scan" || return 1
    require_positive_integer "task creator threads" "$task_creator" || return 1
    if [ "$scan" -lt 3 ]; then
        die "scan threads must be at least 3, got '$scan'"
        return 1
    fi
}

create_temp_config() {
    CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/sirius_thread_sweep.yaml.XXXXXX")"
    export SIRIUS_CONFIG_FILE="$CONFIG_FILE"
}

build_view_sql() {
    local table file_list
    local -a files
    VIEW_SQL=""
    for table in "${TPCH_TABLES[@]}"; do
        files=()
        for file in "$PARQUET_DIR/${table}.parquet" "$PARQUET_DIR/${table}_"*.parquet; do
            [ -f "$file" ] && files+=("'$file'")
        done
        if [ "${#files[@]}" -eq 0 ]; then
            die "no parquet files found for table '$table' under $PARQUET_DIR"
            return 1
        fi
        file_list=$(IFS=,; echo "${files[*]}")
        VIEW_SQL+="CREATE VIEW ${table} AS SELECT * FROM read_parquet([${file_list}]);"$'\n'
    done
}

# update_config <pipeline> <scan> <task_creator>
update_config() {
    local pipeline=$1 scan=$2 task_creator=$3
    validate_config_values "$pipeline" "$scan" "$task_creator" || return 1
    if [ -z "$CONFIG_FILE" ]; then
        die "temporary config has not been created"
        return 1
    fi

    cat > "$CONFIG_FILE" << EOF
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.9
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: 50000000000
      initial_number_pools: 80
      pool_size: 512
      block_size: 1048576
  executor:
    pipeline:
      num_threads: ${pipeline}
    scan_manager:
      num_threads: ${scan}
    task_creator:
      num_threads: ${task_creator}
    downgrade:
      num_threads: 4
EOF
}

# run_sirius_sweep <label> <pipeline> <scan> <task_creator>
run_sirius_sweep() {
    local label=$1 pipeline=$2 scan=$3 task_creator=$4
    local csv="$OUTPUT_DIR/${label}.csv"
    local partial_csv="${csv}.partial"
    local query_file output timing cold warm
    local -a times

    update_config "$pipeline" "$scan" "$task_creator"

    echo ""
    echo "============================================================"
    echo "  $label: pipeline=$pipeline scan=$scan task_creator=$task_creator"
    echo "============================================================"
    CURRENT_CSV="$partial_csv"
    echo "query,cold,warm" > "$partial_csv"

    for q in "${QUERIES[@]}"; do
        query_file="$QUERY_DIR/q${q}.sql"
        if [ ! -f "$query_file" ]; then
            die "query file not found: $query_file"
            return 1
        fi

        CURRENT_SQL=$(mktemp "${TMPDIR:-/tmp}/tpch_sweep_q${q}.sql.XXXXXX")
        {
            printf '%s\n' "$VIEW_SQL"
            echo ".timer on"
            cat "$query_file"
            cat "$query_file"
        } > "$CURRENT_SQL"

        if ! output=$("$DUCKDB" -f "$CURRENT_SQL" 2>&1); then
            echo "$output" >&2
            die "DuckDB failed for $label query $q"
            return 1
        fi
        rm -f "$CURRENT_SQL"
        CURRENT_SQL=""

        times=()
        while IFS= read -r timing; do
            times+=("$timing")
        done < <(printf '%s\n' "$output" | sed -nE 's/.*Run Time \(s\): real ([0-9]+([.][0-9]+)?).*/\1/p')
        if [ "${#times[@]}" -ne 2 ]; then
            echo "$output" >&2
            die "expected exactly two DuckDB timer values for $label query $q, found ${#times[@]}"
            return 1
        fi

        cold=${times[0]}
        warm=${times[1]}
        echo "  Q${q}: cold=${cold}s warm=${warm}s"
        echo "${q},${cold},${warm}" >> "$partial_csv"
    done

    mv "$partial_csv" "$csv"
    CURRENT_CSV=""
}

run_all_sweeps() {
    run_sirius_sweep "baseline_p8_s4_t4" 8 4 4
    run_sirius_sweep "pipeline_p4_s4_t4" 4 4 4
    run_sirius_sweep "pipeline_p12_s4_t4" 12 4 4
    run_sirius_sweep "pipeline_p16_s4_t4" 16 4 4
    run_sirius_sweep "scan_p8_s3_t4" 8 3 4
    run_sirius_sweep "scan_p8_s8_t4" 8 8 4
    run_sirius_sweep "taskcreator_p8_s4_t2" 8 4 2
    run_sirius_sweep "taskcreator_p8_s4_t8" 8 4 8
}

print_summary() {
    local name label csv val total q
    echo ""
    echo "============================================================"
    echo "  Thread Sweep Summary (Sirius cold times)"
    echo "============================================================"
    echo ""

    printf "%-5s" "Query"
    for name in "${SHORT_NAMES[@]}"; do
        printf " | %11s" "$name"
    done
    echo ""

    printf "%-5s" "-----"
    for name in "${SHORT_NAMES[@]}"; do
        printf -- "-+-%-11s" "-----------"
    done
    echo ""

    for q in "${QUERIES[@]}"; do
        printf "%-5s" "Q${q}"
        for label in "${LABELS[@]}"; do
            csv="$OUTPUT_DIR/${label}.csv"
            val=$(awk -F, -v query="$q" '$1 == query { print $2 }' "$csv")
            printf " | %10ss" "$val"
        done
        echo ""
    done

    printf "%-5s" "TOTAL"
    for label in "${LABELS[@]}"; do
        csv="$OUTPUT_DIR/${label}.csv"
        total=$(awk -F, 'NR > 1 { sum += $2 } END { printf "%.1f", sum }' "$csv")
        printf " | %10ss" "$total"
    done
    echo ""
}

main() {
    [ -x "$DUCKDB" ] || { die "DuckDB binary is not executable: $DUCKDB"; return 1; }
    [ -d "$QUERY_DIR" ] || { die "query directory not found: $QUERY_DIR"; return 1; }
    [ -d "$PARQUET_DIR" ] || { die "parquet directory not found: $PARQUET_DIR"; return 1; }

    mkdir -p "$OUTPUT_DIR"
    create_temp_config
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    build_view_sql

    echo "============================================================"
    echo "  Thread Sweep Benchmark (SF${SF}, Sirius-only)"
    echo "============================================================"
    run_all_sweeps
    print_summary

    echo ""
    echo "CSVs saved to: $OUTPUT_DIR/"
    echo "============================================================"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
