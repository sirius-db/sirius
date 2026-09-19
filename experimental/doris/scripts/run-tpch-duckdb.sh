#!/usr/bin/env bash
# Runs the TPC-H queries (sql/tpch/qNN.sql, the same files the FE runs) straight through the
# Sirius build tree's DuckDB shell — the benchmark's single-process references (plan-doc
# experiments/sf10-bench §3): R1 = DuckDB on the CPU (SIRIUS_DISABLE=1), R2 = Sirius's
# transparent path (DuckDB plans, Sirius executes on the GPU). One DuckDB process per call,
# all queries in it back to back (like the upstream harness test/tpch_performance/
# run_tpch_parquet.sh: a fresh process would re-initialize the engine per query).
#
#   scripts/run-tpch-duckdb.sh --data DIR --engine duckdb|sirius [--config sirius.yaml]
#                              [--pin gpu|host] [--threads N] [--queries LIST] [--out DIR]
#                              [--expected DIR] [--ulps N] [--tolerance T] [--no-validate]
#
#   --data DIR         dataset root with <table>/*.parquet
#   --engine           duckdb: CPU only; sirius: the GPU path (needs --config or
#                      SIRIUS_CONFIG_FILE)
#   --config FILE      Sirius config (SIRIUS_CONFIG_FILE) for --engine sirius
#   --pin TIER         --engine sirius only: CALL pin_table for every column of every table
#                      before the queries (test/tpch_performance/tpch_pin_columns.py pin-all),
#                      the tables stay resident in that tier — the "engine upper bound" row
#   --threads N        PRAGMA threads (default: DuckDB's default, all cores)
#   --queries LIST     comma-separated query numbers (default 1..22)
#   --out DIR          per-query artifacts (default log/tpch-duckdb): <q>/result.tsv,
#                      timings.csv (query, rows, wall_ms, engine_ms, query_id, start_ms,
#                      end_ms — wall_ms is the shell's `.timer` real time, engine_ms and
#                      query_id are `-`), all.sql, stdout.log, stderr.log
#   --expected DIR     validate every result.tsv against the DuckDB baseline (default
#                      tests/expected/tpch-sf1; skipped when DIR does not exist)
#   --ulps / --tolerance / --no-validate   as in run-tpch.sh
#
# Env: SIRIUS_DUCKDB (default ../../build/release/duckdb).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DUCKDB="${SIRIUS_DUCKDB:-$(cd ../.. && pwd)/build/release/duckdb}"
PIN_SCRIPT="$(cd ../.. && pwd)/test/tpch_performance/tpch_pin_columns.py"

DATA=""
ENGINE=""
CONFIG="${SIRIUS_CONFIG_FILE:-}"
PIN=""
THREADS=""
QUERIES=""
OUT="${ROOT}/log/tpch-duckdb"
EXPECTED="${ROOT}/tests/expected/tpch-sf1"
VALIDATE=true
ULPS=1
TOLERANCE=1e-9
while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --pin) PIN="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --queries) QUERIES="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --expected) EXPECTED="$2"; shift 2 ;;
        --ulps) ULPS="$2"; shift 2 ;;
        --tolerance) TOLERANCE="$2"; shift 2 ;;
        --no-validate) VALIDATE=false; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "${DATA}" ] || { echo "error: --data DIR is required" >&2; exit 2; }
[ -x "${DUCKDB}" ] || { echo "error: no DuckDB shell at ${DUCKDB} (build the engine: pixi run make)" >&2; exit 1; }
DATA="$(cd "${DATA}" && pwd)"
[ -n "${QUERIES}" ] || QUERIES=$(seq -s, 1 22)
case "${ENGINE}" in
    duckdb)
        export SIRIUS_DISABLE=1
        [ -z "${PIN}" ] || { echo "error: --pin needs --engine sirius" >&2; exit 2; }
        ;;
    sirius)
        unset SIRIUS_DISABLE
        [ -n "${CONFIG}" ] || { echo "error: --engine sirius needs --config FILE (or SIRIUS_CONFIG_FILE)" >&2; exit 2; }
        export SIRIUS_CONFIG_FILE="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
        ;;
    *) echo "error: --engine must be duckdb or sirius" >&2; exit 2 ;;
esac

mkdir -p "${OUT}"
script="${OUT}/all.sql"
{
    echo ".mode tabs"
    echo ".headers on"
    echo ".nullvalue NULL"
    [ -z "${THREADS}" ] || echo "PRAGMA threads=${THREADS};"
    # Views over explicit file lists (what pin_table's glob must expand to, see
    # tpch_pin_columns.py); revenue0 as in sql/tpch-views.sql.
    for table in nation region part supplier partsupp customer orders lineitem; do
        files=$(ls "${DATA}/${table}"/*.parquet | sed "s|.*|'&'|" | paste -sd, -)
        echo "CREATE VIEW ${table} AS SELECT * FROM read_parquet([${files}]);"
    done
    cat <<'SQL'
CREATE VIEW revenue0 (supplier_no, total_revenue) AS
SELECT l_suppkey, sum(l_extendedprice * (1 - l_discount))
FROM lineitem
WHERE l_shipdate >= date '1996-01-01' AND l_shipdate < date '1996-01-01' + interval '3' month
GROUP BY l_suppkey;
SQL
    if [ -n "${PIN}" ]; then
        echo ".print __PIN__"
        SIRIUS_PIN_TIER="${PIN}" python3 "${PIN_SCRIPT}" pin-all "${DATA}"
    fi
    echo ".timer on"
    IFS=',' read -ra query_numbers <<< "${QUERIES}"
    for n in "${query_numbers[@]}"; do
        q=$(printf "q%02d" "${n}")
        sql_file="sql/tpch/${q}.sql"
        [ -f "${sql_file}" ] || { echo "missing ${sql_file}" >&2; exit 1; }
        rm -rf "${OUT}/${q}"
        mkdir -p "${OUT}/${q}"
        cp "${sql_file}" "${OUT}/${q}/query.sql"
        # Marker, then the query's start time, the query itself (its result to a file, its
        # .timer line to stdout), then its end time; all three .timer lines follow the marker
        # on stdout and the parser takes the middle one.
        echo ".print __Q__ ${q}"
        echo ".output ${OUT}/${q}/start.txt"
        echo "SELECT epoch_ms(current_timestamp)::BIGINT AS start_ms;"
        echo ".output ${OUT}/${q}/result.tsv"
        grep -v '^--' "${sql_file}" | sed -e 's/;[[:space:]]*$//'
        echo ";"
        echo ".output ${OUT}/${q}/end.txt"
        echo "SELECT epoch_ms(current_timestamp)::BIGINT AS end_ms;"
        echo ".output"
    done
    echo ".print __END__"
} > "${script}"

echo "==> ${ENGINE} (${DUCKDB}) over ${DATA}, queries ${QUERIES}${PIN:+, pinned to ${PIN}}"
start=$(date +%s)
"${DUCKDB}" < "${script}" > "${OUT}/stdout.log" 2> "${OUT}/stderr.log" || echo "warning: the shell exited with status $? (see ${OUT}/stderr.log)" >&2
echo "==> shell finished in $(( $(date +%s) - start )) s"

# stdout: "__Q__ qNN" then three ".timer" lines ("Run Time (s): real 0.123 user ... sys ...");
# a query that failed has no result and fewer lines (its error is in stderr.log).
timings="${OUT}/timings.csv"
echo "query,rows,wall_ms,engine_ms,query_id,start_ms,end_ms" > "${timings}"
passed=0
failed=0
IFS=',' read -ra query_numbers <<< "${QUERIES}"
for n in "${query_numbers[@]}"; do
    q=$(printf "q%02d" "${n}")
    qdir="${OUT}/${q}"
    real=$(awk -v q="${q}" '$1 == "__Q__" { on = ($2 == q); n = 0; next } on && $1 == "Run" { n++; if (n == 2) { print $5; exit } }' "${OUT}/stdout.log")
    start_ms=$(sed -n 2p "${qdir}/start.txt" 2>/dev/null || true)
    end_ms=$(sed -n 2p "${qdir}/end.txt" 2>/dev/null || true)
    rows=0
    if [ -s "${qdir}/result.tsv" ] && [ -n "${real}" ] && [ -n "${end_ms}" ]; then
        rows=$(($(wc -l < "${qdir}/result.tsv") - 1))
        wall_ms=$(awk -v r="${real}" 'BEGIN { printf "%d", r * 1000 + 0.5 }')
        echo "  ${q}: ok, ${rows} row(s) in ${wall_ms} ms"
        passed=$((passed + 1))
    else
        wall_ms="-"
        echo "  ${q}: FAILED: $(grep -m1 -i "error" "${OUT}/stderr.log" | head -c 200)"
        failed=$((failed + 1))
    fi
    echo "${q},${rows},${wall_ms},-,-,${start_ms:--},${end_ms:--}" >> "${timings}"
done
echo "==> ${passed} ok, ${failed} failed; artifacts in ${OUT}"

validated=0
if [ "${VALIDATE}" = true ] && [ -d "${EXPECTED}" ]; then
    python3 scripts/validate_tpch_results.py validate --actual "${OUT}" --expected "${EXPECTED}" \
        --queries "${QUERIES}" --csv "${OUT}/summary.csv" --ulps "${ULPS}" --tolerance "${TOLERANCE}" || validated=$?
elif [ "${VALIDATE}" = true ]; then
    echo "==> no expected results to validate against (${EXPECTED}); see --expected"
fi
[ "${failed}" -eq 0 ] && [ "${validated}" -eq 0 ]
