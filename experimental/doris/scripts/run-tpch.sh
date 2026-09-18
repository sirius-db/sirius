#!/usr/bin/env bash
# Runs the 22 TPC-H queries (sql/tpch/qNN.sql) through the local FE against the Sirius
# backend, and collects what the FE sent us.
#
#   scripts/run-tpch.sh --data DIR [--queries 1,6,...] [--translate-only] [--out DIR]
#   scripts/run-tpch.sh --data DIR --sql-dir sql/gaps --translate-only --out tests/fixtures/gaps
#
#   --data DIR         dataset root with <table>/part.N.parquet (generate_tpch_data.sh)
#   --sql-dir DIR      directory of <name>.sql files to run (default: sql/tpch)
#   --queries LIST     comma-separated query numbers (qNN in --sql-dir) or file stems
#                      (default: 1..22 for sql/tpch, every .sql file otherwise)
#   --translate-only   corpus mode: the backend runs with SIRIUS_BE_TRANSLATE_ONLY=1, every
#                      query is expected to fail at fetch_data with a message saying whether
#                      the dispatch translated into one Substrait plan, and the fragment dump
#                      the backend wrote for it (SIRIUS_BE_DUMP_FRAGMENTS, default log/dump)
#                      is copied into --out/<name>/ together with the EXPLAIN output. The
#                      summary line counts the queries that did not translate (expected for
#                      sql/gaps); the exit status only reflects queries that were not captured.
#   --out DIR          where per-query artifacts go (default: tests/fixtures/tpch in corpus
#                      mode, log/tpch otherwise)
#   --dump-dir DIR     the backend's SIRIUS_BE_DUMP_FRAGMENTS (default: log/dump)
#
# Needs the `mysql` client (pixi run -e fe ...) and a healthy FE + registered backend
# (scripts/fe.sh start; scripts/be.sh start).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
MYSQL=(mysql -h 127.0.0.1 -P "${FE_QUERY_PORT:-9030}" -u root)

DATA=""
SQL_DIR="sql/tpch"
QUERIES=""
TRANSLATE_ONLY=false
OUT=""
DUMP_DIR="${SIRIUS_BE_DUMP_FRAGMENTS:-${ROOT}/log/dump}"
while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --sql-dir) SQL_DIR="$2"; shift 2 ;;
        --queries) QUERIES="$2"; shift 2 ;;
        --translate-only) TRANSLATE_ONLY=true; shift ;;
        --out) OUT="$2"; shift 2 ;;
        --dump-dir) DUMP_DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ -z "${DATA}" ]; then
    echo "error: --data DIR is required" >&2
    exit 2
fi
DATA="$(cd "${DATA}" && pwd)"
SQL_DIR="${SQL_DIR%/}"
if [ -z "${QUERIES}" ]; then
    if [ "${SQL_DIR}" = "sql/tpch" ]; then
        QUERIES=$(seq -s, 1 22)
    else
        QUERIES=$(ls "${SQL_DIR}"/*.sql | xargs -n1 basename | sed 's/\.sql$//' | paste -sd, -)
    fi
fi
if [ -z "${OUT}" ]; then
    if [ "${TRANSLATE_ONLY}" = true ]; then OUT="${ROOT}/tests/fixtures/tpch"; else OUT="${ROOT}/log/tpch"; fi
fi
mkdir -p "${OUT}"

if ! "${MYSQL[@]}" -e "SELECT 1" >/dev/null 2>&1 && ! "${MYSQL[@]}" -e "SHOW BACKENDS" >/dev/null 2>&1; then
    echo "error: cannot reach the FE on 127.0.0.1:${FE_QUERY_PORT:-9030}" >&2
    exit 1
fi

echo "==> applying sql/session.sql and TPC-H views over ${DATA}"
"${MYSQL[@]}" < sql/session.sql
sed "s|@@TPCH_DIR@@|${DATA}|g" sql/tpch-views.sql | "${MYSQL[@]}"

index="${OUT}/INDEX.md"
if [ "${TRANSLATE_ONLY}" = true ]; then
    {
        if [ "${SQL_DIR}" = "sql/tpch" ]; then echo "# TPC-H fragment corpus"; else echo "# Fragment corpus: ${SQL_DIR}"; fi
        echo
        echo "Captured from Doris FE $(source scripts/doris-version.sh && echo "${DORIS_VERSION}") with sql/session.sql, dataset ${DATA} (SF1 parquet)."
        echo 'Per query: `query.sql`, `explain.txt`, the raw dispatch (`batch-NN-request.tcompact`, a'
        echo 'TCompact `TPipelineFragmentParamsList` exactly as `exec_plan_fragment` received it) and'
        echo '`batch-NN-summary.txt` (root fragment first). Pretty-print a dispatch with'
        echo '`cargo run -p sirius-doris-be --no-default-features --bin dump-fragments -- <tcompact>`.' 
        echo
        echo "| query | fragments | shapes (root first) |"
        echo "|---|---|---|"
    } > "${index}"
fi

passed=0
failed=0
untranslated=0
# Dump directories of the queries this run captured (the dump dir may hold older runs).
captured_dumps=()
IFS=',' read -ra query_numbers <<< "${QUERIES}"
for n in "${query_numbers[@]}"; do
    case "${n}" in
        *[!0-9]*) q="${n}" ;;
        *) q=$(printf "q%02d" "${n}") ;;
    esac
    sql_file="${SQL_DIR}/${q}.sql"
    [ -f "${sql_file}" ] || { echo "missing ${sql_file}" >&2; exit 1; }
    sql=$(grep -v '^--' "${sql_file}" | sed -e 's/;[[:space:]]*$//')
    qdir="${OUT}/${q}"
    rm -rf "${qdir}"
    mkdir -p "${qdir}"
    cp "${sql_file}" "${qdir}/query.sql"

    "${MYSQL[@]}" -e "USE tpch; EXPLAIN ${sql}" > "${qdir}/explain.txt" 2>&1 || true

    start=$(date +%s)
    if "${MYSQL[@]}" -e "USE tpch; ${sql}" > "${qdir}/result.tsv" 2> "${qdir}/error.txt"; then
        status="ok"
    else
        status="error"
    fi
    elapsed=$(( $(date +%s) - start ))

    if [ "${TRANSLATE_ONLY}" = true ]; then
        rm -f "${qdir}/result.tsv"
        query_id=$(grep -o 'query [0-9a-f]*-[0-9a-f]* was recorded' "${qdir}/error.txt" | head -1 | awk '{print $2}' || true)
        if [ -z "${query_id}" ] || [ ! -d "${DUMP_DIR}/${query_id}" ]; then
            echo "  ${q}: no fragment dump found (status=${status}); see ${qdir}/error.txt"
            failed=$((failed + 1))
            echo "| ${q} | — | not dispatched: $(head -c 160 "${qdir}/error.txt" | tr '\n|' '  ') |" >> "${index}"
            continue
        fi
        # Keep the replayable wire payload and the summary; the Debug-form fragment text is
        # several MB per query (regenerate it with `cargo run --bin dump-fragments -- <tcompact>`).
        cp "${DUMP_DIR}/${query_id}/"*.tcompact "${DUMP_DIR}/${query_id}/"*-summary.txt "${qdir}/"
        # The backend's verdict on the whole dispatch (stitched into one plan or not).
        if grep -q 'translated into one plan' "${qdir}/error.txt"; then
            verdict="translated into one plan"
        else
            verdict="NOT translated: $(grep -o 'translation failed: .*' "${qdir}/error.txt" | head -c 300 || head -c 300 "${qdir}/error.txt")"
            untranslated=$((untranslated + 1))
        fi
        rm -f "${qdir}/error.txt"
        echo "${query_id}" > "${qdir}/query_id.txt"
        captured_dumps+=("${DUMP_DIR}/${query_id}")
        summary=$(ls "${qdir}"/batch-*-summary.txt | head -1)
        shapes=$(grep '^\[' "${summary}" | sed -e 's/^\[[0-9]*\] //' -e 's/ instances=[0-9]*//' | tr '\n' ';' | sed 's/;$//; s/;/<br>/g')
        fragments=$(grep -c '^\[' "${summary}" || true)
        echo "| ${q} | ${fragments} | ${shapes} |" >> "${index}"
        echo "  ${q}: ${fragments} fragment(s) captured, ${verdict} (${elapsed}s)"
        passed=$((passed + 1))
    else
        if [ "${status}" = ok ]; then
            echo "  ${q}: ok, $(($(wc -l < "${qdir}/result.tsv") - 1)) row(s) in ${elapsed}s"
            passed=$((passed + 1))
        else
            echo "  ${q}: FAILED in ${elapsed}s: $(head -c 200 "${qdir}/error.txt")"
            failed=$((failed + 1))
        fi
    fi
done

if [ "${TRANSLATE_ONLY}" = true ] && [ "${#captured_dumps[@]}" -gt 0 ]; then
    # Coverage summary: what the translator has to handle for this corpus (this run only).
    fragments=$(for dump in "${captured_dumps[@]}"; do cat "${dump}"/batch-*-fragment-*.txt; done)
    count_sorted() { sort | uniq -c | sort -rn | awk '{printf "%s (%s)", $2, $1; if (NR>0) printf ", "}' | sed 's/, $//'; }
    {
        echo
        echo "## Coverage"
        echo
        echo "- plan nodes: $(echo "${fragments}" | grep -o 'node_type: [A-Z_]*_NODE' | sed 's/node_type: //' | count_sorted)"
        echo "- expression nodes: $(echo "${fragments}" | grep -o 'node_type: [A-Z_]*' | grep -v '_NODE$' | sed 's/node_type: //' | count_sorted)"
        echo '  (`NULL_LITERAL` is almost entirely `TFileScanRangeParams.default_value_of_src_slot`, one per scanned column)' 
        echo "- functions (scalar + aggregate, TFunctionName.function_name): $(echo "${fragments}" | grep -A2 'name: TFunctionName {' | grep -o 'function_name: "[^"]*"' | sed 's/function_name: //; s/"//g' | count_sorted)"
        echo "- join ops: $(echo "${fragments}" | grep -o 'join_op: [A-Z_]*' | sed 's/join_op: //' | count_sorted)"
        echo "- sinks: $(cat "${OUT}"/*/batch-*-summary.txt | grep -o 'sink=[A-Z_]*' | sed 's/sink=//' | count_sorted)"
    } >> "${index}"
fi

if [ "${TRANSLATE_ONLY}" = true ]; then
    echo "==> ${passed} captured (${untranslated} not translated), ${failed} not captured; artifacts in ${OUT}"
    [ "${failed}" -eq 0 ]
else
    echo "==> ${passed} ok, ${failed} failed; artifacts in ${OUT}"
    [ "${failed}" -eq 0 ]
fi
