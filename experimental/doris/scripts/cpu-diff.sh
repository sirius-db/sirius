#!/usr/bin/env bash
# CPU differential of the translator (MVP-A0 preparation, no GPU): every captured dispatch in
# the corpus is stitched into one Substrait plan, the plan bytes are run through DuckDB's
# substrait consumer (the reader Sirius compiles into libsirius, built by
# scripts/build-duckdb-substrait.sh) and the rows are validated against the DuckDB baseline
# in tests/expected/tpch-sf1.
#
#   scripts/cpu-diff.sh [--data DIR] [--queries 1,6,...] [--corpus DIR] [--out DIR]
#
#   --data DIR      dataset root with <table>/*.parquet (default: /tmp/tpch-sf1, the path the
#                   corpus was captured against — the plans are re-rooted to it)
#   --queries LIST  comma-separated query numbers or names (default: every qNN in the corpus)
#   --corpus DIR    captured dispatches (default: tests/fixtures/tpch)
#   --sql-dir DIR   the queries' SQL, for the ORDER BY rules and to generate expected results
#                   that are missing (default: sql/tpch)
#   --expected DIR  expected results (default: tests/expected/tpch-sf1)
#   --out DIR       plans, explain text and per-query results (default: log/cpu-diff)
#
# The gap probes that translate (G-13 SELECT DISTINCT) go through the same path:
#   scripts/cpu-diff.sh --corpus tests/fixtures/gaps --sql-dir sql/gaps --expected tests/expected/gaps-sf1 \
#       --queries g13-distinct,g13-distinct-topn --out log/cpu-diff-gaps
#
# Runs cargo through the `be` pixi environment and python through `check`; call it as
# `pixi run -e check tpch-cpu-diff -- [args]` or directly.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
DATA="/tmp/tpch-sf1"
CORPUS_ROOT="/tmp/tpch-sf1"
CORPUS="${ROOT}/tests/fixtures/tpch"
SQL_DIR="${ROOT}/sql/tpch"
EXPECTED="${ROOT}/tests/expected/tpch-sf1"
OUT="${ROOT}/log/cpu-diff"
QUERIES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="$2"; shift 2 ;;
        --queries) QUERIES="$2"; shift 2 ;;
        --corpus) CORPUS="$2"; shift 2 ;;
        --sql-dir) SQL_DIR="$2"; shift 2 ;;
        --expected) EXPECTED="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
DATA="$(cd "${DATA}" && pwd -P)"
mkdir -p "${OUT}/plans"

if [ -z "${QUERIES}" ]; then
    QUERIES=$(ls -d "${CORPUS}"/*/ | xargs -n1 basename | paste -sd, -)
fi

if [ ! -d "${EXPECTED}" ]; then
    echo "==> no expected results in ${EXPECTED}; generating them with DuckDB over ${DATA}"
    pixi run -e check python scripts/validate_tpch_results.py expected \
        --data "${DATA}" --sql-dir "${SQL_DIR}" --out "${EXPECTED}" --queries "${QUERIES}"
fi

echo "==> building dump-fragments"
pixi run -e be cargo build -q --no-default-features --bin dump-fragments
DUMP="${ROOT}/target/debug/dump-fragments"

echo "==> exporting stitched plans to ${OUT}/plans (scan paths ${CORPUS_ROOT} -> ${DATA})"
exported=0
IFS=',' read -ra names <<< "${QUERIES}"
for n in "${names[@]}"; do
    case "${n}" in
        *[!0-9]*) q="${n}" ;;
        *) q=$(printf "q%02d" "${n}") ;;
    esac
    payload=$(ls "${CORPUS}/${q}"/batch-*-request.tcompact 2>/dev/null | head -1 || true)
    if [ -z "${payload}" ]; then
        echo "  ${q}: no captured dispatch in ${CORPUS}/${q}" >&2
        continue
    fi
    if "${DUMP}" --stitch --write-plan "${OUT}/plans/${q}.substrait" \
            --rewrite-path "${CORPUS_ROOT}=${DATA}" "${payload}" > "${OUT}/plans/${q}.explain.txt" 2> "${OUT}/plans/${q}.log"; then
        exported=$((exported + 1))
    else
        echo "  ${q}: not exported: $(tail -1 "${OUT}/plans/${q}.log")" >&2
        rm -f "${OUT}/plans/${q}.substrait"
    fi
done
echo "==> ${exported} plan(s) exported"

pixi run -e check python scripts/validate_tpch_results.py consume \
    --plans "${OUT}/plans" --expected "${EXPECTED}" --sql-dir "${SQL_DIR}" --out "${OUT}" \
    --queries "${QUERIES}" --csv "${OUT}/summary.csv"
