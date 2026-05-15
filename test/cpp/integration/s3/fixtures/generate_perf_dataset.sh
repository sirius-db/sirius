#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../.." && pwd)"

MC="${MC:-mc}"
DUCKDB="${DUCKDB:-duckdb}"
ALIAS="${SIRIUS_BENCH_MC_ALIAS:-sirius-bench}"
ENDPOINT="${SIRIUS_BENCH_S3_ENDPOINT:-${SIRIUS_TEST_S3_ENDPOINT:-http://127.0.0.1:9000}}"
ACCESS_KEY="${SIRIUS_BENCH_S3_ACCESS_KEY:-${SIRIUS_TEST_S3_ACCESS_KEY:-minioadmin}}"
SECRET_KEY="${SIRIUS_BENCH_S3_SECRET_KEY:-${SIRIUS_TEST_S3_SECRET_KEY:-minioadmin}}"
BUCKET="${SIRIUS_BENCH_S3_BUCKET:-sirius-bench}"
KEY="${SIRIUS_BENCH_S3_KEY:-tpch/lineitem_sf10.parquet}"
WORK_DIR="${SIRIUS_BENCH_WORK_DIR:-${PROJECT_ROOT}/test/cpp/integration/s3/fixtures/generated}"
PARQUET="${WORK_DIR}/lineitem_sf10.parquet"
DB="${WORK_DIR}/tpch_sf10.duckdb"

mkdir -p "${WORK_DIR}"

"${MC}" alias set "${ALIAS}" "${ENDPOINT}" "${ACCESS_KEY}" "${SECRET_KEY}" >/dev/null
"${MC}" mb --ignore-existing "${ALIAS}/${BUCKET}" >/dev/null

"${DUCKDB}" "${DB}" <<SQL
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=10);
COPY (
  SELECT
    l_orderkey,
    l_partkey,
    l_suppkey,
    l_linenumber,
    l_quantity,
    l_extendedprice,
    l_discount
  FROM lineitem
) TO '${PARQUET}' (FORMAT PARQUET);
SQL

"${MC}" cp "${PARQUET}" "${ALIAS}/${BUCKET}/${KEY}"

bytes="$(wc -c < "${PARQUET}" | tr -d ' ')"
cat <<EOF
S3 perf dataset uploaded.
  backend: ${ENDPOINT}
  bucket : ${BUCKET}
  key    : ${KEY}
  local  : ${PARQUET}
  bytes  : ${bytes}
EOF
