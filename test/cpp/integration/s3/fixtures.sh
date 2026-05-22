#!/usr/bin/env bash
# Generate local S3 fixtures and upload them to the MinIO container.
#
# Prerequisites (handled by `make s3-up` / `make s3-bench-fixtures`):
#   - MinIO container running at http://127.0.0.1:9000 with credentials
#     minioadmin/minioadmin (see docker-compose.yml).
#   - Python 3 (stdlib only - no extra deps).
#   - build/release/duckdb only when --perf is requested.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--perf] [--help]

  no flag   Generate and upload the standard small S3 fixtures.
  --perf    Generate standard fixtures, then generate and upload the SF10
            lineitem benchmark fixture.
  --help    Show this help.
EOF
}

PERF_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --perf)
      PERF_MODE=1
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      echo "[fixtures] unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures/local"
PARQUET_SOURCE="${SIRIUS_TEST_S3_PARQUET_SOURCE:-${PROJECT_ROOT}/test/cpp/integration/data/parquet}"
MC_IMAGE="${MC_IMAGE:-minio/mc:RELEASE.2025-08-13T08-35-41Z-cpuv1}"

if [[ "${PERF_MODE}" -eq 1 ]]; then
  ENDPOINT="${SIRIUS_BENCH_S3_ENDPOINT:-${SIRIUS_TEST_S3_ENDPOINT:-http://127.0.0.1:9000}}"
  ACCESS_KEY="${SIRIUS_BENCH_S3_ACCESS_KEY:-${SIRIUS_TEST_S3_ACCESS_KEY:-minioadmin}}"
  SECRET_KEY="${SIRIUS_BENCH_S3_SECRET_KEY:-${SIRIUS_TEST_S3_SECRET_KEY:-minioadmin}}"
  BUCKET="${SIRIUS_BENCH_S3_BUCKET:-${SIRIUS_TEST_S3_BUCKET:-sirius-test}}"
else
  ENDPOINT="${SIRIUS_TEST_S3_ENDPOINT:-http://127.0.0.1:9000}"
  ACCESS_KEY="${SIRIUS_TEST_S3_ACCESS_KEY:-minioadmin}"
  SECRET_KEY="${SIRIUS_TEST_S3_SECRET_KEY:-minioadmin}"
  BUCKET="${SIRIUS_TEST_S3_BUCKET:-sirius-test}"
fi

KEY="${SIRIUS_BENCH_S3_KEY:-tpch/lineitem_sf10.parquet}"
WORK_DIR="${SIRIUS_BENCH_WORK_DIR:-${SCRIPT_DIR}/fixtures/generated}"
PERF_PARQUET="${WORK_DIR}/lineitem_sf10.parquet"
PERF_DB="${WORK_DIR}/tpch_sf10.duckdb"

resolve_duckdb() {
  local candidate="${DUCKDB:-${PROJECT_ROOT}/build/release/duckdb}"
  if [[ -x "${candidate}" ]]; then
    DUCKDB_BIN="${candidate}"
    return
  fi
  if command -v "${candidate}" >/dev/null 2>&1; then
    DUCKDB_BIN="$(command -v "${candidate}")"
    return
  fi

  cat >&2 <<EOF
[fixtures] DuckDB binary not found for --perf: ${candidate}
[fixtures] Run \`make release\` first, or pass DUCKDB=/path/to/duckdb.
EOF
  exit 1
}

echo "[fixtures] generating local fixtures under ${FIXTURE_DIR}"
python3 "${SCRIPT_DIR}/generate_fixtures.py" \
  --out "${FIXTURE_DIR}" \
  --parquet-source "${PARQUET_SOURCE}"

if [[ "${PERF_MODE}" -eq 1 ]]; then
  mkdir -p "${WORK_DIR}"
fi

echo "[fixtures] waiting for MinIO at ${ENDPOINT}"
for _ in $(seq 1 30); do
  if curl -sf "${ENDPOINT}/minio/health/ready" >/dev/null; then
    break
  fi
  sleep 1
done
if ! curl -sf "${ENDPOINT}/minio/health/ready" >/dev/null; then
  echo "[fixtures] MinIO did not become ready at ${ENDPOINT}; is it up?" >&2
  exit 1
fi

# mc talks to the host-network MinIO. host.docker.internal works on macOS;
# on Linux we use --network=host + the loopback endpoint.
if [[ "$(uname -s)" == "Linux" ]]; then
  MC_NET=(--network=host)
  MC_ENDPOINT="${ENDPOINT}"
else
  MC_NET=()
  MC_ENDPOINT="${ENDPOINT/127.0.0.1/host.docker.internal}"
fi

mc_run() {
  local -a mounts=(-v "${FIXTURE_DIR}:/fixtures:ro")
  if [[ "${PERF_MODE}" -eq 1 ]]; then
    mounts+=(-v "${WORK_DIR}:/work:ro")
  fi

  docker run --rm "${MC_NET[@]}" \
    "${mounts[@]}" \
    --entrypoint /bin/sh \
    "${MC_IMAGE}" \
    -c "mc alias set local ${MC_ENDPOINT} ${ACCESS_KEY} ${SECRET_KEY} >/dev/null && $*"
}

echo "[fixtures] ensuring bucket ${BUCKET} exists"
mc_run "mc mb --ignore-existing local/${BUCKET}"

echo "[fixtures] uploading fixtures to s3://${BUCKET}/"
mc_run "mc cp --recursive /fixtures/ local/${BUCKET}/"

if [[ "${PERF_MODE}" -eq 1 ]]; then
  resolve_duckdb
  export SIRIUS_CONFIG_FILE="${SIRIUS_CONFIG_FILE:-${SCRIPT_DIR}/sirius.yaml}"

  echo "[fixtures] generating SF10 lineitem benchmark parquet with ${DUCKDB_BIN}"
  rm -f "${PERF_DB}" "${PERF_PARQUET}"
  "${DUCKDB_BIN}" "${PERF_DB}" <<SQL
LOAD tpch;
CALL dbgen(sf=10);
COPY (
  SELECT * FROM lineitem
) TO '${PERF_PARQUET}' (FORMAT PARQUET);
SQL
  rm -f "${PERF_DB}"

  echo "[fixtures] uploading perf fixture to s3://${BUCKET}/${KEY}"
  mc_run "mc cp /work/$(basename "${PERF_PARQUET}") local/${BUCKET}/${KEY}"

  bytes="$(wc -c < "${PERF_PARQUET}" | tr -d ' ')"
  cat <<EOF
[fixtures] S3 perf dataset uploaded.
  backend: ${ENDPOINT}
  bucket : ${BUCKET}
  key    : ${KEY}
  local  : ${PERF_PARQUET}
  bytes  : ${bytes}
EOF
fi

echo "[fixtures] listing s3://${BUCKET}/"
mc_run "mc ls --recursive local/${BUCKET}"

echo "[fixtures] done"
