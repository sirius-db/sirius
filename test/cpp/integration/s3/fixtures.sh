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
TLS_ENDPOINT="${SIRIUS_TEST_S3_HTTPS_ENDPOINT:-https://127.0.0.1:9443}"

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

wait_for_minio() {
  local endpoint="$1"
  local curl_tls_flag="$2"
  local -a curl_args=(-sf)
  if [[ -n "${curl_tls_flag}" ]]; then
    curl_args+=("${curl_tls_flag}")
  fi
  echo "[fixtures] waiting for MinIO at ${endpoint}"
  for _ in $(seq 1 30); do
    if curl "${curl_args[@]}" "${endpoint}/minio/health/ready" >/dev/null; then
      return
    fi
    sleep 1
  done
  if ! curl "${curl_args[@]}" "${endpoint}/minio/health/ready" >/dev/null; then
    echo "[fixtures] MinIO did not become ready at ${endpoint}; is it up?" >&2
    exit 1
  fi
}

docker_reachable_endpoint() {
  local endpoint="$1"
  if [[ "$(uname -s)" == "Linux" ]]; then
    printf '%s' "${endpoint}"
  else
    printf '%s' "${endpoint/127.0.0.1/host.docker.internal}"
  fi
}

wait_for_minio "${ENDPOINT}" ""
wait_for_minio "${TLS_ENDPOINT}" "-k"

# mc talks to the host-network MinIO. host.docker.internal works on macOS;
# on Linux we use --network=host + the loopback endpoint.
if [[ "$(uname -s)" == "Linux" ]]; then
  MC_NET=(--network=host)
else
  MC_NET=()
fi
MC_ENDPOINT="$(docker_reachable_endpoint "${ENDPOINT}")"
MC_TLS_ENDPOINT="$(docker_reachable_endpoint "${TLS_ENDPOINT}")"

mc_run() {
  local alias_name="$1"
  local endpoint="$2"
  local insecure="$3"
  shift 3

  local -a mounts=(-v "${FIXTURE_DIR}:/fixtures:ro")
  if [[ "${PERF_MODE}" -eq 1 ]]; then
    mounts+=(-v "${WORK_DIR}:/work:ro")
  fi
  local -a mc_tls_flag=()
  if [[ "${insecure}" == "1" ]]; then
    mc_tls_flag=(--insecure)
  fi

  docker run --rm "${MC_NET[@]}" \
    "${mounts[@]}" \
    --entrypoint /bin/sh \
    "${MC_IMAGE}" \
    -c "mc ${mc_tls_flag[*]} alias set ${alias_name} ${endpoint} ${ACCESS_KEY} ${SECRET_KEY} >/dev/null && mc ${mc_tls_flag[*]} $*"
}

upload_standard_fixtures() {
  local alias_name="$1"
  local endpoint="$2"
  local insecure="$3"

  echo "[fixtures] ensuring bucket ${BUCKET} exists at ${endpoint}"
  mc_run "${alias_name}" "${endpoint}" "${insecure}" "mb --ignore-existing ${alias_name}/${BUCKET}"

  echo "[fixtures] uploading fixtures to ${endpoint}/${BUCKET}/"
  mc_run "${alias_name}" "${endpoint}" "${insecure}" "cp --recursive /fixtures/ ${alias_name}/${BUCKET}/"
}

upload_standard_fixtures "local" "${MC_ENDPOINT}" "0"
upload_standard_fixtures "localtls" "${MC_TLS_ENDPOINT}" "1"

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
  mc_run "local" "${MC_ENDPOINT}" "0" "cp /work/$(basename "${PERF_PARQUET}") local/${BUCKET}/${KEY}"

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
mc_run "local" "${MC_ENDPOINT}" "0" "ls --recursive local/${BUCKET}"
echo "[fixtures] listing tls s3://${BUCKET}/"
mc_run "localtls" "${MC_TLS_ENDPOINT}" "1" "ls --recursive localtls/${BUCKET}"

echo "[fixtures] done"
