#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
S3_TEST_BIN="${S3_TEST_BIN:-${PROJECT_ROOT}/build/release/extension/sirius/test/cpp/sirius_unittest}"
TEST_NAME="S3 REST read consumers benchmark partitions SF10 by whole row groups"
SAMPLES="${SIRIUS_BENCH_READ_SAMPLES:-5}"
OUTPUT_DIR="${SIRIUS_BENCH_READ_OUTPUT_DIR:-${PROJECT_ROOT}/build/release/extension/sirius/test/cpp/log/read_consumers_$(date +%s)}"
MANIFEST="${OUTPUT_DIR}/samples.tsv"
REUSE_MINIO="${SIRIUS_BENCH_READ_REUSE_MINIO:-1}"
MINIO_IMAGE="${SIRIUS_BENCH_READ_MINIO_IMAGE:-minio/minio:RELEASE.2025-09-07T16-13-09Z-cpuv1}"
OBJECT_KEY="${SIRIUS_BENCH_S3_KEY:-tpch/lineitem_sf10.parquet}"
LOCAL_PARQUET="${SIRIUS_BENCH_READ_LOCAL_PARQUET:-/tmp/sirius-s3-testcontainers/lineitem_sf10.parquet}"
SAMPLE_TIMEOUT_SECONDS="${SIRIUS_BENCH_READ_SAMPLE_TIMEOUT_SECONDS:-180}"
UPLOAD_TIMEOUT_SECONDS="${SIRIUS_BENCH_READ_UPLOAD_TIMEOUT_SECONDS:-300}"
GRAINS="${SIRIUS_BENCH_READ_GRAINS:-1048576 4194304}"

MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"
MINIO_REGION="us-east-1"
MINIO_BUCKET="sirius-test"
SHARED_CONTAINER=""
SHARED_ENDPOINT=""
SHARED_WORK_DIR=""
ACTIVE_BENCH_PID=""
ACTIVE_PIDSTAT_PID=""
ACTIVE_WATCHDOG_PID=""
EXECUTION_INDEX=0
BASELINE_PAYLOAD=""

if [[ "${REUSE_MINIO}" == "1" ]]; then
  BACKEND_LIFECYCLE="shared_warm"
else
  BACKEND_LIFECYCLE="per_sample_testcontainers"
fi

if [[ ! -x "${S3_TEST_BIN}" ]]; then
  echo "run_read_consumers_bench: ${S3_TEST_BIN} not found; run make release first" >&2
  exit 1
fi
if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
  echo "run_read_consumers_bench: Bash 5.1+ is required for the sample watchdog" >&2
  exit 1
fi
for tool in jq pidstat; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "run_read_consumers_bench: ${tool} is required" >&2
    exit 1
  }
done
if [[ "${REUSE_MINIO}" == "1" ]]; then
  for tool in curl docker; do
    command -v "${tool}" >/dev/null 2>&1 || {
      echo "run_read_consumers_bench: ${tool} is required for shared MinIO mode" >&2
      exit 1
    }
  done
fi
if ! [[ "${SAMPLES}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${SAMPLE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "run_read_consumers_bench: samples and timeout must be positive integers" >&2
  exit 1
fi

terminate_process() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    sleep 0.2
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
}

cleanup() {
  terminate_process "${ACTIVE_WATCHDOG_PID}"
  terminate_process "${ACTIVE_BENCH_PID}"
  terminate_process "${ACTIVE_PIDSTAT_PID}"
  if [[ -n "${SHARED_CONTAINER}" ]]; then
    docker rm --force "${SHARED_CONTAINER}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SHARED_WORK_DIR}" ]]; then
    rm -rf "${SHARED_WORK_DIR}"
  fi
}
trap cleanup EXIT

ensure_local_parquet() {
  [[ -s "${LOCAL_PARQUET}" ]] && return

  local duckdb_bin="${SIRIUS_TEST_DUCKDB:-${PROJECT_ROOT}/build/release/duckdb}"
  if [[ ! -x "${duckdb_bin}" ]]; then
    echo "run_read_consumers_bench: ${LOCAL_PARQUET} is absent and ${duckdb_bin} is unavailable" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${LOCAL_PARQUET}")"
  local db="${SHARED_WORK_DIR}/tpch_sf10.duckdb"
  local tmp="${LOCAL_PARQUET}.tmp.$$"
  "${duckdb_bin}" "${db}" -c \
    "INSTALL tpch; LOAD tpch; CALL dbgen(sf=10); COPY (SELECT * FROM lineitem) TO '${tmp}' (FORMAT PARQUET);"
  mv "${tmp}" "${LOCAL_PARQUET}"
}

container_port() {
  local mapping=""
  for _ in $(seq 1 50); do
    mapping="$(docker port "${SHARED_CONTAINER}" 9000/tcp 2>/dev/null | head -n 1)"
    if [[ -n "${mapping}" ]]; then
      printf '%s\n' "${mapping##*:}"
      return
    fi
    sleep 0.1
  done
  echo "run_read_consumers_bench: no mapped MinIO port" >&2
  exit 1
}

wait_for_minio() {
  for _ in $(seq 1 100); do
    if curl --fail --silent --show-error --noproxy '*' --connect-timeout 1 --max-time 2 \
      "${SHARED_ENDPOINT}/minio/health/ready" >/dev/null 2>&1; then
      return
    fi
    sleep 0.1
  done
  echo "run_read_consumers_bench: MinIO did not become ready at ${SHARED_ENDPOINT}" >&2
  exit 1
}

start_shared_minio() {
  SHARED_WORK_DIR="$(mktemp -d /tmp/sirius-read-consumers.XXXXXX)"
  ensure_local_parquet
  SHARED_CONTAINER="sirius-read-consumers-$$"
  docker run --detach --rm --name "${SHARED_CONTAINER}" \
    --env "MINIO_ROOT_USER=${MINIO_ACCESS_KEY}" \
    --env "MINIO_ROOT_PASSWORD=${MINIO_SECRET_KEY}" \
    --env "MINIO_REGION=${MINIO_REGION}" \
    --publish 127.0.0.1::9000 \
    "${MINIO_IMAGE}" server /data --console-address :9001 >/dev/null
  SHARED_ENDPOINT="http://127.0.0.1:$(container_port)"
  wait_for_minio

  local auth=(--aws-sigv4 "aws:amz:${MINIO_REGION}:s3" --user "${MINIO_ACCESS_KEY}:${MINIO_SECRET_KEY}")
  curl --fail --silent --show-error --noproxy '*' --connect-timeout 5 \
    --max-time "${UPLOAD_TIMEOUT_SECONDS}" "${auth[@]}" --request PUT \
    "${SHARED_ENDPOINT}/${MINIO_BUCKET}"
  curl --fail --silent --show-error --noproxy '*' --connect-timeout 5 \
    --max-time "${UPLOAD_TIMEOUT_SECONDS}" "${auth[@]}" --upload-file "${LOCAL_PARQUET}" \
    "${SHARED_ENDPOINT}/${MINIO_BUCKET}/${OBJECT_KEY}"
  echo "[read-consumers] shared MinIO ready at ${SHARED_ENDPOINT}"
}

mkdir -p "${OUTPUT_DIR}"
printf 'arm\treactors\tconnections\tconsumers\tbounce_block_size\tsample\texecution_index\tbackend_lifecycle\tpayload_bytes\twall_ms\tprocess_wall_ms\tharness_overhead_ms\twall_gb_per_sec\tjson\tpidstat\tlog\n' >"${MANIFEST}"

if [[ "${REUSE_MINIO}" == "1" ]]; then
  start_shared_minio
fi

GIT_SHA="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST_NAME="${HOSTNAME:-$(hostname)}"

run_sample() {
  local reactors="$1"
  local connections="$2"
  local consumers="$3"
  local grain="$4"
  local sample="$5"
  local stem="m1_r${reactors}_c${connections}_n${consumers}_grain${grain}_s${sample}"
  local json="${OUTPUT_DIR}/${stem}.json"
  local pidstat_log="${OUTPUT_DIR}/${stem}.pidstat.txt"
  local test_log="${OUTPUT_DIR}/${stem}.test.log"
  local timeout_marker="${OUTPUT_DIR}/${stem}.timeout"
  local -a s3_env=(
    SIRIUS_TEST_S3_AUTO=1
    SIRIUS_TEST_S3_LARGE=1
    SIRIUS_TEST_S3_STRICT=1
  )
  if [[ "${REUSE_MINIO}" == "1" ]]; then
    s3_env=(
      SIRIUS_TEST_S3_AUTO=0
      SIRIUS_TEST_S3_LARGE=1
      SIRIUS_TEST_S3_STRICT=1
      SIRIUS_TEST_S3_ENDPOINT="${SHARED_ENDPOINT}"
      SIRIUS_TEST_S3_REGION="${MINIO_REGION}"
      SIRIUS_TEST_S3_ACCESS_KEY="${MINIO_ACCESS_KEY}"
      SIRIUS_TEST_S3_SECRET_KEY="${MINIO_SECRET_KEY}"
      SIRIUS_TEST_S3_SESSION_TOKEN=
      SIRIUS_TEST_S3_BUCKET="${MINIO_BUCKET}"
      SIRIUS_TEST_S3_KEY=hello.txt
      SIRIUS_TEST_S3_LOCAL_DIR="$(dirname "${LOCAL_PARQUET}")"
      SIRIUS_PR6_LARGE_LOCAL_PARQUET="${LOCAL_PARQUET}"
      SIRIUS_PR6_LARGE_S3_KEY="${OBJECT_KEY}"
      SIRIUS_BENCH_S3_KEY="${OBJECT_KEY}"
    )
  fi

  rm -f "${json}" "${timeout_marker}"
  EXECUTION_INDEX=$((EXECUTION_INDEX + 1))
  local process_start_ns
  process_start_ns="$(date +%s%N)"
  env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY="localhost,127.0.0.1,::1,host.docker.internal" \
    no_proxy="localhost,127.0.0.1,::1,host.docker.internal" \
    "${s3_env[@]}" \
    SIRIUS_BENCH_GIT_SHA="${GIT_SHA}" \
    HOSTNAME="${HOST_NAME}" \
    SIRIUS_BENCH_READ_CONSUMERS="${consumers}" \
    SIRIUS_BENCH_READ_REACTORS="${reactors}" \
    SIRIUS_BENCH_READ_MAX_CONNECTIONS="${connections}" \
    SIRIUS_BENCH_READ_BOUNCE_BLOCK_SIZE="${grain}" \
    SIRIUS_BENCH_READ_TRANSPORT=http \
    SIRIUS_BENCH_READ_SAMPLE="${sample}" \
    SIRIUS_BENCH_READ_ARM=M1 \
    SIRIUS_BENCH_READ_BACKEND_LIFECYCLE="${BACKEND_LIFECYCLE}" \
    SIRIUS_BENCH_READ_BASELINE_BYTES="${BASELINE_PAYLOAD}" \
    SIRIUS_BENCH_READ_CONSUMERS_OUTPUT="${json}" \
    "${S3_TEST_BIN}" "${TEST_NAME}" >"${test_log}" 2>&1 &
  ACTIVE_BENCH_PID=$!
  pidstat -t -p "${ACTIVE_BENCH_PID}" 1 >"${pidstat_log}" 2>&1 &
  ACTIVE_PIDSTAT_PID=$!
  sleep "${SAMPLE_TIMEOUT_SECONDS}" &
  ACTIVE_WATCHDOG_PID=$!

  local completed_pid=""
  local completed_rc=0
  wait -n -p completed_pid "${ACTIVE_BENCH_PID}" "${ACTIVE_WATCHDOG_PID}" || completed_rc=$?
  if [[ "${completed_pid}" == "${ACTIVE_WATCHDOG_PID}" ]]; then
    printf 'sample exceeded %s seconds\n' "${SAMPLE_TIMEOUT_SECONDS}" >"${timeout_marker}"
    terminate_process "${ACTIVE_BENCH_PID}"
    completed_rc=124
  else
    terminate_process "${ACTIVE_WATCHDOG_PID}"
  fi
  ACTIVE_BENCH_PID=""
  ACTIVE_WATCHDOG_PID=""
  kill -INT "${ACTIVE_PIDSTAT_PID}" 2>/dev/null || true
  wait "${ACTIVE_PIDSTAT_PID}" 2>/dev/null || true
  ACTIVE_PIDSTAT_PID=""
  local process_wall_ms=$((($(date +%s%N) - process_start_ns) / 1000000))

  if [[ "${completed_rc}" -ne 0 || -s "${timeout_marker}" || ! -s "${json}" ]]; then
    echo "[read-consumers] ${stem} failed" >&2
    tail -80 "${test_log}" >&2 || true
    exit "${completed_rc:-1}"
  fi

  local pinned_bytes=$((reactors * connections * grain))
  jq -e \
    --argjson reactors "${reactors}" \
    --argjson connections "${connections}" \
    --argjson consumers "${consumers}" \
    --argjson grain "${grain}" \
    --argjson pinned "${pinned_bytes}" \
    --argjson sample "${sample}" \
    '.scenario == "s3_read_consumers" and .arm == "M1" and .transport == "http" and
     .rest_n_reactors == $reactors and .max_connections == $connections and
     .read_consumers == $consumers and .bounce_block_size == $grain and
     .pinned_bytes == $pinned and .sample == $sample and
     .rows == 59986052 and .orderkey_sum == 1799465265420123 and
     (.wall_clock_ms | type == "number") and (.aggregate.chunk_get_count > 0) and
     (.aggregate.payload_bytes_read_total | type == "number") and
     .aggregate.terminal_failures_total == 0 and .aggregate.device_stream_sync_total == 0' \
    "${json}" >/dev/null || {
      echo "[read-consumers] ${stem} produced invalid JSON" >&2
      exit 1
    }

  local thread_index
  for ((thread_index = 0; thread_index < reactors; ++thread_index)); do
    grep -Fq "rest-${thread_index}_worker" "${pidstat_log}" || {
      echo "[read-consumers] ${stem} pidstat missed rest-${thread_index}_worker" >&2
      exit 1
    }
  done
  for ((thread_index = 0; thread_index < consumers; ++thread_index)); do
    grep -Fq "read_cons_${thread_index}" "${pidstat_log}" || {
      echo "[read-consumers] ${stem} pidstat missed read_cons_${thread_index}" >&2
      exit 1
    }
  done

  local payload wall_ms wall_gbps overhead_ms
  payload="$(jq -r '.aggregate.payload_bytes_read_total' "${json}")"
  wall_ms="$(jq -r '.wall_clock_ms' "${json}")"
  wall_gbps="$(jq -r '.effective_wall_gb_per_sec' "${json}")"
  overhead_ms="$(awk -v process="${process_wall_ms}" -v measured="${wall_ms}" 'BEGIN { printf "%.3f", process - measured }')"
  if [[ -n "${BASELINE_PAYLOAD}" ]]; then
    awk -v payload="${payload}" -v baseline="${BASELINE_PAYLOAD}" 'BEGIN {
      ratio = payload / baseline
      if (ratio < 0.98 || ratio > 1.02) exit 1
    }' || {
      echo "[read-consumers] payload guardrail failed: ${payload} vs ${BASELINE_PAYLOAD}" >&2
      exit 1
    }
  else
    BASELINE_PAYLOAD="${payload}"
  fi
  printf 'M1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${reactors}" "${connections}" "${consumers}" "${grain}" "${sample}" "${EXECUTION_INDEX}" \
    "${BACKEND_LIFECYCLE}" "${payload}" "${wall_ms}" "${process_wall_ms}" "${overhead_ms}" \
    "${wall_gbps}" "${json}" "${pidstat_log}" "${test_log}" >>"${MANIFEST}"
}

run_point() {
  local reactors="$1" connections="$2" consumers="$3" grain="$4"
  local sample
  for sample in $(seq 1 "${SAMPLES}"); do
    run_sample "${reactors}" "${connections}" "${consumers}" "${grain}" "${sample}"
  done
}

for grain in ${GRAINS}; do
  [[ "${grain}" =~ ^[1-9][0-9]*$ ]] || {
    echo "run_read_consumers_bench: SIRIUS_BENCH_READ_GRAINS must contain byte counts" >&2
    exit 1
  }
  run_point 4 8 1 "${grain}"
  run_point 4 8 2 "${grain}"
  run_point 4 8 4 "${grain}"
  run_point 1 32 1 "${grain}"
done

echo "[read-consumers] samples: ${MANIFEST}"
echo "[read-consumers] raw JSON, pidstat, and test logs: ${OUTPUT_DIR}"
