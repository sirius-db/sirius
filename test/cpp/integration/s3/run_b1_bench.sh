#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
S3_TEST_BIN="${S3_TEST_BIN:-${PROJECT_ROOT}/build/release/extension/sirius/test/cpp/sirius_unittest}"
DOC_OUTPUT="/Users/tengyu/code/doc/s3support/pr6-b1-bench-results.md"

if [[ -z "${SIRIUS_BENCH_OUTPUT_PATH:-}" ]]; then
  if [[ -d "$(dirname "${DOC_OUTPUT}")" ]]; then
    export SIRIUS_BENCH_OUTPUT_PATH="${DOC_OUTPUT}"
  else
    export SIRIUS_BENCH_OUTPUT_PATH="${SCRIPT_DIR}/pr6-b1-bench-results.md"
  fi
fi

if [[ ! -x "${S3_TEST_BIN}" ]]; then
  echo "run_b1_bench: ${S3_TEST_BIN} not found; run make release first" >&2
  exit 1
fi

# MinIO is now started by the test binary itself (testcontainers); just opt in
# and request the SF10 fixture. No s3-up/s3-down, no env.sh to source.
export SIRIUS_TEST_S3_AUTO=1
export SIRIUS_TEST_S3_LARGE=1
export SIRIUS_TEST_S3_STRICT=1
export SIRIUS_BENCH_BRANCH="$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD)"
export SIRIUS_BENCH_SHA="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD)"
export SIRIUS_BENCH_DATE="$(date +%F)"
export SIRIUS_BENCH_HOST="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)"

echo "[b1-bench] output: ${SIRIUS_BENCH_OUTPUT_PATH}"
"${S3_TEST_BIN}" "[!benchmark][b1_bench]"
