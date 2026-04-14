#!/usr/bin/env bash
# Copyright 2025, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Run sirius_gpu_microbench using a profile from microbench_sweep.json.
#
# Usage:
#   ./tools/microbench/run_microbench_sweep.sh [profile]
#   SIRIUS_GPU_MICROBENCH_BIN=... ./tools/microbench/run_microbench_sweep.sh weekly
#
# Environment:
#   SIRIUS_GPU_MICROBENCH_BIN  — path to binary (default: build/release/.../sirius_gpu_microbench)
#   SIRIUS_MICROBENCH_CONFIG   — path to microbench_sweep.json
#   SIRIUS_MICROBENCH_OUT      — output JSON path (default: under runs/microbench/)
#   SIRIUS_MICROBENCH_PARQUET_FILE / SIRIUS_MICROBENCH_PARQUET_COLUMN — optional Parquet bench

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROFILE="${1:-daily}"
CONFIG="${SIRIUS_MICROBENCH_CONFIG:-$ROOT/tools/microbench/microbench_sweep.json}"
DEFAULT_BIN="$ROOT/build/release/extension/sirius/test/cpp/sirius_gpu_microbench"
BIN="${SIRIUS_GPU_MICROBENCH_BIN:-$DEFAULT_BIN}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$BIN" ]]; then
  echo "Microbench binary not found: $BIN (build with: pixi run make)" >&2
  exit 1
fi

# shellcheck disable=SC1090
eval "$(python3 "$ROOT/tools/microbench/read_profile.py" "$CONFIG" "$PROFILE")"

STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
OUT_DIR="${SIRIUS_MICROBENCH_RUN_DIR:-$ROOT/runs/microbench/${STAMP}_${PROFILE}}"
mkdir -p "$OUT_DIR"
OUT_JSON="${SIRIUS_MICROBENCH_OUT:-$OUT_DIR/benchmark.json}"

set -x
"$BIN" \
  --benchmark_filter="$SIRIUS_MICROBENCH_FILTER" \
  "${SIRIUS_MICROBENCH_EXTRA_ARGS[@]}" \
  --benchmark_out="$OUT_JSON"
set +x

echo "Wrote $OUT_JSON"
echo "SIRIUS_MICROBENCH_LAST_OUT=$OUT_JSON"
echo "SIRIUS_MICROBENCH_LAST_DIR=$OUT_DIR"
