#!/usr/bin/env bash
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: bash $0 ENGINE_ROOT HOST_INPUT CPU_REFERENCES NEW_OUTPUT" >&2
  echo "Run on London g7e.8xlarge with Sirius 26b5caba already built." >&2
  exit 2
fi

study_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
engine_root=$(realpath -- "$1")
host_input=$(realpath -- "$2")
references=$(realpath -- "$3")
output=$(realpath -m -- "$4")

export CONDA_OVERRIDE_CUDA=13.5 AWS_REGION=eu-west-2 AWS_DEFAULT_REGION=eu-west-2
unset AWS_PROFILE AWS_DEFAULT_PROFILE

exec pixi run --locked --manifest-path "$engine_root/pixi.toml" \
  python "$study_dir/../../run.py" \
  --source-dir "$engine_root" \
  --input s3://sae-dataproc-aws-dev-sirius-scratch-eu-west-2/datasets/tpch_sf1000_regenerated/ \
  --host-input "$host_input" \
  --references "$references" \
  --output "$output" \
  --order compressed-first
