#!/usr/bin/env bash
# Generate one sirius.yaml per CN for the pinned-table benchmark.
#
# The pin compression keys are reachable ONLY via --sirius-config (the CN's
# --gpu-memory-limit carve-out flags and --sirius-config are mutually
# exclusive), so this benchmark always launches CNs with a full YAML.
#
# Env (defaults target a ~96 GiB card; see README for GB200 sizing):
#   NUM_CNS     number of CNs / GPUs                 (default 2)
#   GPU_MEM     engine GPU pool per CN               (default 60GiB)
#   HOST_MEM    pinned host capacity per CN          (default 180GiB)
#   PLAN_DIR    Simpatico plan dir for the dataset   (default repo tpch_sf1000 plans)
#   OUT_DIR     where cn<i>.yaml land                (default this directory/generated)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
NUM_CNS=${NUM_CNS:-2}
GPU_MEM=${GPU_MEM:-60GiB}
HOST_MEM=${HOST_MEM:-180GiB}
PLAN_DIR=${PLAN_DIR:-$REPO_ROOT/src/compression/simpatico_codegen/plans/tpch_sf1000}
OUT_DIR=${OUT_DIR:-$HERE/generated}

mkdir -p "$OUT_DIR"
for i in $(seq 0 $((NUM_CNS - 1))); do
  cat > "$OUT_DIR/cn$i.yaml" <<EOF
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_bytes: "$GPU_MEM"
      reservation_limit_fraction: 1.0
    host:
      capacity_bytes: "$HOST_MEM"
  operator_params:
    scan_task_batch_size: "1GiB"
    hash_partition_bytes: "1GiB"
    concat_batch_bytes: "1GiB"
    max_build_hash_table_bytes: "2GiB"
  compression:
    enable_pin_table_compression: true
    input_plan_dir: "$PLAN_DIR"
  telemetry:
    output_directory: ".cn$i/telemetry"
EOF
done
echo "wrote $NUM_CNS config(s) to $OUT_DIR (GPU_MEM=$GPU_MEM HOST_MEM=$HOST_MEM)"
