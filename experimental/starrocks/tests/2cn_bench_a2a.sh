#!/usr/bin/env bash
# All-to-all bandwidth over the packed NIXL hop between two CNs on GPUs 0 and 1, no FE.
# Shaped after distributed-join's all_to_all.cpp and cascade-tpc-shuttle's bench_a2a:
# bytes per GPU split across the two workers, 15 MiB frames, one warm-up + 4 timed rounds.
# Prints one "a2a<size>" line per size from CN0 (worker 0), like their harness.
#
#   /home/ubuntu/sirius-wt/all22/gpu-lock.sh experimental/starrocks/tests/2cn_bench_a2a.sh
#   SIZES="64000000 1024000000" CHUNK_BYTES=$((64<<20)) ... to override.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SR_DIR=$(cd "$HERE/.." && pwd)

# shellcheck source=/dev/null
source /home/ubuntu/sirius-wt/env.sh
export TMPDIR=${TMPDIR:-/opt/dlami/nvme/tmp}
export TOOLS_DIR=${TOOLS_DIR:-/home/ubuntu/sirius-wt/tools}
# shellcheck source=/dev/null
source "$SR_DIR/scripts/cn-env.sh"

OUT=${BENCH_OUT:-/opt/dlami/nvme/tmp/sirius-bench-a2a}
CN_BIN=${CN_BIN:-$SR_DIR/target/release/sirius-starrocks-cn}
SIZES=${SIZES:-64000000 1024000000 2048000000}
CHUNK_BYTES=${CHUNK_BYTES:-15728640}
ROUNDS=${ROUNDS:-4}
PORT_BASE=${PORT_BASE:-9200}
export SIRIUS_EXCHANGE_STAGING_BYTES=${SIRIUS_EXCHANGE_STAGING_BYTES:-1GiB}
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
unset CUDA_VISIBLE_DEVICES

mkdir -p "$OUT"
cat >"$OUT/sirius.yaml" <<'YAML'
sirius:
  topology:
    num_gpus: 1
  memory:
    gpu:
      usage_limit_fraction: 0.5
    host:
      capacity_bytes: 17179869184
  executor:
    pipeline:
      num_threads: 4
YAML

pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

run_size() {
    local bytes=$1 tag
    tag="a2a$((bytes / 1000000))mb"
    pids=()
    for i in 0 1; do
        local me=$((PORT_BASE + i * 10)) peer=$((PORT_BASE + (1 - i) * 10))
        CUDA_VISIBLE_DEVICES=$i RUST_LOG="${RUST_LOG:-sirius_starrocks_cn=info}" \
            "$CN_BIN" \
            --fe-host 127.0.0.1 \
            --advertise-host 127.0.0.1 \
            --heartbeat-port "$me" \
            --thrift-port $((me + 1)) \
            --brpc-port $((me + 2)) \
            --http-port $((me + 3)) \
            --starlet-port $((me + 4)) \
            --sirius-config "$OUT/sirius.yaml" \
            --bench-a2a-bytes "$bytes" \
            --bench-peer "127.0.0.1:$((peer + 2))" \
            --bench-chunk-bytes "$CHUNK_BYTES" \
            --bench-rounds "$ROUNDS" \
            </dev/null >"$OUT/$tag.cn$i.log" 2>&1 &
        pids+=("$!")
    done
    local rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
    pids=()
    local line
    line=$(grep -h '^bench a2a' "$OUT/$tag.cn0.log" | head -1 || true)
    echo "$tag rc=$rc ${line:-no result (see $OUT/$tag.cn0.log)}"
    [[ $rc -eq 0 && -n $line ]]
}

status=0
for bytes in $SIZES; do
    run_size "$bytes" || status=1
done
exit "$status"
