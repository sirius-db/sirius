#!/usr/bin/env bash
# 8-GPU two-host launch. Same argv as cn-2host.sh.
# SCALE_FACTOR selects bench/gb200-8gpu/sf<N>/env.sh (1000, 3000, 10000).
#
#   SCALE_FACTOR=3000 ./configs/gb200-8gpu/launch.sh 10.87.140.53 10.87.140.53
#   SCALE_FACTOR=3000 ./configs/gb200-8gpu/launch.sh 10.87.140.44 10.87.140.53 --no-fe
set -euo pipefail

SR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$(cd "$SR_DIR/../.." && pwd)
cd "$SR_DIR"

export SCALE_FACTOR=${SCALE_FACTOR:-1000}
KNOBS=$REPO/bench/gb200-8gpu/sf${SCALE_FACTOR}/env.sh
[ -f "$KNOBS" ] || { echo "launch: no knobs at $KNOBS (SCALE_FACTOR=1000|3000|10000)" >&2; exit 1; }
# shellcheck disable=SC1090
. "$KNOBS"

export NUM_CNS=${NUM_CNS:-$(( ${NUM_CNS_PER_HOST:-4} * 2 ))}
# shellcheck source=engine-a.env
. configs/gb200-8gpu/engine-a.env

unset CUDA_VISIBLE_DEVICES
exec ./benchmarks/cn-2host.sh "$@"
