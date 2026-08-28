#!/usr/bin/env bash
# Stop and bring up the 8-CN cluster on gcn-09 + gcn-18.
# SCALE_FACTOR selects bench/gb200-8gpu/sf<N>/env.sh.
#
#   SCALE_FACTOR=1000  ./configs/gb200-8gpu/relaunch.sh
#   SCALE_FACTOR=3000  ./configs/gb200-8gpu/relaunch.sh
#   SCALE_FACTOR=10000 ./configs/gb200-8gpu/relaunch.sh
#
# Sequential, no SSH:
#   [09]  SCALE_FACTOR=3000 ./configs/gb200-8gpu/relaunch.sh --local-only --stop-only
#   [18]  SCALE_FACTOR=3000 ./configs/gb200-8gpu/relaunch.sh --local-only
#   [09]  SCALE_FACTOR=3000 ./configs/gb200-8gpu/relaunch.sh --local-only --no-fe
set -euo pipefail

SR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$(cd "$SR/../.." && pwd)
REMOTE_HOST=${REMOTE_HOST:-presto-gb200-gcn-09}
FE_HOST=${FE_HOST:-10.87.140.53}
CN09_HOST=${CN09_HOST:-10.87.140.44}

export SCALE_FACTOR=${SCALE_FACTOR:-1000}
KNOBS=$REPO/bench/gb200-8gpu/sf${SCALE_FACTOR}/env.sh
[ -f "$KNOBS" ] || { echo "relaunch: no knobs at $KNOBS" >&2; exit 1; }
# shellcheck disable=SC1090
. "$KNOBS"

LOCAL_ONLY=0
STOP_ONLY=0
NO_FE=0
for arg in "$@"; do
  case $arg in
    --local-only) LOCAL_ONLY=1 ;;
    --stop-only)  STOP_ONLY=1 ;;
    --no-fe)      NO_FE=1 ;;
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

say() { printf 'relaunch: %s\n' "$*"; }

stop_here() {
  "$SR/benchmarks/stop-cn-2host.sh"
}

if [ "$LOCAL_ONLY" = 1 ]; then
  stop_here
  if [ "$STOP_ONLY" = 1 ]; then
    say "stopped $(hostname); --stop-only"
    exit 0
  fi
  # shellcheck disable=SC1091
  source /scratch/prestouser/aocsa/env.sh
  cd "$SR"
  unset CUDA_VISIBLE_DEVICES
  export SCALE_FACTOR
  if [ "$NO_FE" = 1 ]; then
    exec ./configs/gb200-8gpu/launch.sh "$CN09_HOST" "$FE_HOST" --no-fe
  fi
  META=/scratch/prestouser/aocsa/fe/meta
  mkdir -p "$META"
  rm -rf "$META"/*
  exec ./configs/gb200-8gpu/launch.sh "$FE_HOST" "$FE_HOST"
fi

host=$(hostname)
case $host in
  *gcn-18*) ;;
  *) echo "relaunch: run the SSH driver on gcn-18 (this host is $host). Use --local-only on 09." >&2
     exit 1 ;;
esac

say "SCALE_FACTOR=$SCALE_FACTOR GPU_MEM=$GPU_MEM STAGING=$STAGING HOST_MEM=$HOST_MEM"
say "stopping $REMOTE_HOST"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
  "cd $(printf %q "$SR") && ./benchmarks/stop-cn-2host.sh"
say "stopping $(hostname)"
stop_here

META=/scratch/prestouser/aocsa/fe/meta
mkdir -p "$META"
rm -rf "$META"/*
say "wiped FE meta $META"

# shellcheck disable=SC1091
source /scratch/prestouser/aocsa/env.sh
cd "$SR"
unset CUDA_VISIBLE_DEVICES
export SCALE_FACTOR

say "starting FE+4 CNs on 18 -> /tmp/gb200-8gpu-launch-18.log"
nohup env SCALE_FACTOR="$SCALE_FACTOR" GPU_MEM="$GPU_MEM" HOST_MEM="$HOST_MEM" \
  STAGING="$STAGING" SIRIUS_EXCHANGE_STAGING_BYTES="$SIRIUS_EXCHANGE_STAGING_BYTES" \
  PIPELINE_DOP="$PIPELINE_DOP" \
  SIRIUS_QUERY_WATCHDOG_SECS="$SIRIUS_QUERY_WATCHDOG_SECS" \
  SIRIUS_CN_RPC_TIMEOUT_SECS="$SIRIUS_CN_RPC_TIMEOUT_SECS" \
  SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS="$SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS" \
  ./configs/gb200-8gpu/launch.sh "$FE_HOST" "$FE_HOST" \
  >/tmp/gb200-8gpu-launch-18.log 2>&1 &
echo $! >/tmp/gb200-8gpu-launch-18.pid

MYSQL=${MYSQL:-$SR/.pixi/envs/default/bin/mysql}
say "waiting for 4 local CNs"
n=0
for _ in $(seq 1 90); do
  n=$("$MYSQL" -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES' 2>/dev/null \
      | awk -F'\t' '$9=="true"{c++} END{print c+0}') || n=0
  [ "$n" -ge 4 ] && break
  sleep 2
done
if [ "$n" -lt 4 ]; then
  say "FE did not reach 4 alive CNs"
  tail -20 /tmp/gb200-8gpu-launch-18.log || true
  exit 1
fi
say "18 has $n alive CNs"

say "starting 4 CNs on 09 -> /tmp/gb200-8gpu-launch-09.log"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" bash -s <<EOF
set -euo pipefail
source /scratch/prestouser/aocsa/env.sh
unset CUDA_VISIBLE_DEVICES
cd $(printf %q "$SR")
export SCALE_FACTOR=$(printf %q "$SCALE_FACTOR")
export GPU_MEM=$(printf %q "$GPU_MEM")
export HOST_MEM=$(printf %q "$HOST_MEM")
export STAGING=$(printf %q "$STAGING")
export SIRIUS_EXCHANGE_STAGING_BYTES=$(printf %q "$SIRIUS_EXCHANGE_STAGING_BYTES")
export PIPELINE_DOP=$(printf %q "$PIPELINE_DOP")
export SIRIUS_QUERY_WATCHDOG_SECS=$(printf %q "$SIRIUS_QUERY_WATCHDOG_SECS")
export SIRIUS_CN_RPC_TIMEOUT_SECS=$(printf %q "$SIRIUS_CN_RPC_TIMEOUT_SECS")
export SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=$(printf %q "$SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS")
nohup env SCALE_FACTOR=$SCALE_FACTOR GPU_MEM=$GPU_MEM HOST_MEM=$HOST_MEM \
  STAGING=$STAGING SIRIUS_EXCHANGE_STAGING_BYTES=$SIRIUS_EXCHANGE_STAGING_BYTES \
  PIPELINE_DOP=$PIPELINE_DOP \
  SIRIUS_QUERY_WATCHDOG_SECS=$SIRIUS_QUERY_WATCHDOG_SECS \
  SIRIUS_CN_RPC_TIMEOUT_SECS=$SIRIUS_CN_RPC_TIMEOUT_SECS \
  SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=$SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS \
  ./configs/gb200-8gpu/launch.sh \
  $(printf %q "$CN09_HOST") $(printf %q "$FE_HOST") --no-fe \\
  >/tmp/gb200-8gpu-launch-09.log 2>&1 &
echo \$! >/tmp/gb200-8gpu-launch-09.pid
echo started_09 pid=\$(cat /tmp/gb200-8gpu-launch-09.pid)
EOF

say "waiting for 8 Alive=true"
n=0
for _ in $(seq 1 90); do
  n=$("$MYSQL" -h127.0.0.1 -P9030 -uroot -N -e 'SHOW COMPUTE NODES' 2>/dev/null \
      | awk -F'\t' '$9=="true"{c++} END{print c+0}') || n=0
  [ "$n" -eq 8 ] && break
  sleep 2
done
if [ "$n" -ne 8 ]; then
  say "expected 8 alive, got $n"
  "$MYSQL" -h127.0.0.1 -P9030 -uroot --vertical -e 'SHOW COMPUTE NODES' || true
  exit 1
fi

"$MYSQL" -h127.0.0.1 -P9030 -uroot -e \
  "SET GLOBAL enable_pipeline_engine=true; SET GLOBAL pipeline_dop=${PIPELINE_DOP}; SET GLOBAL query_timeout=${FE_QUERY_TIMEOUT};"
say "8 CNs alive, pipeline_dop=$PIPELINE_DOP GPU_MEM=$GPU_MEM STAGING=$STAGING HOST_MEM=$HOST_MEM"
"$MYSQL" -h127.0.0.1 -P9030 -uroot --vertical -e 'SHOW COMPUTE NODES' \
  | awk '/^[[:space:]]*IP:/ {ip=$2} /^[[:space:]]*HeartbeatPort:/ {hp=$2} /^[[:space:]]*Alive:/ {print ip, hp, $2}'
