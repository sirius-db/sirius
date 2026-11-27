#!/usr/bin/env bash
# Runs the exchange's [nixl_pairs] test on two nodes of a Slurm partition: a raw NIXL bandwidth
# benchmark's phases, through the Sirius exchange. Rank r runs on GPU r % N of node r / N and
# pairs with rank (r + N) % 2N, N ranks per node. Writes each rank's log, the UCX transport
# choices and summary.md (GB/s per phase next to raw NIXL's) to OUT.
#   SLURM_CONF=... scripts/distributed_test_nixl_pairs.sh <node1>,<node2> [out dir]
# Environment:
#   RANKS_PER_NODE   ranks (GPUs) per node, default 4
#   TEST             the Catch2 tag, default [nixl_pairs]; [nixl_bandwidth] runs an all-to-all
#   PARTITION        Slurm partition, default nvidia-part
#   SECONDS_PER_PHASE  default 1.5
#   SIRIUS_EXCHANGE_MNNVL, SIRIUS_EXCHANGE_FRAME_BYTES, SIRIUS_EXCHANGE_SEND_FRAMES,
#   SIRIUS_EXCHANGE_RECV_FRAMES_PER_PEER   passed to every rank (defaults: require, 16 MiB, 32, 16)
#   SIRIUS_EXCHANGE_HOST_STAGING (auto), SIRIUS_EXCHANGE_PAIRS_GPU_GIB (40)   likewise; with
#                    [nixl_bandwidth] also SIRIUS_EXCHANGE_BW_BATCHES
#   SIRIUS_EXCHANGE_TCP_LANES   lanes per rank where lanes carry the writes (default: 16 shared
#                    by the ranks of a host, at least 2)
#   RANK_TIMEOUT     seconds before a rank is killed, default 900
#   UCX_TLS, UCX_NET_DEVICES  default: scripts/launcher_env.sh's, as a gateway's ranks: tcp for
#                    notifications on bond0. SIRIUS_UCX_RAILS=1: UCX_TLS=all on the node's GPU
#                    rail NICs.
# Never run it on nodes you use outside Slurm.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)

if [[ ${1:-} == --rank ]]; then
  # One task of the job step: this rank's environment, then the test.
  out=$2
  rank=$SLURM_PROCID
  # Slurm orders the nodes of -w itself: this rank must run on the host its peers entry names.
  mine=$(cut -d, -f$((rank + 1)) <<< "$SIRIUS_EXCHANGE_PEERS")
  [[ $(ip -o -4 addr show) == *" ${mine%:*}/"* ]] || {
    echo "rank $rank runs on $(hostname), not at ${mine%:*}: give the nodes in Slurm's order" >&2
    exit 1
  }
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export SIRIUS_EXCHANGE_RANK=$rank
  # Sirius writes its log to ./log: one folder per rank.
  mkdir -p "$out/rank$rank" && cd "$out/rank$rank"
  source "$here/launcher_env.sh"
  export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond0}
  echo "rank $rank on $(hostname): UCX_TLS=$UCX_TLS UCX_NET_DEVICES=$UCX_NET_DEVICES" >> "$out/ucx.txt"
  exec timeout -s KILL "${RANK_TIMEOUT:-900}" "$root/build/release/test/cpp/sirius_unittest" \
    "${TEST:-[nixl_pairs]}" > "$out/rank$rank.log" 2>&1
fi

nodes=${1:?usage: distributed_test_nixl_pairs.sh <node1>,<node2> [out dir]}
out=${2:-nixl-pairs-$(date +%Y%m%d-%H%M%S)}
per_node=${RANKS_PER_NODE:-4}
IFS=, read -r -a hosts <<< "$nodes"
if ((${#hosts[@]} != 2)); then echo "give two nodes, not '$nodes'" >&2; exit 2; fi
peers=""
r=0
for host in "${hosts[@]}"; do
  ip=$(getent hosts "$host" | awk '{print $1; exit}') || ip=
  [[ -n $ip ]] || { echo "cannot resolve $host" >&2; exit 2; }
  for ((i = 0; i < per_node; i++, r++)); do peers+="${peers:+,}$ip:$((27000 + r))"; done
done
mkdir -p "$out"
# Each rank changes into its own folder before it writes here.
out=$(cd "$out" && pwd)
export SIRIUS_EXCHANGE_PEERS=$peers
export SIRIUS_EXCHANGE_PAIRS_SECONDS=${SECONDS_PER_PHASE:-1.5}
export SIRIUS_EXCHANGE_MNNVL=${SIRIUS_EXCHANGE_MNNVL:-require}
export UCX_PROTO_INFO=y
{
  echo "nodes: $nodes, $per_node ranks per node, build $(git -C "$root" rev-parse --short HEAD)"
  env | grep -E '^(SIRIUS_EXCHANGE|UCX_)' | sort
} > "$out/config.txt"
echo "logs: $out"
srun -p "${PARTITION:-nvidia-part}" -w "$nodes" -N2 -n$((2 * per_node)) \
  --ntasks-per-node="$per_node" --distribution=block --gres=gpu:"$per_node" --time=20 \
  --immediate=600 --kill-on-bad-exit=1 "$here/distributed_test_nixl_pairs.sh" --rank "$out" ||
  status=$?
[[ ${status:-0} == 0 ]] || echo "srun failed: $status" >> "$out/config.txt"
if [[ ${TEST:-[nixl_pairs]} == *nixl_pairs* ]]; then
  python3 "$here/nixl_pairs_summary.py" "$out" | tee "$out/summary.md" || status=${status:-1}
else
  grep -h BANDWIDTH "$out"/rank*.log | tee "$out/summary.md" || status=${status:-1}
fi
exit "${status:-0}"
