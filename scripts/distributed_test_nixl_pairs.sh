#!/usr/bin/env bash
# Runs the exchange's [nixl_pairs] test on two nodes of a Slurm partition: a raw NIXL bandwidth
# benchmark's phases, through the Sirius exchange. Rank r runs on GPU r % N of node r / N and
# pairs with rank (r + N) % 2N, N ranks per node. Writes each rank's log, the UCX transport
# choices and summary.md (GB/s per phase next to raw NIXL's) to OUT.
#   SLURM_CONF=... scripts/distributed_test_nixl_pairs.sh <node1>,<node2> [out dir]
# Environment:
#   RANKS_PER_NODE   ranks (GPUs) per node, default 4
#   PARTITION        Slurm partition, default nvidia-part
#   SECONDS_PER_PHASE  default 1.5
#   SIRIUS_EXCHANGE_MNNVL, SIRIUS_EXCHANGE_FRAME_BYTES, SIRIUS_EXCHANGE_SEND_FRAMES,
#   SIRIUS_EXCHANGE_RECV_FRAMES_PER_PEER   passed to every rank (defaults: require, 16 MiB, 32, 8)
#   UCX_TLS, UCX_NET_DEVICES  default: what a gateway's ranks use, scripts/nixl_env.sh's UCX_TLS
#                    (cuda_ipc carries the data, tcp the notifications) on bond0. With UCX_TLS=all,
#                    give the rail NICs (mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1): UCX otherwise tries
#                    to reach the peer's bond0 address from a rail NIC and hangs.
# Needs a build configured with -DSIRIUS_WITH_NIXL=ON. Never run it on nodes you use outside Slurm.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)

if [[ ${1:-} == --rank ]]; then
  # One task of the job step: this rank's environment, then the test.
  out=$2
  rank=$SLURM_PROCID
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export SIRIUS_EXCHANGE_RANK=$rank
  # Sirius writes its log to ./log: one folder per rank.
  mkdir -p "$out/rank$rank" && cd "$out/rank$rank"
  source "$here/nixl_env.sh"
  exec timeout -s KILL "${RANK_TIMEOUT:-900}" "$root/build/release/test/cpp/sirius_unittest" \
    "[nixl_pairs]" > "$out/rank$rank.log" 2>&1
fi

nodes=${1:?usage: distributed_test_nixl_pairs.sh <node1>,<node2> [out dir]}
out=${2:-/scratch/prestouser/$USER/mnnvl/pairs-$(date +%Y%m%d-%H%M%S)}
per_node=${RANKS_PER_NODE:-4}
IFS=, read -r -a hosts <<< "$nodes"
if ((${#hosts[@]} != 2)); then echo "give two nodes, not '$nodes'" >&2; exit 2; fi
peers=""
r=0
for host in "${hosts[@]}"; do
  ip=$(getent hosts "$host" | awk '{print $1; exit}')
  [[ -n $ip ]] || { echo "cannot resolve $host" >&2; exit 2; }
  for ((i = 0; i < per_node; i++, r++)); do peers+="${peers:+,}$ip:$((27000 + r))"; done
done
mkdir -p "$out"
export SIRIUS_EXCHANGE_PEERS=$peers
export SIRIUS_EXCHANGE_PAIRS_SECONDS=${SECONDS_PER_PHASE:-1.5}
export SIRIUS_EXCHANGE_MNNVL=${SIRIUS_EXCHANGE_MNNVL:-require}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond0}
export UCX_PROTO_INFO=y
source "$here/nixl_env.sh"
{
  echo "nodes: $nodes, $per_node ranks per node, build $(git -C "$root" rev-parse --short HEAD)"
  env | grep -E '^(SIRIUS_EXCHANGE|UCX_)' | sort
} > "$out/config.txt"
echo "logs: $out"
srun -p "${PARTITION:-nvidia-part}" -w "$nodes" -N2 -n$((2 * per_node)) \
  --ntasks-per-node="$per_node" --distribution=block --gres=gpu:"$per_node" --time=20 \
  --immediate=600 --kill-on-bad-exit=1 "$here/distributed_test_nixl_pairs.sh" --rank "$out" ||
  echo "srun failed: $?" >> "$out/config.txt"
python3 "$here/nixl_pairs_summary.py" "$out" | tee "$out/summary.md"
