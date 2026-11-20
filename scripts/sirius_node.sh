#!/usr/bin/env bash
# Starts one Sirius rank per GPU on this node, each under sirius_rank_node.sh, which restarts it
# and registers it with the gateway. See docs/super-sirius/distributed-gateway.md.
set -euo pipefail

usage()
{
  cat <<'USAGE'
usage: sirius_node.sh [--cluster NAME] [--gpus all|0,1,...|slurm] [--quack-port 9500]
                      [--exchange-port 27000] [--networks CIDRS]
                      [--gateway HOST:PORT --token-file FILE [--fingerprint F]]
                      [--check [--data PATH]]

Runs a rank on each GPU until Ctrl-C or SIGTERM. Rank i listens on <quack port> + i and
exchanges on <exchange port> + i. The gateway's address, token, TLS fingerprint and subnet come
from ~/.sirius/gateways/<cluster>/gateway.conf, which sirius_gateway.sh writes; the ranks wait for
it, and register again when a new gateway starts.

  --cluster NAME    the gateway's name (default: $SIRIUS_CLUSTER, else default)
  --gpus            all GPUs (default), a list, or slurm: one rank on the GPU Slurm gave the task
  --networks CIDRs  pick this node's address in these, instead of the gateway's subnet
  --gateway HOST:PORT --token-file FILE [--fingerprint F]
                    for a node that does not share gateway.conf's directory. FILE holds the
                    gateway's token (gateway_token in its gateway.conf); or export
                    SIRIUS_GATEWAY_TOKEN. Never on the command line, where ps shows it to all.
  --check           check this node and exit: the Sirius build, GPUs, the gateway
  --data PATH       with --check: a dataset path every rank reads
USAGE
}

# Defaults from the distributed: section of sirius.yaml; exported variables win. Captured first:
# eval would ignore the script's failure and run with half the file applied.
defaults=$(python3 "$(dirname "$0")/sirius_distributed_config.py") || exit 1
eval "$defaults"
cluster=${SIRIUS_CLUSTER:-default}
gpus=${SIRIUS_NODE_GPUS:-all}
quack_base=${SIRIUS_NODE_QUACK_PORT:-9500}
exchange_base=${SIRIUS_NODE_EXCHANGE_PORT:-27000}
networks=${SIRIUS_NODE_NETWORKS:-}
gateway=
token=${SIRIUS_GATEWAY_TOKEN:-}
token_file=
fingerprint=
check=0
data=
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage && exit 0 ;;
    --check) check=1 && shift && continue ;;
  esac
  [ $# -ge 2 ] || {
    usage >&2
    exit 2
  }
  case $1 in
    --cluster) cluster=$2 ;;
    --gpus) gpus=$2 ;;
    --quack-port) quack_base=$2 ;;
    --exchange-port) exchange_base=$2 ;;
    --networks) networks=$2 ;;
    --gateway) gateway=$2 ;;
    --token-file) token_file=$2 ;;
    --token)
      echo "--token would show the token to every user in ps: use --token-file FILE or SIRIUS_GATEWAY_TOKEN" >&2
      exit 2
      ;;
    --fingerprint) fingerprint=$2 ;;
    --data) data=$2 ;;
    *) usage >&2 && exit 2 ;;
  esac
  shift 2
done
if ! [[ $quack_base =~ ^[0-9]+$ && $exchange_base =~ ^[0-9]+$ && $cluster =~ ^[A-Za-z0-9._-]+$ ]]; then
  usage >&2
  exit 2
fi
if [ -n "$token_file" ]; then
  [ -r "$token_file" ] || {
    echo "cannot read --token-file $token_file" >&2
    exit 2
  }
  token=$(< "$token_file")
fi
if [ -n "$gateway" ] && [ -z "$token" ]; then
  echo "--gateway needs --token-file FILE or SIRIUS_GATEWAY_TOKEN: the gateway's token (gateway_token in its gateway.conf)" >&2
  exit 2
fi
# These reach the ranks' SQL as they are.
if [ -n "$gateway" ] && ! [[ $gateway =~ ^[A-Za-z0-9.:-]+:[0-9]+$ && $token =~ ^[A-Za-z0-9_-]+$ &&
  $fingerprint =~ ^([0-9A-F]{2}(:[0-9A-F]{2})+)?$ ]]; then
  echo "bad --gateway HOST:PORT, token (letters, digits, _ and -) or --fingerprint (AA:BB:...)" >&2
  exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
source "$here/launcher_env.sh"
source "$here/sirius_net.sh"
build="$here/../sirius-duckdb/build/release"
extension="$build/extension/sirius/sirius.duckdb_extension"
conf=$HOME/.sirius/gateways/$cluster/gateway.conf

case $gpus in
  slurm) ranks=(slurm) ;;
  all)
    count=$(nvidia-smi -L 2> /dev/null | grep -c '^GPU' || true)
    ranks=()
    for ((i = 0; i < count; i++)); do ranks+=("$i"); done
    ;;
  *) IFS=, read -r -a ranks <<< "$gpus" ;;
esac

problems=0
pass()
{
  echo "ok    $1"
}
fail()
{
  echo "FAIL  $1" >&2
  echo "      fix: $2" >&2
  problems=$((problems + 1))
}

check_build()
{
  if [ ! -f "$extension" ] || [ ! -x "$build/duckdb" ]; then
    fail "no Sirius build at $build" "pixi run build-distributed"
    return
  fi
  pass "Sirius build: $build"
}

check_gpus()
{
  local count gpu
  count=$(nvidia-smi -L 2> /dev/null | grep -c '^GPU' || true)
  if [ "$count" -eq 0 ]; then
    fail "no GPU visible to nvidia-smi -L" "run on a GPU node (under Slurm, ask for --gpus-per-task)"
    return
  fi
  if [ "${#ranks[@]}" -eq 0 ]; then
    fail "no GPU in --gpus $gpus" "--gpus all, or a list such as 0,1"
    return
  fi
  if [ "$gpus" = slurm ]; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
      fail "--gpus slurm, but Slurm gave this task no GPU" "srun --gpus-per-task=1"
      return
    fi
  else
    for gpu in "${ranks[@]}"; do
      if ! [[ $gpu =~ ^[0-9]+$ ]] || [ "$gpu" -ge "$count" ]; then
        fail "no GPU $gpu: this node has $count" "--gpus all, or a list of GPUs from 0 to $((count - 1))"
        return
      fi
    done
  fi
  pass "${#ranks[@]} rank(s) on GPU(s) ${ranks[*]} ($count on this node)"
}

check_gateway()
{
  # gateway.conf may set gateway and networks; --networks wins over its networks.
  local gateway=$gateway nets=$networks networks=
  if [ -z "$gateway" ]; then
    if [ ! -r "$conf" ]; then
      fail "no $conf" "start the gateway (pixi run sirius-gateway) on a host that shares \$HOME, or pass --gateway and --token-file"
      return
    fi
    # shellcheck source=/dev/null
    source "$conf"
    pass "gateway $cluster in $conf: $gateway"
  fi
  nets=${nets:-$networks}
  # shellcheck disable=SC2016 # $0 and $1 expand in the inner bash
  if timeout 3 bash -c 'exec 3<> "/dev/tcp/$0/$1"' "${gateway%:*}" "${gateway##*:}" 2> /dev/null; then
    pass "gateway port ${gateway##*:} reachable on ${gateway%:*}"
  else
    fail "cannot reach the gateway at $gateway" \
      "start it (pixi run sirius-gateway), or open TCP port ${gateway##*:} between this node and it"
  fi
  if [ -n "$nets" ]; then
    if picked=$(sirius_pick_address "$nets"); then
      pass "this node's address: ${picked#* } on ${picked% *}, in $nets"
    else
      fail "no address of this node is in $nets" "--networks <CIDR of this node's address that the gateway's other nodes reach>"
    fi
  fi
}

check_build
check_gpus
if [ "$check" = 1 ]; then
  check_gateway
  if [ -n "$data" ]; then
    if [ -e "$data" ]; then
      pass "data: $data"
    else
      fail "no $data on this node" "mount the dataset at the same path on every node"
    fi
  fi
  [ "$problems" -eq 0 ] && echo "node ready for gateway $cluster"
  exit $((problems > 0))
fi
[ "$problems" -eq 0 ] || exit 1

# Per-rank configs and state; they hold the ranks' tokens.
umask 077
root=/tmp/sirius_node_${USER:-$(id -un)}_$cluster
mkdir -p "$root"
chmod 700 "$root"
pids=()
for i in "${!ranks[@]}"; do
  quack_port=$((quack_base + i))
  rank_conf=$root/rank_$quack_port.conf
  {
    printf 'gpu=%q\nquack_port=%q\nexchange_port=%q\ndir=%q\n' \
      "${ranks[$i]}" "$quack_port" "$((exchange_base + i))" "$root/rank_$quack_port"
    if [ -n "$gateway" ]; then
      printf 'gateway=%q\ngateway_token=%q\n' "$gateway" "$token"
      [ -z "$fingerprint" ] || printf 'gateway_fingerprint=%q\n' "$fingerprint"
    else
      printf 'gateway_conf=%q\n' "$conf"
    fi
    [ -z "$networks" ] || printf 'priority_networks=%q\n' "$networks"
  } > "$rank_conf"
  "$here/sirius_rank_node.sh" --config "$rank_conf" > >(sed -u "s/^/[rank $quack_port] /" >&2) 2>&1 &
  pids+=($!)
done

stop()
{
  trap - INT TERM
  kill "${pids[@]}" 2> /dev/null || true
  wait "${pids[@]}" 2> /dev/null || true
}
trap 'stop; exit 0' INT TERM
echo "node: ${#ranks[@]} rank(s) for gateway $cluster, on GPU(s) ${ranks[*]}; logs in $root/rank_<port>/rank.log; Ctrl-C stops them" >&2
# A rank's script only returns on an error it cannot retry: stop the others too.
rc=0
wait -n "${pids[@]}" || rc=$?
echo "a rank's script exited with $rc; stopping this node's ranks" >&2
stop
exit 1
