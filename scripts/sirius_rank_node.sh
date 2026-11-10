#!/usr/bin/env bash
# Runs one Sirius rank for a distributed gateway, and starts it again whenever it exits: when the
# gateway restarts its ranks (sirius_rank_exit), or after a crash. sirius_node.sh starts one per
# GPU; to run one yourself (systemd, srun, tmux):
#   pixi run scripts/sirius_rank_node.sh [--config <file>] [--cluster <name>] <gpu> <quack port> \
#     <exchange port> [listen host=$(hostname -f)] [dir=/tmp/sirius_rank_<quack port>]
# <gpu> "slurm" keeps the CUDA_VISIBLE_DEVICES the rank inherits. The config file (see
# sirius_rank_node.conf.example) is sourced and may set every argument, as gpu, quack_port,
# exchange_port, listen_host, dir; arguments given here win over it. It may also set:
#   nixl_prefix, ucx_tls, ucx_net_devices  NIXL_PREFIX, UCX_TLS, UCX_NET_DEVICES, unless exported
#                      (defaults: scripts/nixl_env.sh)
#   priority_networks  CIDRs, e.g. 10.87.140.0/24; this host's first address in them is the listen
#                      host, and its interface UCX_NET_DEVICES, unless those are set. Default: the
#                      gateway's networks
#   gateway_conf       the gateway's file (--cluster <name> means ~/.sirius/gateways/<name>/
#                      gateway.conf): its address, token, TLS fingerprint, id and networks. The
#                      script waits for it, and re-reads it to follow a restarted gateway
#   gateway, gateway_token, gateway_fingerprint  or these, for a node without that file
# Once the rank listens, the script writes the sirius_gateway_add_rank call that registers it to
# <dir>/add_rank.sql. With a gateway, it runs that call itself after every start of the rank,
# retrying until it succeeds, and again whenever gateway.conf names a new gateway_id; the gateway
# must SET sirius_gateway_allow_join = true. Without one, it prints the call, for the gateway's
# console.
# SIRIUS_RANK_LOG_ROOT: each rank's Sirius log goes to <root>/<short hostname>-<quack port>/
# (SIRIUS_LOG_DIR). Without it, ranks started from one directory share its log/ folder, and
# ranks of several nodes on a shared file system lose lines appending to one file.
# SIRIUS_RANK_TOKEN sets the rank's token; otherwise a random one is kept in <dir>/token. On a
# listen host other than this machine's loopback, Quack serves TLS with a key pair kept in
# <dir>/tls, whose fingerprint the gateway pins.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
while :; do
  case ${1:-} in
    --config)
      # shellcheck source=/dev/null
      source "$2"
      shift 2
      ;;
    --cluster)
      gateway_conf=$HOME/.sirius/gateways/$2/gateway.conf
      shift 2
      ;;
    *) break ;;
  esac
done
gpu=${1:-${gpu:-}}
port=${2:-${quack_port:-}}
exchange_port=${3:-${exchange_port:-}}
if [ -z "$gpu" ] || [ -z "$port" ] || [ -z "$exchange_port" ]; then
  echo "usage: $0 [--config <file>] [--cluster <name>] <gpu> <quack port> <exchange port> [listen host] [dir]" >&2
  exit 2
fi

# Its values win over the config's: a restarted gateway may have a new address or token.
reload_gateway_conf()
{
  [ -n "${gateway_conf:-}" ] && [ -r "$gateway_conf" ] || return 0
  unset gateway_fingerprint gateway_id networks
  # shellcheck source=/dev/null
  source "$gateway_conf"
}
if [ -n "${gateway_conf:-}" ]; then
  # The listen address comes from the gateway's networks, so wait for the gateway's file.
  [ -r "$gateway_conf" ] || echo "waiting for $gateway_conf: start the gateway (pixi run sirius-gateway)" >&2
  until [ -r "$gateway_conf" ]; do sleep 2; done
  reload_gateway_conf
fi

[ -n "${nixl_prefix:-}" ] && export NIXL_PREFIX=${NIXL_PREFIX:-$nixl_prefix}
[ -n "${ucx_tls:-}" ] && export UCX_TLS=${UCX_TLS:-$ucx_tls}
[ -n "${ucx_net_devices:-}" ] && export UCX_NET_DEVICES=${UCX_NET_DEVICES:-$ucx_net_devices}
# Before priority_networks names an interface: the GPU rail NICs win where the host has them.
source "$here/nixl_env.sh"
priority_networks=${priority_networks:-${networks:-}}
if [ -n "$priority_networks" ]; then
  source "$here/sirius_net.sh"
  if ! picked=$(sirius_pick_address "$priority_networks"); then
    echo "no address of this host is in priority_networks $priority_networks" >&2
    exit 1
  fi
  export UCX_NET_DEVICES=${UCX_NET_DEVICES:-${picked% *}}
  listen_host=${listen_host:-${picked#* }}
  echo "priority_networks: listening on ${picked#* }, UCX_NET_DEVICES=$UCX_NET_DEVICES" >&2
fi
host=${4:-${listen_host:-$(hostname -f)}}
dir=${5:-${dir:-/tmp/sirius_rank_$port}}
build="$here/../sirius-duckdb/build/release"
duckdb="$build/duckdb"
# Under Slurm the task's own binding is already in CUDA_VISIBLE_DEVICES.
[ "$gpu" = slurm ] || export CUDA_VISIBLE_DEVICES=$gpu
mkdir -p "$dir"
chmod 700 "$dir"

token=${SIRIUS_RANK_TOKEN:-}
if [ -z "$token" ]; then
  [ -s "$dir/token" ] || (umask 077 && head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$dir/token")
  token=$(cat "$dir/token")
fi
# It goes into SQL as it is.
[[ $token =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "bad rank token: letters, digits, _ and - only (SIRIUS_RANK_TOKEN or $dir/token)" >&2
  exit 2
}
serve="token := '$token'"
pin=
case $host in
  localhost | 127.* | ::1) ;;
  *)
    # One key pair for every restart, so the fingerprint the gateway pins stays valid.
    # -init /dev/null: output a ~/.duckdbrc prints (.timer on) would end up in the fingerprint.
    fingerprint=$("$duckdb" -no-agent -init /dev/null -noheader -list \
      -c "SELECT fingerprint FROM quack_generate_keys(directory := '$dir/tls')")
    [[ $fingerprint =~ ^[0-9A-F]{2}(:[0-9A-F]{2})+$ ]] || {
      echo "unexpected TLS fingerprint from quack_generate_keys: '$fingerprint'" >&2
      exit 1
    }
    serve+=", allow_other_hostname := true, ssl_cert_file := '$dir/tls/server.pem'"
    serve+=", ssl_key_file := '$dir/tls/private_key.pem'"
    pin=", fingerprint := '$fingerprint'"
    ;;
esac
(
  umask 077
  cat > "$dir/init.sql" <<SQL
LOAD '$build/extension/sirius/sirius.duckdb_extension';
CALL quack_serve('quack:$host:$port', $serve);
SQL
)
add_rank="CALL sirius_gateway_add_rank('$host:$port', exchange_port := $exchange_port, token := '$token'$pin);"
(umask 077 && echo "$add_rank" > "$dir/add_rank.sql")

# Runs the add_rank call on the gateway, as one of its clients. The call waits for the gateway's
# turn, which a start of the ranks holds for minutes, past Quack's TLS client's 30 s default.
# The SQL holds both tokens: it goes on stdin from a file in the 700 rank dir, not in argv, where
# ps would show it to every user for those minutes. The client runs in the background so that the
# announcer's TERM trap can stop it at once.
register_client=
register()
{
  [ -n "${gateway:-}" ] || {
    echo "no gateway in $gateway_conf yet"
    return 1
  }
  local tls=
  [ -n "${gateway_fingerprint:-}" ] && tls=", SSL_FINGERPRINT '$gateway_fingerprint'"
  (
    umask 077
    cat > "$dir/register.sql" <<SQL
SET GLOBAL http_timeout = 86400;
ATTACH 'quack:$gateway' AS sirius_gateway (TOKEN '${gateway_token:-}'$tls);
CONNECT sirius_gateway;
$add_rank
SQL
  )
  # -bail: stdin does not stop at the first error, and add_rank must not run without the ATTACH.
  "$duckdb" -no-agent -batch -bail -init /dev/null < "$dir/register.sql" &
  register_client=$!
  wait "$register_client"
}

# Once rank <pid> listens: registers it with the gateway, or prints the add_rank call. Gives up
# when the rank exits; the next start tries again.
announce()
{
  local rank=$1 first=$2 backoff=1 seen
  # Stops an in-flight registration with the announcer. Sleeps run in the background for the same
  # reason: bash runs the trap only once a foreground child returns.
  trap '[ -z "$register_client" ] || kill "$register_client" 2> /dev/null; exit 0' TERM
  until (exec 3<> "/dev/tcp/$host/$port") 2> /dev/null; do
    kill -0 "$rank" 2> /dev/null || return 0
    sleep 1
  done
  if [ -z "${gateway:-}" ] && [ -z "${gateway_conf:-}" ]; then
    [ "$first" = 1 ] && echo "rank listening on $host:$port; register it on the gateway's console:" >&2 &&
      echo "$add_rank" >&2
    return 0
  fi
  reload_gateway_conf
  until register >> "$dir/register.log" 2>&1; do
    kill -0 "$rank" 2> /dev/null || return 0
    echo "registering with the gateway at ${gateway:-?} failed (see $dir/register.log); again in ${backoff}s" >&2
    sleep "$backoff" &
    wait $!
    backoff=$((backoff < 30 ? backoff * 2 : 30))
    reload_gateway_conf
  done
  echo "rank on $host:$port registered with the gateway at $gateway" >&2
  [ -n "${gateway_conf:-}" ] || return 0
  # add_rank waits for the gateway's turn, behind every query: call it only for a new gateway.
  seen=${gateway_id:-}
  while kill -0 "$rank" 2> /dev/null; do
    sleep 10 &
    wait $!
    reload_gateway_conf
    [ "${gateway_id:-}" != "$seen" ] || continue
    if register >> "$dir/register.log" 2>&1; then
      seen=${gateway_id:-}
      echo "rank on $host:$port registered with the new gateway at $gateway" >&2
    else
      echo "registering with the new gateway at $gateway failed (see $dir/register.log); again in 10s" >&2
    fi
  done
}

# A rank reads its stdin until EOF; a FIFO opened read-write never reaches it.
[ -p "$dir/stdin" ] || mkfifo -m 600 "$dir/stdin"
child=
announcer=
# Stops the announcer (and its registration client) and the rank, and waits for the rank, so that
# its shared-memory mirrors can be removed.
stop()
{
  if [ -n "$announcer" ]; then
    kill "$announcer" 2> /dev/null || true
    wait "$announcer" 2> /dev/null || true
  fi
  if [ -n "$child" ]; then
    kill "$child" 2> /dev/null || true
    wait "$child" 2> /dev/null || true
    rm -f /dev/shm/sirius-exchange-"$child"-*
  fi
  exit 0
}
if [ -n "${SIRIUS_RANK_LOG_ROOT:-}" ]; then
  export SIRIUS_LOG_DIR=$SIRIUS_RANK_LOG_ROOT/$(hostname -s)-$port
  mkdir -p "$SIRIUS_LOG_DIR"
fi
# SIRIUS_GDS=1 (set it with scan_manager.backend: gds): give the rank a cuFile configuration
# that cannot fall back to POSIX reads, with its log in the rank's node-local folder.
if [ "${SIRIUS_GDS:-0}" = 1 ]; then
  CUFILE_ENV_PATH_JSON=$("$here/gds/cufile_json.sh" "$dir/cufile") || exit 1
  export CUFILE_ENV_PATH_JSON
fi
# SIRIUS_RANK_BIND=gpu keeps the rank's threads on the CPUs of its GPU's socket; it is the default
# with SIRIUS_GDS=1. Measured on Q6 with one rank: 13.9 s on the GPU's socket, 15.4 s on the other
# socket or unbound.
bind=${SIRIUS_RANK_BIND:-$([ "${SIRIUS_GDS:-0}" = 1 ] && echo gpu || echo none)}
if [ "$bind" = gpu ] && [ -z "${SIRIUS_RANK_WRAPPER:-}" ]; then
  visible=${CUDA_VISIBLE_DEVICES:-}
  bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${visible%%,*}" 2> /dev/null |
    head -n 1 | tr 'A-F' 'a-f' || true)
  cpus=$(cat "/sys/bus/pci/devices/${bus#0000}/local_cpulist" 2> /dev/null || true)
  if [ -n "$cpus" ]; then
    SIRIUS_RANK_WRAPPER="taskset -c $cpus"
    echo "rank on GPU $bus bound to CPUs $cpus" >> "$dir/rank.log"
  else
    echo "SIRIUS_RANK_BIND=gpu: cannot find the CPUs near GPU '${CUDA_VISIBLE_DEVICES:-}'; not binding" >&2
  fi
fi
trap stop TERM INT
backoff=1
first=1
while :; do
  # SIRIUS_RANK_WRAPPER: a command to start the rank under, for example "taskset -c 0-71".
  # shellcheck disable=SC2086 # the wrapper is a command line
  ${SIRIUS_RANK_WRAPPER:-} "$duckdb" -unsigned -init "$dir/init.sql" <> "$dir/stdin" >> "$dir/rank.log" 2>&1 &
  child=$!
  announce "$child" "$first" &
  announcer=$!
  first=0
  rc=0
  wait "$child" || rc=$?
  kill "$announcer" 2> /dev/null || true
  wait "$announcer" 2> /dev/null || true
  # A rank that crashed leaves its shared-memory mirrors behind.
  rm -f /dev/shm/sirius-exchange-"$child"-*
  if [ "$rc" -eq 0 ]; then
    backoff=1
    continue
  fi
  # 70: the GDS backend gave up on a cuFile read that never returned. A new rank would read from
  # the same filesystem that just stopped answering, so it stays down.
  if [ "$rc" -eq 70 ]; then
    echo "rank exited with 70 (GDS read stalled); not starting it again" >&2
    exit 70
  fi
  echo "rank exited with $rc; starting it again in ${backoff}s" >&2
  # In the background, so that the TERM trap runs at once, not after the sleep.
  sleep "$backoff" &
  wait $! || true
  backoff=$((backoff < 30 ? backoff * 2 : 30))
done
