#!/usr/bin/env bash
# Runs one Sirius rank for a distributed gateway, and starts it again whenever it exits: when the
# gateway restarts its ranks (sirius_rank_exit), or after a crash. Start one per GPU on each node,
# outside the gateway (systemd, srun, tmux):
#   pixi run scripts/sirius_rank_node.sh [--config <file>] <gpu> <quack port> <exchange port> \
#     [listen host=$(hostname -f)] [dir=/tmp/sirius_rank_<quack port>]
# The config file (see sirius_rank_node.conf.example) is sourced and may set every argument, as
# gpu, quack_port, exchange_port, listen_host, dir; arguments given here win over it. It may also
# set:
#   ucx_tls, ucx_net_devices  UCX_TLS, UCX_NET_DEVICES, unless exported
#   priority_networks  CIDRs, e.g. 10.87.140.0/24; this host's first address in them is the listen
#                      host, and its interface UCX_NET_DEVICES, unless those are set
#   gateway, gateway_token, gateway_fingerprint  the gateway's Quack address, client token and TLS
#                      fingerprint (sirius_gateway.sh writes them to <its log dir>/gateway.conf)
# Once the rank listens, the script writes the sirius_gateway_add_rank call that registers it to
# <dir>/add_rank.sql. With a gateway, it runs that call itself after every start of the rank,
# retrying until it succeeds; the gateway must SET sirius_gateway_allow_remote_join = true.
# Without one, it prints the call, for the gateway's console.
# SIRIUS_RANK_TOKEN sets the rank's token; otherwise a random one is kept in <dir>/token. On a
# listen host other than this machine's loopback, Quack serves TLS with a key pair kept in
# <dir>/tls, whose fingerprint the gateway pins.
# With every rank on one host, set UCX_TLS=cuda_copy,cuda_ipc,tcp,self, as distributed_run_sql.sh
# does: left to itself, UCX picks InfiniBand ports that GB200 hosts cannot connect through.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
if [ "${1:-}" = --config ]; then
  # shellcheck source=/dev/null
  source "$2"
  shift 2
fi
gpu=${1:-${gpu:-}}
port=${2:-${quack_port:-}}
exchange_port=${3:-${exchange_port:-}}
if [ -z "$gpu" ] || [ -z "$port" ] || [ -z "$exchange_port" ]; then
  echo "usage: $0 [--config <file>] <gpu> <quack port> <exchange port> [listen host] [dir]" >&2
  exit 2
fi
[ -n "${ucx_tls:-}" ] && export UCX_TLS=${UCX_TLS:-$ucx_tls}
[ -n "${ucx_net_devices:-}" ] && export UCX_NET_DEVICES=${UCX_NET_DEVICES:-$ucx_net_devices}
if [ -n "${priority_networks:-}" ]; then
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
export CUDA_VISIBLE_DEVICES=$gpu
mkdir -p -m 700 "$dir"

token=${SIRIUS_RANK_TOKEN:-}
if [ -z "$token" ]; then
  [ -s "$dir/token" ] || (umask 077 && head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$dir/token")
  token=$(cat "$dir/token")
fi
serve="token := '$token'"
pin=
case $host in
  localhost | 127.0.0.1 | ::1) ;;
  *)
    # One key pair for every restart, so the fingerprint the gateway pins stays valid.
    fingerprint=$(env -u AI_AGENT "$duckdb" -noheader -list \
      -c "SELECT fingerprint FROM quack_generate_keys(directory := '$dir/tls')")
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
register()
{
  local tls=
  [ -n "${gateway_fingerprint:-}" ] && tls=", SSL_FINGERPRINT '$gateway_fingerprint'"
  env -u AI_AGENT "$duckdb" -batch -c "SET GLOBAL http_timeout = 86400;
ATTACH 'quack:$gateway' AS sirius_gateway (TOKEN '${gateway_token:-}'$tls);
CONNECT sirius_gateway;
$add_rank"
}

# Once rank <pid> listens: registers it with the gateway, or prints the add_rank call. Gives up
# when the rank exits; the next start tries again.
announce()
{
  local rank=$1 first=$2 backoff=1
  until (exec 3<> "/dev/tcp/$host/$port") 2> /dev/null; do
    kill -0 "$rank" 2> /dev/null || return 0
    sleep 1
  done
  if [ -z "${gateway:-}" ]; then
    [ "$first" = 1 ] && echo "rank listening on $host:$port; register it on the gateway's console:" >&2 &&
      echo "$add_rank" >&2
    return 0
  fi
  until register >> "$dir/register.log" 2>&1; do
    kill -0 "$rank" 2> /dev/null || return 0
    echo "registering with the gateway at $gateway failed (see $dir/register.log); again in ${backoff}s" >&2
    sleep "$backoff"
    backoff=$((backoff < 30 ? backoff * 2 : 30))
  done
  echo "rank on $host:$port registered with the gateway at $gateway" >&2
}

# A rank reads its stdin until EOF; a FIFO opened read-write never reaches it.
[ -p "$dir/stdin" ] || mkfifo -m 600 "$dir/stdin"
child=
announcer=
trap '[ -n "$announcer" ] && kill "$announcer" 2> /dev/null; [ -n "$child" ] && kill "$child" 2> /dev/null; exit 0' TERM INT
backoff=1
first=1
while :; do
  "$duckdb" -unsigned -init "$dir/init.sql" <> "$dir/stdin" >> "$dir/rank.log" 2>&1 &
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
  echo "rank exited with $rc; starting it again in ${backoff}s" >&2
  sleep "$backoff"
  backoff=$((backoff < 30 ? backoff * 2 : 30))
done
