#!/usr/bin/env bash
# Starts a distributed gateway: one SQL endpoint, over Quack, for ranks that register with it
# (sirius_node.sh) or that it starts on this host. See docs/super-sirius/distributed-gateway.md.
set -euo pipefail

usage()
{
  cat <<'USAGE'
usage: sirius_gateway.sh [--serve] [ranks=0] [client port=9494] [log dir]

With 0 ranks (the default), ranks run by sirius_node.sh on any host register themselves, and the
first client statement waits for them. With N > 0 it starts N ranks on this host, rank i on GPU i.

  --serve     no console: serve until killed (for Slurm and services). Otherwise the gateway serves
              until its console's stdin closes (Ctrl-D).
  log dir     rank init files and logs; default /tmp/sirius_gateway_<user>_<cluster>

Environment:
  SIRIUS_CLUSTER                    the gateway's name (default: default). Its state is in
                                    ~/.sirius/gateways/<name>: gateway.conf (address, token,
                                    TLS fingerprint, for nodes and clients), token, tls/, the
                                    DDL log. One gateway per name runs at a time.
  SIRIUS_GATEWAY_HOST               the address to listen on
  SIRIUS_GATEWAY_PRIORITY_NETWORKS  or this host's first address in these CIDRs (e.g.
                                    10.87.140.0/24). Otherwise: with 0 ranks the address of the
                                    default route's interface, else 127.0.0.1. Any address
                                    other than loopback serves TLS.
  SIRIUS_GATEWAY_PORT               the client port when no argument gives it (default 9494)
  SIRIUS_GATEWAY_TOKEN              the client token (default: one made once and kept)
  SIRIUS_GATEWAY_MIN_RANKS          ranks the first statement waits for (default 1)
  SIRIUS_GATEWAY_JOIN_WAIT          sets sirius_gateway_join_wait
  SIRIUS_GATEWAY_STATEMENT_TIMEOUT  sets sirius_gateway_statement_timeout (default 3600 s; 0 is
                                    none): a client cannot set it
  SIRIUS_GATEWAY_ALLOW_JOIN         1 lets ranks register themselves (default: 1 with 0 ranks)
  SIRIUS_GATEWAY_GPUS               with N ranks, each rank's GPU (e.g. 0,0)
  SIRIUS_GATEWAY_EXCHANGE_MNNVL, SIRIUS_GATEWAY_EXCHANGE_FRAME_BYTES,
  SIRIUS_GATEWAY_EXCHANGE_SEND_FRAMES, SIRIUS_GATEWAY_EXCHANGE_RECV_FRAMES_PER_PEER
                                    set sirius_gateway_exchange_*, which every rank takes

Needs the gateway build (make -C gateway release) and a Sirius build with NIXL.
USAGE
}

serve_only=0
args=()
for arg; do
  case $arg in
    -h | --help) usage && exit 0 ;;
    --serve) serve_only=1 ;;
    -*) usage >&2 && exit 2 ;;
    *) args+=("$arg") ;;
  esac
done
# Defaults from the distributed: section of sirius.yaml; exported variables win. Captured first:
# eval would ignore the script's failure and run with half the file applied.
defaults=$(python3 "$(dirname "$0")/sirius_distributed_config.py") || exit 1
eval "$defaults"
ranks=${args[0]:-0}
port=${args[1]:-${SIRIUS_GATEWAY_PORT:-9494}}
cluster=${SIRIUS_CLUSTER:-default}
logs=${args[2]:-/tmp/sirius_gateway_${USER:-$(id -un)}_$cluster}
if ! [[ $ranks =~ ^[0-9]+$ && $port =~ ^[0-9]+$ && $cluster =~ ^[A-Za-z0-9._-]+$ ]]; then
  usage >&2
  exit 2
fi
# These go into the gateway's init SQL as they are.
if ! [[ ${SIRIUS_GATEWAY_MIN_RANKS:-1} =~ ^[0-9]+$ && ${SIRIUS_GATEWAY_JOIN_WAIT:-0} =~ ^[0-9]+$ &&
  ${SIRIUS_GATEWAY_STATEMENT_TIMEOUT:-0} =~ ^[0-9]+$ && ${SIRIUS_GATEWAY_GPUS:-} =~ ^[0-9,]*$ ]]; then
  echo "SIRIUS_GATEWAY_MIN_RANKS, SIRIUS_GATEWAY_JOIN_WAIT and SIRIUS_GATEWAY_STATEMENT_TIMEOUT take a number, SIRIUS_GATEWAY_GPUS a list such as 0,1" >&2
  exit 2
fi

if ! [[ ${SIRIUS_GATEWAY_EXCHANGE_MNNVL:-auto} =~ ^(auto|require|off)$ &&
  ${SIRIUS_GATEWAY_EXCHANGE_FRAME_BYTES:-16777216} =~ ^[0-9]+$ &&
  ${SIRIUS_GATEWAY_EXCHANGE_SEND_FRAMES:-32} =~ ^[0-9]+$ &&
  ${SIRIUS_GATEWAY_EXCHANGE_RECV_FRAMES_PER_PEER:-16} =~ ^[0-9]+$ ]]; then
  echo "SIRIUS_GATEWAY_EXCHANGE_MNNVL takes auto, require or off; _FRAME_BYTES, _SEND_FRAMES and _RECV_FRAMES_PER_PEER a number" >&2
  exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
source "$here/nixl_env.sh"
source "$here/sirius_net.sh"
build="$here/../sirius-duckdb/build/release"
gateway="$here/../gateway/build/release/duckdb"
[ -x "$gateway" ] || {
  echo "no gateway build at $gateway: run make -C gateway release" >&2
  exit 1
}

# Everything here holds the token or the rank logs.
umask 077
state=$HOME/.sirius/gateways/$cluster
mkdir -p "$state" "$logs"
chmod 700 "$state" "$logs"
# The lock is fd 9, which the gateway inherits through exec: it holds the lock for its whole life.
exec 9> "$state/lock"
flock -n 9 || {
  echo "gateway $cluster already runs (lock $state/lock); set SIRIUS_CLUSTER for another" >&2
  exit 1
}

host=${SIRIUS_GATEWAY_HOST:-}
if [ -z "$host" ] && [ -n "${SIRIUS_GATEWAY_PRIORITY_NETWORKS:-}" ]; then
  picked=$(sirius_pick_address "$SIRIUS_GATEWAY_PRIORITY_NETWORKS") || {
    echo "no address of this host is in $SIRIUS_GATEWAY_PRIORITY_NETWORKS" >&2
    exit 1
  }
  host=${picked#* }
fi
# Ranks that register come from other hosts, so they need a routable address.
if [ -z "$host" ] && [ "$ranks" -eq 0 ] && picked=$(sirius_default_address); then
  host=${picked#* }
fi
host=${host:-127.0.0.1}
# The host and token go into SQL and gateway.conf: refuse anything but a plain address or name.
[[ $host =~ ^[A-Za-z0-9.:-]+$ ]] || {
  echo "bad gateway host '$host': an IP address or host name" >&2
  exit 2
}
# Nodes pick their own address in the gateway's subnet; empty if the host is a name.
networks=$(sirius_address_network "$host") || networks=

new_hex()
{
  head -c "$1" /dev/urandom | od -An -tx1 | tr -d ' \n'
}
# Kept across restarts, so nodes and clients holding it stay valid.
[ -s "$state/token" ] || new_hex 16 > "$state/token"
token=${SIRIUS_GATEWAY_TOKEN:-$(cat "$state/token")}
[[ $token =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "bad gateway token: letters, digits, _ and - only (SIRIUS_GATEWAY_TOKEN or $state/token)" >&2
  exit 2
}

serve="allow_other_hostname := true, token := '$token'"
fingerprint=
case $host in
  localhost | 127.* | ::1) ;;
  *)
    # A key pair kept across restarts, so the fingerprint clients and ranks pin stays valid.
    # -init /dev/null: output a ~/.duckdbrc prints (.timer on) would end up in the fingerprint.
    fingerprint=$("$gateway" -no-agent -init /dev/null -noheader -list \
      -c "SELECT fingerprint FROM quack_generate_keys(directory := '$state/tls')")
    [[ $fingerprint =~ ^[0-9A-F]{2}(:[0-9A-F]{2})+$ ]] || {
      echo "unexpected TLS fingerprint from quack_generate_keys: '$fingerprint'" >&2
      exit 1
    }
    serve+=", ssl_cert_file := '$state/tls/server.pem', ssl_key_file := '$state/tls/private_key.pem'"
    ;;
esac
# Replaced in one rename: nodes re-read it every 10 s. A new gateway_id tells them to register again.
# Nodes and clients source it: %q keeps every value one shell word.
{
  printf '%s=%q\n' gateway "$host:$port" gateway_token "$token" gateway_id "$(new_hex 8)"
  [ -z "$fingerprint" ] || printf '%s=%q\n' gateway_fingerprint "$fingerprint"
  [ -z "$networks" ] || printf '%s=%q\n' networks "$networks"
} > "$state/gateway.conf.tmp"
mv -f "$state/gateway.conf.tmp" "$state/gateway.conf"

allow_join=${SIRIUS_GATEWAY_ALLOW_JOIN:-$([ "$ranks" -eq 0 ] && echo 1 || echo 0)}
init="$logs/gateway.init.sql"
{
  echo "SET GLOBAL sirius_gateway_duckdb_binary = '$build/duckdb';"
  echo "SET GLOBAL sirius_gateway_sirius_extension = '$build/extension/sirius/sirius.duckdb_extension';"
  echo "SET GLOBAL sirius_gateway_log_dir = '$logs';"
  echo "SET GLOBAL sirius_gateway_state_dir = '$state';"
  echo "SET GLOBAL sirius_gateway_min_ranks = ${SIRIUS_GATEWAY_MIN_RANKS:-1};"
  echo "SET GLOBAL sirius_gateway_exchange_mnnvl = '${SIRIUS_GATEWAY_EXCHANGE_MNNVL:-auto}';"
  echo "SET GLOBAL sirius_gateway_exchange_frame_bytes = ${SIRIUS_GATEWAY_EXCHANGE_FRAME_BYTES:-16777216};"
  echo "SET GLOBAL sirius_gateway_exchange_send_frames = ${SIRIUS_GATEWAY_EXCHANGE_SEND_FRAMES:-32};"
  echo "SET GLOBAL sirius_gateway_exchange_recv_frames_per_peer = ${SIRIUS_GATEWAY_EXCHANGE_RECV_FRAMES_PER_PEER:-16};"
  echo "SET GLOBAL sirius_gateway_allow_join = $([ "$allow_join" = 1 ] && echo true || echo false);"
  [ -z "${SIRIUS_GATEWAY_JOIN_WAIT:-}" ] || echo "SET GLOBAL sirius_gateway_join_wait = $SIRIUS_GATEWAY_JOIN_WAIT;"
  [ -z "${SIRIUS_GATEWAY_STATEMENT_TIMEOUT:-}" ] ||
    echo "SET GLOBAL sirius_gateway_statement_timeout = $SIRIUS_GATEWAY_STATEMENT_TIMEOUT;"
  # replay_saved: a restarted gateway replays the DDL saved in the state dir, instead of
  # deleting it as an explicit start from the console does.
  [ "$ranks" -eq 0 ] ||
    echo "FROM sirius_gateway_start($ranks, gpus := '${SIRIUS_GATEWAY_GPUS:-}', replay_saved := true);"
  echo "SET GLOBAL quack_authorization_function = 'sirius_gateway_route';"
  # Its result holds the token: print only the address.
  echo "SELECT listen_uri FROM quack_serve('quack:$host:$port', $serve);"
} > "$init"

opt=
[ "$cluster" = default ] || opt=" --cluster $cluster"
echo "gateway $cluster: quack:$host:$port${fingerprint:+ (TLS)}; logs in $logs" >&2
if [ "$ranks" -eq 0 ]; then
  echo "next: on each GPU node, pixi run sirius-node$opt; then pixi run sirius-client$opt" >&2
else
  echo "next: pixi run sirius-client$opt" >&2
fi
if [ "$serve_only" = 1 ]; then
  # A FIFO opened read-write never reaches EOF, so the gateway serves until it is killed.
  [ -p "$logs/serve.stdin" ] || mkfifo "$logs/serve.stdin"
  exec "$gateway" -init "$init" <> "$logs/serve.stdin"
fi
exec "$gateway" -init "$init"
