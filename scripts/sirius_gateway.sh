#!/usr/bin/env bash
# Starts the distributed gateway and N ranks, by default on this host with rank i on GPU i, then
# serves clients over Quack until stdin closes (Ctrl-D) or the gateway is killed; the ranks exit
# with it.
#   pixi run scripts/sirius_gateway.sh <ranks> [client port=9494] [log dir=/tmp/sirius_gateway]
# SIRIUS_GATEWAY_GPUS (e.g. 0,0) picks each rank's GPU. SIRIUS_GATEWAY_HOSTS (e.g.
# 10.0.0.1,10.0.0.2) gives each rank's exchange address; ranks on other hosts start over SSH with
# sirius_gateway_ssh_rank.sh; the gateway turns host staging off when ranks are on several hosts.
# Clients: ATTACH 'quack:<host>:<port>' AS sirius (TOKEN '<token>'); CONNECT sirius;
# The token is SIRIUS_GATEWAY_TOKEN, or a random one printed at start. The gateway listens on
# SIRIUS_GATEWAY_HOST (default 127.0.0.1), or on this host's first address in
# SIRIUS_GATEWAY_PRIORITY_NETWORKS (CIDRs, e.g. 10.87.140.0/24); on any other host than loopback
# Quack serves TLS with a key pair kept in <log dir>/tls, which clients pin with ssl_fingerprint.
# The address, token and fingerprint go to <log dir>/gateway.conf, for sirius_rank_node.sh's
# config to source.
# With 0 ranks it starts none: ranks run by sirius_rank_node.sh register themselves (or run the
# sirius_gateway_add_rank lines it prints, on this console), and the first client statement
# starts them. SIRIUS_GATEWAY_MIN_RANKS (default 1) is how many it waits for;
# SIRIUS_GATEWAY_ALLOW_JOIN=0 refuses ranks registering themselves.
# Needs the gateway build (make -C gateway release).
set -euo pipefail
ranks=$1
port=${2:-9494}
logs=${3:-/tmp/sirius_gateway}
here=$(cd "$(dirname "$0")" && pwd)
source "$here/launcher_env.sh"
build="$here/../sirius-duckdb/build/release"
gateway="$here/../gateway/build/release/duckdb"
host=${SIRIUS_GATEWAY_HOST:-}
if [ -z "$host" ] && [ -n "${SIRIUS_GATEWAY_PRIORITY_NETWORKS:-}" ]; then
  source "$here/sirius_net.sh"
  picked=$(sirius_pick_address "$SIRIUS_GATEWAY_PRIORITY_NETWORKS") || {
    echo "no address of this host is in $SIRIUS_GATEWAY_PRIORITY_NETWORKS" >&2
    exit 1
  }
  host=${picked#* }
fi
host=${host:-127.0.0.1}
token=${SIRIUS_GATEWAY_TOKEN:-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')}
gpus=${SIRIUS_GATEWAY_GPUS:-}
hosts=${SIRIUS_GATEWAY_HOSTS:-}
# Every rank on this host: left to itself, UCX also picks the host's InfiniBand ports, and on
# multi-rail GB200 hosts the ranks then fail to connect ("Endpoint timeout"). Ranks on several
# hosts leave the choice to UCX.
if [ -z "$hosts" ]; then
  export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
fi
mkdir -p "$logs"
allow_join=${SIRIUS_GATEWAY_ALLOW_JOIN:-$([ "$ranks" -eq 0 ] && echo 1 || echo 0)}

# A key pair kept across restarts, so the fingerprint clients and ranks pin stays valid.
serve="allow_other_hostname := true, token := '$token'"
fingerprint=
case $host in
  localhost | 127.0.0.1 | ::1) ;;
  *)
    fingerprint=$(env -u AI_AGENT "$gateway" -noheader -list \
      -c "SELECT fingerprint FROM quack_generate_keys(directory := '$logs/tls')")
    serve+=", ssl_cert_file := '$logs/tls/server.pem', ssl_key_file := '$logs/tls/private_key.pem'"
    ;;
esac
(
  umask 077
  {
    echo "gateway=$host:$port"
    echo "gateway_token=$token"
    [ -z "$fingerprint" ] || echo "gateway_fingerprint=$fingerprint"
  } > "$logs/gateway.conf"
)

init="$logs/gateway.init.sql"
start=
if [ "$ranks" -gt 0 ]; then
  start="FROM sirius_gateway_start($ranks, gpus := '$gpus', hosts := '$hosts');"
fi
cat > "$init" <<SQL
SET GLOBAL sirius_gateway_duckdb_binary = '$build/duckdb';
SET GLOBAL sirius_gateway_sirius_extension = '$build/extension/sirius/sirius.duckdb_extension';
SET GLOBAL sirius_gateway_rank_launcher = '$here/sirius_gateway_ssh_rank.sh';
SET GLOBAL sirius_gateway_log_dir = '$logs';
SET GLOBAL sirius_gateway_min_ranks = ${SIRIUS_GATEWAY_MIN_RANKS:-1};
SET GLOBAL sirius_gateway_allow_remote_join = $([ "$allow_join" = 1 ] && echo true || echo false);
$start
SET GLOBAL quack_authorization_function = 'sirius_gateway_route';
CALL quack_serve('quack:$host:$port', $serve);
SQL
echo "gateway: quack:$host:$port, token $token${fingerprint:+, fingerprint $fingerprint}, logs in $logs" >&2
echo "ranks' node config: source $logs/gateway.conf" >&2
exec "$gateway" -init "$init"
