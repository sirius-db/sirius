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
# SIRIUS_GATEWAY_HOST (default 127.0.0.1); on any other host Quack serves TLS with a self-signed
# certificate, which clients pin with ssl_fingerprint.
# With 0 ranks it starts none: register ranks run by sirius_rank_node.sh with the
# sirius_gateway_add_rank lines it prints, on this console, then CALL sirius_gateway_start(0).
# Needs the gateway build (make -C gateway release) and a Sirius build with -DSIRIUS_WITH_NIXL=ON.
set -euo pipefail
ranks=$1
port=${2:-9494}
logs=${3:-/tmp/sirius_gateway}
here=$(cd "$(dirname "$0")" && pwd)
source "$here/nixl_env.sh"
build="$here/../sirius-duckdb/build/release"
gateway="$here/../gateway/build/release/duckdb"
host=${SIRIUS_GATEWAY_HOST:-127.0.0.1}
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
$start
SET GLOBAL quack_authorization_function = 'sirius_gateway_route';
CALL quack_serve('quack:$host:$port', allow_other_hostname := true, token := '$token');
SQL
echo "gateway: quack:$host:$port, token $token, logs in $logs" >&2
exec "$gateway" -init "$init"
