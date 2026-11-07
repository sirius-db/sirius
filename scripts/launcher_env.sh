# Defaults for every Sirius launcher. Source it (under set -euo pipefail too). Each value you
# export first wins.
# On a host with GPU rail NICs (sirius_net.sh: sirius_rail_devices), with neither UCX_TLS nor
# UCX_NET_DEVICES set, or UCX_NET_DEVICES=rails: every UCX transport, on those NICs only. UCX's
# notifications between nodes then go over RC, which an exchange over the NVLink fabric needs at
# full speed (4 GPU pairs on two GB200 nodes, both ways: 5.9 TB/s against 5.2 with tcp on bond0).
# Never add bond0 to them: UCX then tries to reach a peer's bond0 address from a rail NIC and
# hangs.
source "${BASH_SOURCE[0]%/*}/sirius_net.sh"
if [[ -z ${UCX_TLS:-} && ${UCX_NET_DEVICES:-rails} == rails ]] && _rails=$(sirius_rail_devices); then
  export UCX_TLS=all UCX_NET_DEVICES=$_rails
elif [[ ${UCX_NET_DEVICES:-} == rails ]]; then
  echo "UCX_NET_DEVICES=rails needs UCX_TLS unset and GPU rail NICs on this host" >&2
  return 1
fi
unset _rails
# Otherwise the transports tested on one host and on two. Left to itself, UCX also picks the
# host's InfiniBand ports, and on multi-rail GB200 hosts the ranks then fail to connect ("Endpoint
# timeout").
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
# DuckDB changes its output when an AI coding tool runs it; the ranks and the scripts parse it.
export DUCKDB_AGENT_MODE=${DUCKDB_AGENT_MODE:-0}
