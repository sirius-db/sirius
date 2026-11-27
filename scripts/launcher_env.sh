# Defaults for every Sirius launcher. Source it (under set -euo pipefail too). Each value you
# export first wins.
# SIRIUS_UCX_RAILS=1, on a host with GPU rail NICs (sirius_net.sh: sirius_rail_devices): UCX_TLS
# defaults to every transport and UCX_NET_DEVICES to those NICs. UCX's notifications between nodes
# then go over RC, which an exchange over the NVLink fabric needs at full speed (4 GPU pairs on
# two GB200 nodes, both ways: 5.9 TB/s against 5.2 with tcp on bond0). Never add bond0 to them:
# UCX then tries to reach a peer's bond0 address from a rail NIC and hangs.
# The gateway's ranks then also write GPU memory over NIXL to every peer the NVLink fabric does
# not reach (host_staging never). auto reaches ranks on other hosts over lanes, which on these
# NICs are RDMA too, but of host copies: 33 GB/s per rank against 65 when every rank sends to
# every other (8 ranks on two GB200 nodes, fabric off).
source "$(dirname "${BASH_SOURCE[0]}")/sirius_net.sh"
case ${SIRIUS_UCX_RAILS:-0} in
  1 | true)
    if ! _rails=$(sirius_rail_devices); then
      echo "SIRIUS_UCX_RAILS needs GPU rail NICs on this host" >&2
      return 1
    fi
    export UCX_TLS=${UCX_TLS:-all} UCX_NET_DEVICES=${UCX_NET_DEVICES:-$_rails}
    export SIRIUS_GATEWAY_EXCHANGE_HOST_STAGING=${SIRIUS_GATEWAY_EXCHANGE_HOST_STAGING:-never}
    unset _rails
    ;;
  0 | false) ;;
  *)
    echo "SIRIUS_UCX_RAILS takes 0 or 1" >&2
    return 1
    ;;
esac
# Otherwise the transports tested on one host and on two. Left to itself, UCX also picks the
# host's InfiniBand ports, and on multi-rail GB200 hosts the ranks then fail to connect ("Endpoint
# timeout").
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
# A value as a SQL string literal, for the SQL the launchers write: a path may hold an apostrophe.
sirius_sql_quote()
{
  local q="'"
  printf "'%s'" "${1//$q/$q$q}"
}
# DuckDB changes its output when an AI coding tool runs it; the ranks and the scripts parse it.
export DUCKDB_AGENT_MODE=${DUCKDB_AGENT_MODE:-0}
