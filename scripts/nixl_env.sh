# Defaults for every Sirius launcher. Source it (under set -euo pipefail too). Each value you
# export first wins.
#   NIXL_PREFIX (optional): only for a build whose RPATH does not find NIXL, such as a static one.
#   UCX_PREFIX (optional): the UCX that NIXL was built against, when it is not on the default path.
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
# Without the InfiniBand transports nothing looks a buffer up in UCX's registration cache, so
# nothing drains the list where it queues every freed GPU buffer, and each cudaFreeAsync walks
# that whole list under one lock (UCX 1.21, ucs_rcache_unmapped_callback). The join after Q14's
# exchange took 12 s instead of under 1 s. The exchange registers its rings itself. Any UCX_TLS
# but "all" turns the cache off; with a list of your own that has rc, dc or ud in it, export
# UCX_RCACHE_ENABLE=y.
[[ $UCX_TLS == all ]] || export UCX_RCACHE_ENABLE=${UCX_RCACHE_ENABLE:-n}
# DuckDB changes its output when an AI coding tool runs it; the ranks and the scripts parse it.
export DUCKDB_AGENT_MODE=${DUCKDB_AGENT_MODE:-0}
[[ -n ${NIXL_PREFIX:-} ]] || return 0
_nixl_lib=$(set -- "$NIXL_PREFIX"/lib/*-linux-gnu; if [[ -d $1 ]]; then echo "$1"; else echo "$NIXL_PREFIX/lib"; fi)
export NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-$_nixl_lib/plugins}
# NIXL and its UCX plugin must come before the pixi environment's own UCX.
export LD_LIBRARY_PATH="$_nixl_lib${UCX_PREFIX:+:$UCX_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _nixl_lib
