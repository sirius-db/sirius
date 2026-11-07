# Defaults for every Sirius launcher. Source it (under set -euo pipefail too). Each value you
# export first wins.
#   NIXL_PREFIX (optional): only for a build whose RPATH does not find NIXL, such as a static one.
#   UCX_PREFIX (optional): the UCX that NIXL was built against, when it is not on the default path.
# The transports tested on one host and on two. Left to itself, UCX also picks the host's
# InfiniBand ports, and on multi-rail GB200 hosts the ranks then fail to connect ("Endpoint
# timeout").
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
# DuckDB changes its output when an AI coding tool runs it; the ranks and the scripts parse it.
export DUCKDB_AGENT_MODE=${DUCKDB_AGENT_MODE:-0}
[[ -n ${NIXL_PREFIX:-} ]] || return 0
_nixl_lib=$(set -- "$NIXL_PREFIX"/lib/*-linux-gnu; if [[ -d $1 ]]; then echo "$1"; else echo "$NIXL_PREFIX/lib"; fi)
export NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-$_nixl_lib/plugins}
# NIXL and its UCX plugin must come before the pixi environment's own UCX.
export LD_LIBRARY_PATH="$_nixl_lib${UCX_PREFIX:+:$UCX_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _nixl_lib
