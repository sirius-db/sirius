# Runtime paths for a Sirius build with -DSIRIUS_WITH_NIXL=ON. Source it.
#   NIXL_PREFIX (default ~/tools/nvda_nixl) and UCX_PREFIX (default ~/tools/ucx-install).
export NIXL_PREFIX=${NIXL_PREFIX:-$HOME/tools/nvda_nixl}
UCX_PREFIX=${UCX_PREFIX:-$HOME/tools/ucx-install}
_nixl_lib=$(set -- "$NIXL_PREFIX"/lib/*-linux-gnu; echo "$1")
export NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-$_nixl_lib/plugins}
# cuda_copy lets UCX see device pointers; cuda_ipc moves them between processes on one host.
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
# NIXL and its UCX plugin must come before the pixi environment's own UCX.
export LD_LIBRARY_PATH="$_nixl_lib:$UCX_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _nixl_lib UCX_PREFIX
