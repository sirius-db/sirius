# Runtime paths for a Sirius build with -DSIRIUS_WITH_NIXL=ON. Source it.
#   NIXL_PREFIX (required): the NIXL install Sirius was built with.
#   UCX_PREFIX (optional): the UCX that NIXL was built against, when it is not on the default path.
export NIXL_PREFIX=${NIXL_PREFIX:?set NIXL_PREFIX to the NIXL install Sirius was built with}
_nixl_lib=$(set -- "$NIXL_PREFIX"/lib/*-linux-gnu; if [[ -d $1 ]]; then echo "$1"; else echo "$NIXL_PREFIX/lib"; fi)
export NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-$_nixl_lib/plugins}
# cuda_copy lets UCX see device pointers; cuda_ipc moves them between processes on one host.
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
# NIXL and its UCX plugin must come before the pixi environment's own UCX.
export LD_LIBRARY_PATH="$_nixl_lib${UCX_PREFIX:+:$UCX_PREFIX/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _nixl_lib
