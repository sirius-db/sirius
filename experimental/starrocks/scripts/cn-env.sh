# Source (don't execute) to derive every machine-specific path the engine-linked CN needs, at
# build time (nixl-sys bindgen + system toolchain) and at run time (nixl/UCX libraries and
# plugins). Everything is computed from this file's own location, so the repo works from any
# clone path; the only external input is where the out-of-tree nixl/UCX installs live:
#
#   TOOLS_DIR   (default: a `tools/` directory next to the repo root, per notes-setup.md §3)
#
# NIXL_PREFIX / UCX_PREFIX / NIXL_PLUGIN_DIR may also be overridden individually; values
# already exported by the operator always win.
#
# ONE MANUAL STEP THIS FILE DOES NOT DO FOR YOU: the CN's exchange RPCs are a Sirius-only
# extension to StarRocks' internal_service.proto, and nothing applies them automatically. On a
# fresh clone, or after any `git submodule update` that resets the submodule working tree:
#
#   git -C ../starrocks apply ../patches/nixl-exchange-proto.patch
#
# build.rs checks for it and fails with that command in the message, so you cannot get far
# without noticing -- but doing it up front saves a build.
#
# For the environment variables the CN itself reads, see docs/TUNABLES.md.

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # experimental/starrocks/scripts
SR_DIR=$(cd "$_here/.." && pwd)                       # experimental/starrocks
REPO_ROOT=$(cd "$SR_DIR/../.." && pwd)                # repo root
TOOLS_DIR=${TOOLS_DIR:-$(cd "$REPO_ROOT/.." && pwd)/tools}

export NIXL_PREFIX=${NIXL_PREFIX:-$TOOLS_DIR/nvda_nixl}
UCX_PREFIX=${UCX_PREFIX:-$TOOLS_DIR/ucx-install}

if [ ! -d "$NIXL_PREFIX/lib" ]; then
    echo "no nixl install at $NIXL_PREFIX -- set TOOLS_DIR (or NIXL_PREFIX), or build it:" >&2
    echo "see benchmarks/nixl-nvlink/notes-setup.md section 3" >&2
    return 1 2>/dev/null || exit 1
fi

# Multiarch libdir: lib/x86_64-linux-gnu or lib/aarch64-linux-gnu, whichever was built.
_nixl_lib=$(set -- "$NIXL_PREFIX"/lib/*-linux-gnu; echo "$1")
[ -d "$_nixl_lib" ] || { echo "no multiarch libdir under $NIXL_PREFIX/lib" >&2
                         return 1 2>/dev/null || exit 1; }
export NIXL_PLUGIN_DIR=${NIXL_PLUGIN_DIR:-$_nixl_lib/plugins}

# Without the stubs guard a broken nixl link silently degrades to a dlopen stub; without
# cuda_copy UCX cannot detect VRAM pointers, without cuda_ipc same-host transfers take a ~200x
# slower host bounce.
export NIXL_NO_STUBS_FALLBACK=1
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}

# Engine .so first; the repo pixi env's lib/ supplies the GLIBCXX_3.4.31 that libcudf and the
# extension need when launched outside pixi activation.
_pixi_env=$REPO_ROOT/.pixi/envs/default
# nixl + UCX must precede the pixi env lib: the UCX plugin is built against the system
# toolchain, and a pixi lib ahead of it shadows a dependency so dlopen of
# libplugin_UCX.so fails -- nixl then reports the opaque "backend 'UCX' not found".
export LD_LIBRARY_PATH="$REPO_ROOT/build/release/extension/sirius:$_nixl_lib:$UCX_PREFIX/lib:$_pixi_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$UCX_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# System gcc/g++ must compile nixl-sys: pixi activation injects CXXFLAGS mixing /usr/include
# with the conda sysroot, which fails on glibc headers (bits/timesize.h). Clear the conda
# flags and force the host toolchain; LIBRARY_PATH keeps libucs visible to the link.
unset CXXFLAGS CFLAGS CPPFLAGS CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH || true
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/usr/bin/gcc
# libsirius.so declares libcuda.so.1 and libnvidia-ml.so.1 in DT_NEEDED, so the CN link has to be
# able to FIND them -- LIBRARY_PATH is link time, LD_LIBRARY_PATH above is only run time. Without
# the driver dir the link fails with:
#   libsirius.so: undefined reference to `cuLaunchKernel'
#   libsirius.so: undefined reference to `nvmlDeviceGetMemoryAffinity'
# (MEASURED 2026-08-11. This surfaced only once the engine started calling the CUDA driver API
# directly; an older libsirius had no such DT_NEEDED entry, so the omission was latent.)
_driver_lib=/usr/lib/$(uname -m)-linux-gnu
export LIBRARY_PATH="$UCX_PREFIX/lib:$_driver_lib:/usr/local/cuda/lib64/stubs"

# Force the SYSTEM linker. CC is /usr/bin/gcc above, but gcc resolves `ld` from PATH, and under
# `pixi run` that is the conda ld -- which then links the conda sysroot's libpthread.so.0 against
# the system libc and fails with 39 GLIBC_PRIVATE undefined references
# (__libc_dlopen_mode, __nanosleep_nocancel, _dl_make_stack_executable, ...). MEASURED 2026-08-11.
#
# Do NOT do this with RUSTFLAGS="-C link-arg=-B/usr/bin": setting RUSTFLAGS invalidates cargo's
# whole fingerprint cache, which re-runs the nixl-sys build script, which then fails on
# `/usr/include/features-time64.h: fatal error: bits/timesize.h: No such file or directory`
# -- a separate latent breakage that a warm target/ dir had been hiding. Prepending PATH gets the
# system ld without touching the fingerprint.
export PATH="/usr/bin:$PATH"

# nixl-sys bindgen needs libclang plus its builtin headers; the system libclang ships none, the
# repo pixi env's clang does. Glob the version so a clang bump doesn't break the build.
export LIBCLANG_PATH=${LIBCLANG_PATH:-$_pixi_env/lib}
if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]; then
    _clang_inc=$(set -- "$_pixi_env"/lib/clang/*/include; echo "$1")
    [ -d "$_clang_inc" ] || { echo "no clang builtin headers under $_pixi_env/lib/clang" >&2
                              echo "run \`pixi install\` at the repo root first" >&2
                              return 1 2>/dev/null || exit 1; }
    export BINDGEN_EXTRA_CLANG_ARGS="-isystem $_clang_inc"
fi

unset _here _nixl_lib _pixi_env _clang_inc
