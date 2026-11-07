#!/usr/bin/env bash
# Builds NIXL (core and UCX plugin) against the pixi environment's UCX and installs it where a
# -DSIRIUS_WITH_NIXL=ON configure finds it without NIXL_PREFIX. Skips an install that works.
#   pixi run scripts/install_nixl.sh [prefix]
#   scripts/install_nixl.sh --print-prefix    # the install prefix; CMake uses it as its default
# Prefix: the argument, else NIXL_PREFIX, else ~/.sirius/nixl/<pinned commit>.
# NIXL_SRC: a local NIXL clone to build instead of cloning NIXL_REPO. With the default prefix it
# must be at the pinned commit, so that the prefix name says what is installed.
set -euo pipefail

# The NIXL commit the exchange is tested with (v1.3.1-81).
nixl_commit=4d030b94e4705e660f2ebe9f39ef7b636bdd6204
nixl_repo=${NIXL_REPO:-https://github.com/ai-dynamo/nixl}
default_prefix="$HOME/.sirius/nixl/${nixl_commit:0:8}"

if [[ ${1:-} == --print-prefix ]]; then
  echo "${NIXL_PREFIX:-$default_prefix}"
  exit 0
fi
prefix=${1:-${NIXL_PREFIX:-$default_prefix}}

ucx=${CONDA_PREFIX:?run through pixi run, so that NIXL builds against the pixi environment UCX}
if [[ ! -e $ucx/include/ucp/api/ucp.h ]]; then
  echo "no UCX in $ucx" >&2
  exit 1
fi
# Checked before anything is deleted, so that a missing pixi does not leave an emptied prefix.
pixi=${PIXI_EXE:-pixi}
if ! command -v "$pixi" >/dev/null; then
  echo "no pixi ($pixi); run through pixi run" >&2
  exit 1
fi

# The marker says this script installed into the prefix and lists, one per line, the files it
# wrote, so that a reinstall removes only those. The stamp says that install finished.
marker="$prefix/.sirius-install-nixl"
stamp="$prefix/.sirius-install-nixl-done"
# A NIXL whose UCX plugin still resolves libucp: its RPATH names the pixi environment it was built
# in, which goes away with that checkout. ldd's output is kept whole: piping it into grep -q could
# end ldd early, and a file ldd cannot read is not a working plugin.
ucx_plugin_resolves() {
  local deps
  deps=$(ldd "$1" 2>&1) && [[ $deps != *"not found"* ]]
}
# Done when the prefix holds a working NIXL: one this script installed must also have finished.
if [[ -e $stamp || ! -e $marker ]]; then
  for plugin in "$prefix"/lib/plugins/libplugin_UCX.so "$prefix"/lib/*-linux-gnu/plugins/libplugin_UCX.so; do
    if [[ -e $plugin ]] && ucx_plugin_resolves "$plugin"; then
      echo "NIXL: $prefix (installed)"
      exit 0
    fi
  done
fi
# Never install over someone else's files.
if [[ -d $prefix && ! -e $marker && -n $(ls -A "$prefix") ]]; then
  echo "$prefix holds something other than a working NIXL; give an empty or new prefix" >&2
  exit 1
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/sirius-nixl.XXXXXX")
# meson logs each file it installs; adding them to the marker even when the install fails or is
# interrupted lets the next run remove them.
install_log="$work/build/meson-logs/install-log.txt"
trap '[[ -e $install_log ]] && grep -v "^#" "$install_log" >>"$marker"; rm -rf "$work"' EXIT
if [[ -n ${NIXL_SRC:-} ]]; then
  src=$NIXL_SRC
  if [[ $prefix == "$default_prefix" && $(git -C "$src" rev-parse HEAD) != "$nixl_commit" ]]; then
    echo "$src is not at NIXL $nixl_commit; check it out or give a prefix" >&2
    exit 1
  fi
else
  src="$work/src"
  git clone --quiet --filter=blob:none "$nixl_repo" "$src"
  git -C "$src" checkout --quiet "$nixl_commit"
fi

echo "NIXL: building $(git -C "$src" describe --tags --always) against UCX in $ucx into $prefix"
rm -f "$stamp"
if [[ -e $marker ]]; then
  # Remove only what the last install wrote, so that files added to the prefix since then survive.
  # meson logs directories too: they go once the files in them are gone and they are empty (the
  # marker keeps rmdir -p from reaching the prefix).
  while IFS= read -r file; do
    if [[ $file != "$prefix"/* || (-d $file && ! -L $file) ]]; then
      continue
    fi
    rm -f -- "$file"
    rmdir -p --ignore-fail-on-non-empty -- "${file%/*}" 2>/dev/null || true
  done <"$marker"
fi
mkdir -p "$prefix"
: >"$marker"
# meson, ninja and pybind11 (NIXL's Python bindings always build) come from a throwaway pixi
# environment, so they do not enter pixi.lock. pkg-config then searches that environment, so
# Abseil comes from NIXL's own subproject, as in the install the exchange was tested with; the
# compiler and UCX are the project environment's.
# liburing is the project environment's too: NIXL would otherwise install its own copy next to
# libnixl, where libsirius's RPATH finds it first.
mkdir "$work/pkgconfig"
ln -s "$ucx/lib/pkgconfig/liburing.pc" "$work/pkgconfig/"
# -Ducx_path puts the pixi UCX in the UCX plugin's RPATH; the link argument adds the pixi
# libstdc++ that the compiler needs to every NIXL library. Without bytecode compilation every
# installed file is in meson's install log (its .pyc files are not).
PKG_CONFIG_PATH="$work/pkgconfig" "$pixi" exec --spec 'meson>=1.3' --spec ninja --spec pybind11 -- bash -c '
  set -euo pipefail
  meson setup "$1" "$2" --prefix="$3" --libdir=lib -Dbuildtype=release -Dwerror=false \
    -Ducx_path="$4" -Denable_plugins=UCX -Dbuild_tests=false -Dbuild_examples=false \
    -Dcpp_link_args="-Wl,-rpath,$4/lib" -Dpython.bytecompile=-1
  ninja -C "$1" install' _ "$work/build" "$src" "$prefix" "$ucx"

plugin="$prefix/lib/plugins/libplugin_UCX.so"
if ! ucx_plugin_resolves "$plugin"; then
  echo "NIXL: $plugin does not resolve its libraries:" >&2
  ldd "$plugin" >&2 || true
  exit 1
fi
echo "$(git -C "$src" describe --tags --always) built against $ucx" >"$stamp"
echo "NIXL: $prefix (installed)"
