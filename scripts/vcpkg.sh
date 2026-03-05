#!/usr/bin/env sh -e

if [ ! -f "./vcpkg/vcpkg" ]; then
  ./vcpkg/bootstrap-vcpkg.sh
fi

# Strip conda-injected -Wl,-rpath from LDFLAGS so the compiler wrapper
# does not embed an RPATH to the pixi/conda env in built binaries.
# All deps are statically linked; no runtime RPATH is needed.
# -Wl,-rpath-link (link-time only) is intentionally kept.
LDFLAGS=$(echo "$LDFLAGS" | sed 's/-Wl,-rpath,[^ ]*//g; s/  */ /g; s/^ //; s/ $//')
export LDFLAGS
