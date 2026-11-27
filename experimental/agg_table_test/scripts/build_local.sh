#!/usr/bin/env bash
# Configure + build the standalone agg_table differential test into
# experimental/agg_table_test/build/, then (with `run`) run it.
#
#   scripts/build_local.sh          build
#   scripts/build_local.sh run ...  build, then run the test with the remaining arguments
#
# Env overrides:
#   AGG_PIXI_MANIFEST  pixi.toml whose environment provides cmake/ninja/nvcc/libcudf
#   AGG_CUDA_ARCH      CMAKE_CUDA_ARCHITECTURES (default 121-real: GB10)
#   AGG_BUILD_DIR      build dir (default <agg_table_test>/build)
#   AGG_JOBS           ninja -j  (default: ninja's default)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
export PATH="$HOME/.pixi/bin:$PATH"

# Pick a pixi environment: this worktree's if installed, else an already
# installed sibling worktree env (same pixi.lock).
manifest="${AGG_PIXI_MANIFEST:-}"
if [[ -z "$manifest" ]]; then
  manifest="$repo/pixi.toml"
  if [[ ! -d "$repo/.pixi/envs/default" ]]; then
    for cand in "$repo"/../*/pixi.toml; do
      if [[ -d "$(dirname "$cand")/.pixi/envs/default" ]] &&
        cmp -s "$(dirname "$cand")/pixi.lock" "$repo/pixi.lock"; then
        manifest="$(realpath "$cand")"
        break
      fi
    done
  fi
fi

arch="${AGG_CUDA_ARCH:-121-real}"
build="${AGG_BUILD_DIR:-$here/build}"
echo "[build_local] pixi manifest: $manifest  cuda arch: $arch" >&2

cd "$here" # pixi's activation script does `mkdir -p build` in the cwd
run=(pixi run --manifest-path "$manifest" --frozen --)
"${run[@]}" cmake -S "$here" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$arch"
"${run[@]}" cmake --build "$build" ${AGG_JOBS:+-j "$AGG_JOBS"}
echo "[build_local] built $build/agg_table_test" >&2
if [[ "${1:-}" == "run" ]]; then
  shift
  "${run[@]}" "$build/agg_table_test" "$@"
fi
