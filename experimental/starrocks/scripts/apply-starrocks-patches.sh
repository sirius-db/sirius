#!/usr/bin/env bash
# Re-applies Sirius-only patches onto the vendored StarRocks checkout.
# Safe to re-run: skips a patch that already applies (or is already present).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
submodule="$root/starrocks"
patch_dir="$root/patches"

if [[ ! -f "$submodule/gensrc/proto/internal_service.proto" ]]; then
  echo "StarRocks submodule is not checked out at $submodule" >&2
  echo "Run: git submodule update --init --recursive experimental/starrocks/starrocks" >&2
  exit 1
fi

shopt -s nullglob
patches=("$patch_dir"/*.patch)
if ((${#patches[@]} == 0)); then
  echo "No patches under $patch_dir" >&2
  exit 1
fi

cd "$submodule"
for patch in "${patches[@]}"; do
  if git apply --check "$patch" >/dev/null 2>&1; then
    git apply "$patch"
    echo "applied $(basename "$patch")"
  elif git apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "already applied $(basename "$patch")"
  else
    echo "failed to apply $(basename "$patch")" >&2
    exit 1
  fi
done
