#!/usr/bin/env sh
# Sourced by pixi during environment activation — must be POSIX-compatible.

_root="${PIXI_PROJECT_ROOT:-.}"

if [ -d "$_root/duckdb" ]; then
  # Remove stale symlink from dev branch (pointed to ../cmake/CMakePresets.json which
  # doesn't exist on this branch), then create the file if missing.
  rm -f "$_root/duckdb/CMakePresets.json"
  printf '{"version":6,"include":["../CMakePresets.json"]}\n' > "$_root/duckdb/CMakePresets.json"
fi

mkdir -p build
pixi shell-hook -s bash > build/sirius_pixi_env_for_clion.sh
