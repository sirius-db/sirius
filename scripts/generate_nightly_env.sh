#!/usr/bin/env bash
# Generates envs/nightly/pixi.toml from the main pixi.toml.
# Swaps rapidsai → rapidsai-nightly and unpins libcudf to get the latest nightly build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$REPO_ROOT/envs/nightly"

mkdir -p "$TARGET_DIR"

sed \
  -e 's|channels = \["rapidsai", "conda-forge" \]|channels = ["rapidsai-nightly", "conda-forge"]|' \
  -e 's|name = "sirius"|name = "sirius-nightly"|' \
  -e 's|libcudf = "[^"]*"|libcudf = "*"|' \
  -e 's|librmm = "[^"]*"|librmm = "*"|' \
  -e 's|libcuvs = "[^"]*"|libcuvs = "*"|' \
  -e 's|"scripts/pixi_activate.sh"|"../../scripts/pixi_activate.sh"|' \
  -e '/^\[feature\.nightly-runner/,/^$/d' \
  -e '/^nightly-/d' \
  -e '/^# .*nightly cudf/d' \
  "$REPO_ROOT/pixi.toml" > "$TARGET_DIR/pixi.toml"

echo "Generated $TARGET_DIR/pixi.toml"
