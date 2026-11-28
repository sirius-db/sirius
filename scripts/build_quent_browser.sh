#!/usr/bin/env bash
# Build the Quent browser analyzer for the web.
#
# Usage: build_quent_browser.sh [out_dir]
#
# Writes sirius_telemetry_browser.js and sirius_telemetry_browser_bg.wasm to out_dir
# (default build/quent-browser). Run through `pixi run -e browser build-quent-browser`.
set -euo pipefail

out_dir="${1:-build/quent-browser}"
manifest="rust/Cargo.toml"
artifact="sirius_telemetry_browser"

# The wasm-bindgen CLI must match the wasm-bindgen crate that Cargo.lock selects, or the
# generated JavaScript cannot load the module.
locked_version=$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' rust/Cargo.lock)
cli_version=$(wasm-bindgen --version | awk '{ print $2 }')
if [[ "$cli_version" != "$locked_version" ]]; then
  echo "wasm-bindgen CLI $cli_version does not match rust/Cargo.lock ($locked_version);" \
    "update wasm-bindgen-cli in the browser feature of pixi.toml" >&2
  exit 1
fi

cargo build --locked --release --manifest-path "$manifest" \
  -p sirius-telemetry-browser --target wasm32-unknown-unknown

wasm-bindgen "${CARGO_TARGET_DIR:-rust/target}/wasm32-unknown-unknown/release/${artifact}.wasm" \
  --target web --out-dir "$out_dir"

test -s "$out_dir/${artifact}.js"
test -s "$out_dir/${artifact}_bg.wasm"
ls -l "$out_dir"
