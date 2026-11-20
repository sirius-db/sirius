#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
sf="${1:-10000}"
shift || true
output="$repo/test_datasets/gb300-device-spill-sweep/sf$sf"
mkdir -p "$output"
export PATH="$repo/test_datasets/.generation-tools:$PATH"
pixi="$repo/test_datasets/.generation-tools/pixi"
ulimit -Sn 65536

cd "$repo"
set +e
"$pixi" run --locked python bench/gb300-device-spill-sweep/run.py --sf "$sf" \
  --timeout "${QUERY_TIMEOUT:-3600}" "$@" \
  >> "$output/progress.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$output/exit-code"
exit "$status"
