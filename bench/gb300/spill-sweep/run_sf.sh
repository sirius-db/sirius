#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
sf="${1:?usage: run_sf.sh SCALE_FACTOR}"
shift
output_root="${SWEEP_OUTPUT:-$repo/test_datasets/gb300-spill-sweep}"
mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"
output="$output_root/sf$sf"
mkdir -p "$output"
rm -f "$output/exit-code"
if [[ -n "${PIXI_BIN:-}" ]]; then
  pixi="$PIXI_BIN"
elif [[ -x "$repo/test_datasets/.generation-tools/pixi" ]]; then
  pixi="$repo/test_datasets/.generation-tools/pixi"
else
  pixi="pixi"
fi
# SF10000 has hundreds of Parquet parts plus the io_uring descriptors.
ulimit -Sn 65536

cd "$repo"
set +e
"$pixi" run --locked python bench/gb300/spill-sweep/run.py --sf "$sf" \
  --output "$output_root" \
  --timeout "${QUERY_TIMEOUT:-3600}" "$@" \
  >> "$output/progress.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$output/exit-code"
exit "$status"
