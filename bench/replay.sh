#!/usr/bin/env bash
# Shared entry point; dry runs need only Python's standard library.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo 'Usage: replay.sh <gb300|g7e> <SF> <cold|hot|pinned> [options]' >&2
    exit 2
fi
hardware=$1
sf=$2
profile=$3
shift 3
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ "$hardware" == g7e ]]; then
    [[ "$profile" != pin ]] || profile=pinned
    exec bash "$repo/bench/g7e/guide-26b5caba/replay.sh" \
        --sf "$sf" --profile "$profile" "$@"
fi
for option in "$@"; do
    if [[ "$option" == --dry-run ]]; then
        exec python3 "$repo/bench/common/replay.py" --hardware "$hardware" --sf "$sf" --profile "$profile" "$@"
    fi
done
if [[ -n "${PIXI_BIN:-}" ]]; then
    pixi_bin=$PIXI_BIN
elif [[ -x "$repo/test_datasets/.generation-tools/pixi" ]]; then
    pixi_bin="$repo/test_datasets/.generation-tools/pixi"
elif [[ -x "$HOME/sirius/test_datasets/.generation-tools/pixi" ]]; then
    pixi_bin="$HOME/sirius/test_datasets/.generation-tools/pixi"
else
    pixi_bin=pixi
fi
cd "$repo"
exec "$pixi_bin" run --locked python bench/common/replay.py --hardware "$hardware" --sf "$sf" --profile "$profile" "$@"
