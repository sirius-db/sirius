#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
engine=
dry_run=false
arguments=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)
            [[ $# -ge 2 ]] || { echo '--source-dir needs a path' >&2; exit 2; }
            engine=$2
            shift
            ;;
        --dry-run) dry_run=true ;;
    esac
    shift
done
if "$dry_run"; then
    exec python3 "$here/replay.py" "${arguments[@]}"
fi
[[ -n "$engine" ]] || { echo '--source-dir is required' >&2; exit 2; }
export CONDA_OVERRIDE_CUDA=13.5
unset AWS_PROFILE AWS_DEFAULT_PROFILE
ulimit -l unlimited
ulimit -n 1048576
ulimit -c 0
exec "${PIXI_BIN:-pixi}" run --locked --manifest-path "$engine/pixi.toml" \
    python "$here/replay.py" "${arguments[@]}"
