#!/usr/bin/env bash
# Replay the selected GB300 Sirius TPC-H cold, hot, or host-pin workload.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'Usage: replay.sh <100|300|1000|3000|10000> <cold|hot|pin> [--dry-run]' >&2
    exit 2
fi
sf=$1
profile=$2
dry_run=${3:-}
case "$sf" in 100|300|1000|3000|10000) ;; *) echo "Unsupported SF: $sf" >&2; exit 2 ;; esac
case "$profile" in cold|hot|pin) ;; *) echo "Unsupported profile: $profile" >&2; exit 2 ;; esac
if [[ -n "$dry_run" && "$dry_run" != --dry-run ]]; then
    echo "Unknown option: $dry_run" >&2
    exit 2
fi
if [[ "$sf" == 10000 && "$profile" == pin ]]; then
    echo 'SF10000 host pin has no successful saved result: Q13 exhausted the host pool.' >&2
    exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dataset="${DATASET_ROOT:-/mnt/datasets}/tpch_parquet_sf${sf}"
output="${GB300_REPLAY_OUTPUT:-$repo/test_datasets/gb300-replay-output}"
if [[ -n "${PIXI_BIN:-}" ]]; then
    pixi_bin=$PIXI_BIN
elif [[ -x "$repo/test_datasets/.generation-tools/pixi" ]]; then
    pixi_bin="$repo/test_datasets/.generation-tools/pixi"
elif [[ -x "$HOME/sirius/test_datasets/.generation-tools/pixi" ]]; then
    pixi_bin="$HOME/sirius/test_datasets/.generation-tools/pixi"
else
    pixi_bin=pixi
fi
if [[ ! -d "$dataset" ]]; then
    echo "Dataset is missing: $dataset" >&2
    exit 1
fi

mode=$(awk -F '"' '/^CoherentGPUMemoryMode:/ {print $2}' \
    /proc/driver/nvidia/params 2>/dev/null || true)
if [[ "$mode" != driver ]]; then
    echo "Sirius GB300 replay requires CDMM (driver mode); found '${mode:-unknown}'." >&2
    exit 1
fi

unset SIRIUS_CONFIG_FILE SIRIUS_PRE_SQL SIRIUS_DISABLE
case "$profile" in
    cold|hot)
        unset SIRIUS_PIN_COMPRESSION_TABLES
        config="$repo/bench/gb300-$profile/config.yaml"
        iterations=2
        runner_profile=$profile
        extra=()
        ;;
    pin)
        export SIRIUS_PIN_COMPRESSION_TABLES=lineitem,orders
        config="$repo/bench/gb300-pin/config.yaml"
        iterations=1
        runner_profile=hot
        extra=(--pin host --pin-compression)
        ;;
esac

cd "$repo"
run_queries() {
    local config_path=$1
    local queries=$2
    local label=$3
    local command=("$pixi_bin" run python test/tpch_performance/performance_test.py
        --input "$dataset" --scale-factor "$sf" --engine gpu
        --profile "$runner_profile" --iterations "$iterations"
        --queries "$queries" --config "$config_path" --output "$output"
        --name "$label" "${extra[@]}")
    if [[ "$dry_run" == --dry-run ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

if [[ "$sf" == 10000 && "$profile" != pin ]]; then
    run_queries "$config" '1-9,11-22' "gb300_${profile}_sf${sf}_except_q10"
    run_queries "$repo/bench/gb300-sf10k-q10-$profile/config.yaml" 10 \
        "gb300_${profile}_sf${sf}_q10"
else
    run_queries "$config" 1-22 "gb300_${profile}_sf${sf}"
fi
