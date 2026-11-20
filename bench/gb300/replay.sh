#!/usr/bin/env bash
# Compatibility entry point for the GB300 workload.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/../replay.sh" gb300 "$@"
