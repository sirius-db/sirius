#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$project_root/duckdb/CMakePresets.json" ]]; then
  printf '{"version":6,"include":["../CMakePresets.json"]}\n' > "$project_root/duckdb/CMakePresets.json"
fi
