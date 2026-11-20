#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if (( $# != 0 )); then
  echo "usage: $0 (set SWEEP_OUTPUT to choose an output directory)" >&2
  exit 2
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output_root="${SWEEP_OUTPUT:-$repo/test_datasets/gb300-spill-sweep/reproductions/all-sf-$run_id}"
mkdir -p "$output_root"
export SWEEP_OUTPUT="$(cd "$output_root" && pwd)"
export QUERY_TIMEOUT=3600
cd "$repo"

exec > >(tee -a "$SWEEP_OUTPUT/run.log") 2>&1
echo "Output: $SWEEP_OUTPUT"
for sf in 1000 3000 10000; do
  echo "Starting SF$sf at 32, 64, 128, 256, and 450 GiB host limits"
  bash bench/gb300-spill-sweep/run_sf.sh "$sf" --caps 32,64,128,256,450
  "$repo/test_datasets/.generation-tools/pixi" run --locked python - \
    "$SWEEP_OUTPUT/sf$sf/per-query.csv" \
    "$SWEEP_OUTPUT/sf$sf/totals.csv" <<'PY'
import csv
import sys
from pathlib import Path

queries = list(csv.DictReader(Path(sys.argv[1]).open()))
totals = list(csv.DictReader(Path(sys.argv[2]).open()))
caps = {32, 64, 128, 256, 450}
if len(queries) != 110 or {int(row["host_cap_gib"]) for row in totals} != caps:
    sys.exit("incomplete sweep")
for cap in caps:
    rows = [row for row in queries if int(row["host_cap_gib"]) == cap]
    if len(rows) != 22 or any(row["status"] != "success" for row in rows):
        sys.exit(f"host {cap} GiB has incomplete or failed queries")
    total = next(row for row in totals if int(row["host_cap_gib"]) == cap)
    if int(total["completed_queries"]) != 22 or int(total["successful_queries"]) != 22:
        sys.exit(f"host {cap} GiB totals are incomplete")
print("Validated 22 successful queries at each host limit")
PY
  echo "Completed SF$sf: $SWEEP_OUTPUT/sf$sf"
done
echo "All three scale factors completed successfully"
