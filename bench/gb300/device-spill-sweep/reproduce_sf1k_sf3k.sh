#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if (( $# != 0 )); then
  echo "usage: $0 (set SWEEP_OUTPUT to choose an output directory)" >&2
  exit 2
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
output_root="${SWEEP_OUTPUT:-$repo/test_datasets/gb300-device-spill-sweep/reproductions/sf1k-sf3k-$run_id}"
mkdir -p "$output_root"
export SWEEP_OUTPUT="$(cd "$output_root" && pwd)"
export HOST_CAP_GIB=384
export QUERY_TIMEOUT=3600
cd "$repo"

exec > >(tee -a "$SWEEP_OUTPUT/run.log") 2>&1
echo "Output: $SWEEP_OUTPUT"
for sf in 1000 3000; do
  echo "Starting SF$sf at 32, 64, 128, and 256 GB"
  bash bench/gb300/device-spill-sweep/run_sf.sh "$sf" --caps 32,64,128,256
  test_datasets/.generation-tools/pixi run --locked python - \
    "$SWEEP_OUTPUT/host384Gi/sf$sf/per-query.csv" \
    "$SWEEP_OUTPUT/host384Gi/sf$sf/totals.csv" <<'PY'
import csv
import sys
from pathlib import Path

queries = list(csv.DictReader(Path(sys.argv[1]).open()))
totals = list(csv.DictReader(Path(sys.argv[2]).open()))
caps = {32, 64, 128, 256}
if len(queries) != 88 or {int(row["gpu_cap_gb"]) for row in totals} != caps:
    sys.exit("incomplete sweep")
for cap in caps:
    rows = [row for row in queries if int(row["gpu_cap_gb"]) == cap]
    if len(rows) != 22 or any(row["status"] != "success" for row in rows):
        sys.exit(f"GPU {cap} GB has incomplete or failed queries")
    total = next(row for row in totals if int(row["gpu_cap_gb"]) == cap)
    if int(total["completed_queries"]) != 22 or int(total["successful_queries"]) != 22:
        sys.exit(f"GPU {cap} GB totals are incomplete")
print("Validated 22 successful queries at each GPU limit")
PY
  echo "Completed SF$sf: $SWEEP_OUTPUT/host384Gi/sf$sf"
done
echo "Both scale factors completed successfully"
