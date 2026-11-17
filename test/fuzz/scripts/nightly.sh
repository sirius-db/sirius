#!/usr/bin/env bash
set -euo pipefail

duration="${1:?usage: nightly.sh DURATION [OUTPUT_ROOT] [PRIOR_SIGNATURES]}"
repo="$(git rev-parse --show-toplevel)"
root="${2:-$repo/test/fuzz/out/nightly}"
prior="${3:-$root/prior-signatures.txt}"
shell="$repo/sirius-duckdb/build/release/duckdb"
extension="$repo/sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
test -x "$shell"
test -f "$extension"
mkdir -p "$root"
prior_args=()
if test -f "$prior"; then
  prior_args=(--prior-signatures "$prior")
fi
cd "$repo"

pixi run fuzz doctor --shell "$shell" --extension "$extension" \
  --out "$root/doctor"
set +e
pixi run fuzz corpus --shell "$shell" --extension "$extension" \
  --out "$root/corpus.json" | tee "$root/corpus.log"
corpus_rc=${PIPESTATUS[0]}
# A new seed each day; set FUZZ_SEED to repeat a run.
seed="${FUZZ_SEED:-$(date -u +%Y%m%d)}"
pixi run fuzz run --no-doctor --seed "$seed" --workers 2 \
  --duration "$duration" --set oracle.reduce_max_steps=5 \
  --set oracle.query_timeout_seconds=60 --fail-on-findings \
  "${prior_args[@]}" \
  --shell "$shell" --extension "$extension" --out "$root/campaign" \
  | tee "$root/campaign.log"
campaign_rc=${PIPESTATUS[0]}
run_dir="$(ls -td "$root"/campaign/run-* | head -1)"
pixi run fuzz coverage-check "$run_dir" | tee "$root/coverage-check.log"
coverage_rc=${PIPESTATUS[0]}
set -e
printf 'Completed %s; yield history: %s\n' "$run_dir" "$root/campaign/yield.jsonl"
if (( corpus_rc || campaign_rc || coverage_rc )); then
  exit 1
fi
