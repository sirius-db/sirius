#!/usr/bin/env bash
# Run TPC-DS on Sirius through performance_test.py, with bench/sf1000-repro/run.sh's settings.
# Usage: DATA=<tpcds parquet dir or .duckdb file> pixi run bash test/performance/run_tpcds.sh \
#          [performance_test.py args]
#
# If <dataset>_duckdb_results/ exists next to DATA (DuckDB reference results saved from an
# earlier --engine cpu run), and neither --engine nor --duckdb-results is given, only the GPU
# runs and its results are validated against the saved ones. Override with REF=<dir>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

DATA="${DATA:?set DATA to a TPC-DS parquet directory or .duckdb file}"
CFG="${CFG:-$HERE/tpcds_sirius.yaml}"
DATA_BASE="${DATA%/}"
REF="${REF:-${DATA_BASE%.duckdb}_duckdb_results}"

export SIRIUS_PRE_SQL="SET pin_table_compression = true; \
SET expression_evaluator_strategy = 'ast_jit'; \
SET parquet_metadata_cache = true"

source_args=(--input "$DATA")
[[ "$DATA" == *.duckdb ]] && source_args+=(--data-source duckdb)

ref_args=()
if [[ -d "$REF" && " $* " != *" --engine"* && " $* " != *" --duckdb-results"* ]]; then
  echo "run_tpcds.sh: validating against saved DuckDB results in $REF"
  ref_args=(--engine gpu --duckdb-results "$REF")
fi

cd "$REPO"
python3 test/performance/performance_test.py --benchmark tpcds "${source_args[@]}" \
  --config "$CFG" "${ref_args[@]}" "$@"
