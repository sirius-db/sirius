#!/usr/bin/env bash
# Generate the Phase-5 clustered-stretch SF1000 parquet dataset: TPC-H SF1000 physically sorted on
# the Top-N dynamic filter's ORDER-BY keys (orders by o_totalprice for Q18's group producer,
# supplier by s_acctbal for Q2's row producer), so on-disk row-group min/max zone maps become
# selective and the parquet reader's pruning gate can actually skip work. On the unclustered
# dbgen-order dataset those keys span every row group and the gate correctly disables itself, so
# this dataset is what demonstrates the mechanism's upside (it is not part of the zero-regression
# bar).
#
# Wraps generate_tpch_data.sh --cluster-keys, whose parquet path sorts through DuckDB but writes
# through pyarrow (cluster_parquet.py) so FIXED_LEN_BYTE_ARRAY decimals survive and the clustered
# dataset stays comparable to the unclustered original.
#
# COST: ~265 GB on disk plus a multi-hour sort/write. Run it as a background job in an idle
# window, NEVER concurrently with a timing cell -- the sort competes for host memory and I/O.
#
# Usage:
#   nohup ./test/tpch_performance/phase5_generate_clustered_sf1000.sh \
#       [output_dir] > /tmp/phase5_cluster_gen.log 2>&1 &
#
#   output_dir default: /localhome/local-kkristensen/tpch_parquet_sf1000_topn_clustered
#
# Verification after generation (per test/tpch_performance/CLAUDE.md, "Clustered (sorted)
# datasets"):
#   1. FLBA preserved:   pixi run python -c "import pyarrow.parquet as pq; \
#        print(pq.ParquetFile('<out>/orders/part.1.parquet').schema)"   # DECIMAL stays FLBA
#   2. Row counts match the unclustered original per table (parquet footer num_rows).
#   3. Sortedness: min/max of o_totalprice per row group are non-overlapping ranges.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SF=1000
OUTPUT="${1:-/localhome/local-kkristensen/tpch_parquet_sf1000_topn_clustered}"
CLUSTER_KEYS="orders:o_totalprice,supplier:s_acctbal"

echo "Generating clustered TPC-H SF${SF} parquet:"
echo "  Output:       $OUTPUT"
echo "  Cluster keys: $CLUSTER_KEYS"
echo "  (multi-hour job; ~265 GB)"

bash "$SCRIPT_DIR/generate_tpch_data.sh" "$SF" \
    --format parquet \
    --output "$OUTPUT" \
    --cluster-keys "$CLUSTER_KEYS"

echo "Done. Verify FLBA preservation, row counts, and per-row-group min/max before first use."
