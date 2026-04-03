# T-Digest

**Status**: UNUSED
**Path**: `cudf/tdigest/`

## Summary
T-Digest data structure for approximate percentile/quantile computation. Useful for streaming approximate analytics.

## Key APIs
- `cudf::tdigest_column_view` — View into t-digest column
- `cudf::percentile_approx()` — Approximate percentile from t-digest

## Potential Relevance
Could be useful for implementing PERCENTILE_CONT/PERCENTILE_DISC approximate variants for large datasets.
