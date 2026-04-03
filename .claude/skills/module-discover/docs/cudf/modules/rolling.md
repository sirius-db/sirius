# Rolling

**Status**: UNUSED
**Path**: `cudf/rolling/`

## Summary
Rolling (sliding) window aggregation operations. Computes aggregates over a window of rows (e.g., moving average, running sum).

## Key APIs
- `cudf::rolling_window()` — Apply aggregation over rolling window
- `cudf::grouped_rolling_window()` — Rolling window within groups
- `cudf::range_window_bounds` — Window boundary specification

## Potential Relevance
Would be needed to support SQL window functions (OVER clause with ROWS/RANGE). Currently Sirius falls back to DuckDB CPU for window functions.
