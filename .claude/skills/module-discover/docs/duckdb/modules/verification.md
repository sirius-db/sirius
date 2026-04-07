# verification

**Status**: UNUSED
**Path**: `duckdb/src/include/duckdb/verification/`

## Summary

The `verification` module provides DuckDB's internal query verification infrastructure — re-executes queries with different configurations to detect correctness bugs.

## Key APIs
- `StatementVerifier` — Runs queries multiple ways to verify consistency
- `PhysicalVerifyVector` — Verifies vector data integrity

## Potential Relevance

Could be useful for testing Sirius correctness — verifying that GPU results match CPU results. Currently not used; Sirius has its own validation in `performance_test.py`.
