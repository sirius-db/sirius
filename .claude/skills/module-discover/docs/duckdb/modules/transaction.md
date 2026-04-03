# transaction

**Status**: UNUSED
**Path**: `duckdb/src/include/duckdb/transaction/`

## Summary

The `transaction` module provides MVCC transaction management — transaction lifecycle, conflict detection, undo buffers, and isolation levels.

## Key APIs
- `TransactionContext` — Per-connection transaction state (indirectly used via ClientContext)
- `TransactionManager` — Global transaction coordination
- `DuckTransaction` — Individual transaction instance

## Potential Relevance

Not directly relevant. Sirius runs read-only analytical queries and relies on DuckDB's transaction context implicitly through `ClientContext`. No direct transaction management is needed.
