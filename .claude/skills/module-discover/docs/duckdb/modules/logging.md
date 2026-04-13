# logging

**Status**: UNUSED
**Path**: `duckdb/src/include/duckdb/logging/`

## Summary

The `logging` module provides DuckDB's internal logging infrastructure for diagnostic output.

## Key APIs
- `Logger` — Logging interface
- `LogManager` — Log configuration and routing
- `LogType` — Log entry classification

## Potential Relevance

Not applicable — Sirius uses its own logging via spdlog (`SIRIUS_LOG_DIR`, `SIRIUS_LOG_LEVEL`).
