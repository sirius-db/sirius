# storage

**Status**: USED (minimal)
**Path**: `duckdb/src/include/duckdb/storage/`
**Headers we include**:
- `duckdb/storage/data_table.hpp`

## Summary

The `storage` module implements DuckDB's persistent storage engine. Sirius only uses `DataTable` — included in operator headers for table scan and grouped aggregate operators that reference DuckDB's in-storage table representation.

## API Reference

### DataTable

**Header**: `duckdb/storage/data_table.hpp`

**Description**: Represents a table's physical storage. Referenced by GPU physical operators that need access to table metadata.

**Our usage**:
- `src/include/operator/gpu_physical_table_scan.hpp` — Table scan references DataTable
- `src/include/operator/gpu_physical_grouped_aggregate.hpp` — Grouped aggregate references DataTable
- `src/include/op/sirius_physical_duckdb_scan.hpp` — New code path scan operator
- `src/include/op/sirius_physical_grouped_aggregate.hpp` — New code path aggregate

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `StorageManager` | `storage_manager.hpp` | Database storage lifecycle |
| `BufferManager` | `buffer/buffer_manager.hpp` | Buffer pool for disk blocks |
| `ColumnSegment` | `segment/column_segment.hpp` | Columnar storage with compression |
| `CheckpointManager` | `checkpoint/checkpoint_manager.hpp` | WAL and checkpointing |
