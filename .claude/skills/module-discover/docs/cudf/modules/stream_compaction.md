# Stream Compaction

**Status**: USED
**Path**: `cudf/stream_compaction.hpp`, `cudf/detail/stream_compaction.hpp`
**Headers we include**: `cudf/stream_compaction.hpp`, `cudf/detail/stream_compaction.hpp`

## Summary

Used for duplicate elimination (SQL DISTINCT). Sirius accesses both public and detail APIs for version compatibility.

## API Reference

### `cudf::drop_duplicates`

**Header**: `cudf/stream_compaction.hpp`
```cpp
std::unique_ptr<table> drop_duplicates(table_view const& input,
                                        std::vector<size_type> const& keys,
                                        duplicate_keep_option keep, ...);
```

**Our usage**:
- `src/cuda/cudf/cudf_duplicate_elimination.cu` — SQL DISTINCT implementation

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::apply_boolean_mask()` | `stream_compaction.hpp` | Filter rows by boolean column |
| `cudf::unique()` | `stream_compaction.hpp` | Remove consecutive duplicates (sorted input) |
| `cudf::distinct()` | `stream_compaction.hpp` | Keep distinct rows |
| `cudf::stable_distinct()` | `stream_compaction.hpp` | Stable distinct preserving order |
