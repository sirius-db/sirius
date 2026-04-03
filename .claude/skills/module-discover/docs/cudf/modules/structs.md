# Structs

**Status**: UNUSED
**Path**: `cudf/structs/`

## Summary
Support for struct (record) columns — composite types with named fields. Provides views into struct column components.

## Key APIs
- `cudf::structs_column_view` — View into struct column
- `cudf::structs::detail::flatten_nested_columns()` — Flatten nested structs

## Potential Relevance
Not applicable — Sirius does not support nested/struct data types and falls back to DuckDB for these.
