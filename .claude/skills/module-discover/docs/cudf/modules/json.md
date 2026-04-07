# JSON

**Status**: UNUSED
**Path**: `cudf/json/`

## Summary
JSON column support for querying semi-structured JSON data stored in columns.

## Key APIs
- `cudf::json::get_json_object()` — Extract values from JSON strings using JSONPath

## Potential Relevance
Not applicable to current TPC-H/analytical workloads. Could be useful for JSON data processing if Sirius expands to semi-structured data support.
