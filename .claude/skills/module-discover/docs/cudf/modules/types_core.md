# Types / Core

**Status**: USED
**Path**: `cudf/types.hpp`, `cudf/version_config.hpp`
**Headers we include**: `cudf/types.hpp`, `cudf/version_config.hpp`

## Summary

The cuDF core type system defines fundamental types used throughout the library. Sirius maps DuckDB's type system to cuDF types for GPU execution, and uses `size_type` and `bitmask_type` extensively for column/row addressing and null mask management.

## API Reference

### `cudf::type_id` (enum)

**Header**: `cudf/types.hpp`

**Values used by Sirius**:
- `INT8`, `INT16`, `INT32`, `INT64` — signed integers
- `UINT8`, `UINT16`, `UINT32`, `UINT64` — unsigned integers
- `FLOAT32`, `FLOAT64` — floating point
- `BOOL8` — boolean
- `STRING` — variable-length string
- `TIMESTAMP_DAYS`, `TIMESTAMP_SECONDS`, `TIMESTAMP_MILLISECONDS`, `TIMESTAMP_MICROSECONDS`, `TIMESTAMP_NANOSECONDS` — temporal types
- `DECIMAL32`, `DECIMAL64`, `DECIMAL128` — fixed-point decimal
- `STRUCT`, `EMPTY` — structural types

**Our usage**:
- `src/gpu_columns.cpp` — Maps DuckDB `LogicalTypeId` to `cudf::type_id` for column construction
- `src/expression_executor/gpu_expression_translator.cpp` — Type dispatch for expression evaluation

### `cudf::data_type`

**Header**: `cudf/types.hpp`
```cpp
class data_type {
    data_type(type_id id);
    data_type(type_id id, int32_t scale);  // For DECIMAL types
    type_id id() const;
    int32_t scale() const;
};
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_operator.cpp` — Constructing target types for `binary_operation`
- `src/op/sirius_physical_ungrouped_aggregate.cpp` — Type construction for reduction results

### `cudf::size_type`

**Header**: `cudf/types.hpp`
**Type**: `int32_t`

Fundamental row-count and index type. Limits cuDF to ~2 billion rows per column.

**Our usage**: Pervasive — used in every operator for row counts, gather maps, and indexing.

### `cudf::bitmask_type`

**Header**: `cudf/types.hpp`
**Type**: `uint32_t`

Used for null validity bitmasks. Each bit represents one row's validity.

**Our usage**:
- `src/operator/gpu_materialize.cpp` — Validity mask construction
- `src/operator/gpu_physical_result_collector.cpp` — Null mask handling during result collection

### `cudf::null_equality` (enum)

**Header**: `cudf/types.hpp`
**Values**: `EQUAL`, `UNEQUAL`

**Our usage**:
- `src/cuda/cudf/cudf_join.cu` — Controls null matching behavior in joins

### `cudf::null_policy` (enum)

**Header**: `cudf/types.hpp`
**Values**: `INCLUDE`, `EXCLUDE`

**Our usage**:
- `src/cuda/cudf/cudf_groupby.cu` — Controls null inclusion in aggregation
- `src/cuda/cudf/cudf_aggregate.cu` — Null handling in reductions

### `cudf::order` / `cudf::null_order` (enums)

**Header**: `cudf/types.hpp`
- `cudf::order`: `ASCENDING`, `DESCENDING`
- `cudf::null_order`: `BEFORE`, `AFTER`

**Our usage**:
- `src/cuda/cudf/cudf_orderby.cu` — Sort direction and null placement
- `src/op/merge/gpu_merge_impl.cpp` — Merge order specification

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::thread_allocation_mode` | `cudf/types.hpp` | Thread-local allocation mode |
| `cudf::interpolation` | `cudf/types.hpp` | Interpolation method for quantiles |
| `cudf::mask_state` | `cudf/types.hpp` | Initial state for newly allocated masks |
