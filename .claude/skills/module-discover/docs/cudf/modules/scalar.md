# Scalar

**Status**: USED
**Path**: `cudf/scalar/`
**Headers we include**: `cudf/scalar/scalar.hpp`, `cudf/scalar/scalar_factories.hpp`

## Summary

Scalars represent single device-side values used in binary operations (column op scalar), aggregation results, and constant expression evaluation. Sirius constructs scalars for SQL literal values and extracts scalar results from reductions.

## API Reference

### `cudf::scalar` (base class)

**Header**: `cudf/scalar/scalar.hpp`
```cpp
class scalar {
    data_type type() const;
    bool is_valid() const;
};
```

### `cudf::numeric_scalar<T>`

**Header**: `cudf/scalar/scalar.hpp`
```cpp
template <typename T>
class numeric_scalar : public scalar {
    numeric_scalar(T value, bool is_valid = true, rmm::cuda_stream_view stream = {}, ...);
    T value(rmm::cuda_stream_view stream = {}) const;
};
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_comparison.cpp` — Constructing comparison constants
- `src/operator/gpu_physical_table_scan.cpp` — `cudf::numeric_scalar<bool>(false)` for null replacement
- `src/gpu_columns.cpp` — Extracting scalar results

### `cudf::string_scalar`

**Header**: `cudf/scalar/scalar.hpp`
```cpp
class string_scalar : public scalar {
    string_scalar(std::string const& value, bool is_valid = true, ...);
    std::string to_string(rmm::cuda_stream_view stream = {}) const;
};
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp` — String literal constants for LIKE patterns

### `cudf::fixed_point_scalar<T>`

**Header**: `cudf/scalar/scalar.hpp`
```cpp
template <typename T>  // T = numeric::decimal32, decimal64, decimal128
class fixed_point_scalar : public scalar {
    fixed_point_scalar(T value, bool is_valid = true, ...);
    T value(rmm::cuda_stream_view stream = {}) const;
    rep_type const* data() const;  // Raw representation
};
```

**Our usage**:
- `src/op/sirius_physical_ungrouped_aggregate.cpp:29` — DECIMAL aggregation results
- `src/expression_executor/gpu_expression_translator.cpp` — DECIMAL literal construction

### `cudf::timestamp_scalar<T>`

**Header**: `cudf/scalar/scalar.hpp`

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_comparison.cpp` — Timestamp comparison constants

### Scalar Factory Functions

**Header**: `cudf/scalar/scalar_factories.hpp`
```cpp
std::unique_ptr<scalar> make_default_constructed_scalar(data_type type, ...);
std::unique_ptr<scalar> make_fixed_width_scalar(data_type type, ...);
```

**Our usage**:
- `src/op/sirius_physical_ungrouped_aggregate.cpp:32` — Creating typed scalars for empty aggregation results

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::duration_scalar<T>` | `scalar.hpp` | Duration scalar types |
| `cudf::list_scalar` | `scalar.hpp` | List scalar type |
| `cudf::struct_scalar` | `scalar.hpp` | Struct scalar type |
