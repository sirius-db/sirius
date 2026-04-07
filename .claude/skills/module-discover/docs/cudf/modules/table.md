# Table & Column

**Status**: USED
**Path**: `cudf/table/`, `cudf/column/`
**Headers we include**: `cudf/table/table.hpp`, `cudf/table/table_view.hpp`, `cudf/column/column.hpp`, `cudf/column/column_view.hpp`, `cudf/column/column_factories.hpp`

## Summary

Tables and columns are the fundamental data structures in cuDF. Sirius converts DuckDB data to cuDF tables for GPU processing, manipulates them through operators, and converts results back. The owning vs. view pattern is central to zero-copy data passing.

## API Reference

### `cudf::table`

**Header**: `cudf/table/table.hpp`
```cpp
class table {
    table(std::vector<std::unique_ptr<column>>&& columns);
    table(table_view view, rmm::cuda_stream_view stream = {}, rmm::device_async_resource_ref mr = {});
    table_view view() const;
    mutable_table_view mutable_view();
    size_type num_columns() const;
    size_type num_rows() const;
    std::vector<std::unique_ptr<column>> release();
};
```

**Our usage**:
- `src/cuda/cudf/cudf_join.cu` — Join results returned as tables
- `src/op/sirius_physical_top_n.cpp` — Intermediate table construction
- `src/cuda/cudf/cudf_groupby.cu` — Groupby keys table

### `cudf::table_view`

**Header**: `cudf/table/table_view.hpp`
```cpp
class table_view {
    table_view(std::vector<column_view> const& columns);
    column_view column(size_type index) const;
    size_type num_columns() const;
    size_type num_rows() const;
    table_view select(std::vector<size_type> const& column_indices) const;
};
```

**Our usage**: Pervasive — passed to virtually every cuDF operation as input.

### `cudf::column`

**Header**: `cudf/column/column.hpp`
```cpp
class column {
    column(column_view view, rmm::cuda_stream_view stream = {}, rmm::device_async_resource_ref mr = {});
    column_view view() const;
    mutable_column_view mutable_view();
    data_type type() const;
    size_type size() const;
    size_type null_count() const;
    struct contents { std::unique_ptr<rmm::device_buffer> data; ... };
    contents release();
};
```

**Our usage**:
- `src/expression_executor/specializations/*.cpp` — Expression results as columns
- `src/cuda/operator/empty_str_check.cu` — Column construction from device data

### `cudf::column_view`

**Header**: `cudf/column/column_view.hpp`
```cpp
class column_view {
    column_view(data_type type, size_type size, void const* data,
                bitmask_type const* null_mask = nullptr, size_type null_count = UNKNOWN_NULL_COUNT,
                size_type offset = 0, std::vector<column_view> const& children = {});
    data_type type() const;
    size_type size() const;
    template <typename T> T const* data() const;
    template <typename T> T const* head() const;
    bitmask_type const* null_mask() const;
    size_type null_count() const;
    size_type offset() const;
    size_type num_children() const;
    column_view child(size_type index) const;
};
```

**Our usage**:
- `src/gpu_columns.cpp` — Constructing column_views from GPU buffer data
- `src/expression_executor/specializations/gpu_execute_operator.cpp` — Accessing column data for operations

### `cudf::make_empty_column(data_type)`

**Header**: `cudf/column/column_factories.hpp`
```cpp
std::unique_ptr<column> make_empty_column(data_type type);
std::unique_ptr<column> make_numeric_column(data_type type, size_type size, mask_state state = UNALLOCATED, ...);
std::unique_ptr<column> make_fixed_width_column(data_type type, size_type size, mask_state state = UNALLOCATED, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp:31` — Creating empty result columns
- `src/op/sirius_physical_ungrouped_aggregate.cpp:27` — Empty columns for empty aggregation results

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::make_strings_column()` | `column_factories.hpp` | Create string column from device data |
| `cudf::make_lists_column()` | `column_factories.hpp` | Create list column |
| `cudf::make_structs_column()` | `column_factories.hpp` | Create struct column |
| `cudf::mutable_table_view` | `table_view.hpp` | Mutable non-owning table view |
