# Copying

**Status**: USED
**Path**: `cudf/copying.hpp`, `cudf/concatenate.hpp`, `cudf/detail/contiguous_split.hpp`
**Headers we include**: `cudf/copying.hpp`, `cudf/concatenate.hpp`, `cudf/detail/contiguous_split.hpp`

## Summary

The copying module provides the data movement primitives used throughout Sirius. `gather` is the most frequently used cuDF function — it materializes join results and sorted output by selecting rows via index arrays. `slice` implements LIMIT, and `concatenate` combines pipeline outputs.

## API Reference

### `cudf::gather`

**Header**: `cudf/copying.hpp`
```cpp
std::unique_ptr<table> gather(table_view const& source_table,
                               column_view const& gather_map,
                               out_of_bounds_policy policy = out_of_bounds_policy::DONT_CHECK, ...);
```

**Description**: Selects rows from source table according to gather_map indices. `out_of_bounds_policy::NULLIFY` produces nulls for invalid indices (used in outer joins).

**Our usage** (very high frequency, 10+ sites):
- `src/op/sirius_physical_hash_join.cpp` — Materializes join results from gather maps
- `src/cuda/cudf/cudf_join.cu` — Post-join gather
- `src/cuda/cudf/cudf_orderby.cu` — Materializes sorted output
- `src/op/sirius_physical_top_n.cpp` — Top-N row selection
- `src/op/order/gpu_order_impl.cpp` — Sort result materialization

### `cudf::slice`

**Header**: `cudf/copying.hpp`
```cpp
std::vector<table_view> slice(table_view const& input,
                               host_span<size_type const> indices, ...);
```

**Description**: Zero-copy slicing — returns views into the original table at specified [start, end) ranges.

**Our usage**:
- `src/op/sirius_physical_limit.cpp:21` — SQL LIMIT implementation
- `src/cuda/cudf/cudf_orderby.cu` — Partial sort optimization (top-K)

### `cudf::concatenate`

**Header**: `cudf/concatenate.hpp`
```cpp
std::unique_ptr<table> concatenate(host_span<table_view const> tables_to_concat, ...);
std::unique_ptr<column> concatenate(host_span<column_view const> columns_to_concat, ...);
```

**Description**: Vertically concatenates tables/columns. Allocates new memory and copies.

**Our usage**:
- `src/op/sirius_physical_top_n.cpp:25` — Combining partial top-N results across batches
- `src/op/sirius_physical_sort_sample.cpp:25` — Combining sort samples
- `src/op/merge/gpu_merge_impl.cpp:23` — Concatenating before merge when appropriate

### `cudf::scatter`

**Header**: `cudf/copying.hpp`
```cpp
std::unique_ptr<table> scatter(table_view const& source, column_view const& scatter_map,
                                table_view const& target, ...);
```

**Our usage**:
- `src/op/sirius_physical_nested_loop_join.cpp` — Scatter for mark join results

### `cudf::copy_bitmask`

**Header**: `cudf/copying.hpp`
```cpp
rmm::device_buffer copy_bitmask(column_view const& view, ...);
```

**Our usage**:
- `src/operator/gpu_physical_result_collector.cpp` — Extracting validity masks

### `cudf::contiguous_split`

**Header**: `cudf/detail/contiguous_split.hpp`
```cpp
std::vector<packed_columns> contiguous_split(table_view const& input,
                                              std::vector<size_type> const& splits, ...);
```

**Our usage**:
- `src/include/memory/host_table_utils.hpp:34` — Serializing tables for GPU↔CPU transfer via cuCascade

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::reverse()` | `copying.hpp` | Reverse row order |
| `cudf::repeat()` | `copying.hpp` | Repeat rows N times |
| `cudf::shift()` | `copying.hpp` | Shift column values |
| `cudf::split()` | `copying.hpp` | Split table at indices (like slice but returns tables) |
| `cudf::copy_if_else()` | `copying.hpp` | Conditional column selection |
| `cudf::boolean_mask_scatter()` | `copying.hpp` | Scatter with boolean mask |
