# Sorting, Merge & Search

**Status**: USED
**Path**: `cudf/sorting.hpp`, `cudf/merge.hpp`, `cudf/search.hpp`
**Headers we include**: `cudf/sorting.hpp`, `cudf/merge.hpp`, `cudf/search.hpp`

## Summary

Sirius uses sorting for ORDER BY, merge for combining pre-sorted pipeline outputs, and search for partitioning sorted data. The primary pattern is `sorted_order()` → `gather()` rather than in-place sort.

## API Reference

### `cudf::sorted_order`

**Header**: `cudf/sorting.hpp`
```cpp
std::unique_ptr<column> sorted_order(table_view const& input,
                                      std::vector<order> const& column_order = {},
                                      std::vector<null_order> const& null_precedence = {}, ...);
```

**Description**: Returns a column of row indices that would sort the input table. Used with `gather()` to produce sorted output.

**Our usage**:
- `src/cuda/cudf/cudf_orderby.cu` — Primary ORDER BY implementation
- `src/op/sirius_physical_top_n.cpp:28` — Sort indices for top-N selection

### `cudf::stable_sorted_order`

**Header**: `cudf/sorting.hpp`
```cpp
std::unique_ptr<column> stable_sorted_order(table_view const& input, ...);
```

**Description**: Like `sorted_order` but preserves relative order of equal elements.

**Our usage**:
- `src/cuda/cudf/cudf_orderby.cu` — Used when stable sort is required

### `cudf::merge`

**Header**: `cudf/merge.hpp`
```cpp
std::unique_ptr<table> merge(std::vector<table_view> const& tables_to_merge,
                              std::vector<size_type> const& key_cols,
                              std::vector<order> const& column_order,
                              std::vector<null_order> const& null_precedence = {}, ...);
```

**Description**: Merges multiple pre-sorted tables into a single sorted table. More efficient than concatenate+sort when inputs are already sorted.

**Our usage**:
- `src/op/merge/gpu_merge_impl.cpp:26` — Merging sorted pipeline outputs in multi-stream execution

### `cudf::lower_bound` / `cudf::upper_bound`

**Header**: `cudf/search.hpp`
```cpp
std::unique_ptr<column> lower_bound(table_view const& haystack,
                                     table_view const& needles,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence, ...);
```

**Description**: Binary search in a sorted table. Returns indices where needles would be inserted.

**Our usage**:
- `src/op/sirius_physical_sort_partition.cpp:25` — Partitioning sorted data at sample boundaries for parallel merge

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::sort()` | `sorting.hpp` | In-place sort (returns new table) |
| `cudf::sort_by_key()` | `sorting.hpp` | Sort one table by another's keys |
| `cudf::rank()` | `sorting.hpp` | Compute row ranks |
| `cudf::is_sorted()` | `sorting.hpp` | Check if table is sorted |
| `cudf::segmented_sort_by_key()` | `sorting.hpp` | Sort within segments |
| `cudf::contains()` | `search.hpp` | Check if value exists in column |
