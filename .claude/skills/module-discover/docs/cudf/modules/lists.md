# Lists

**Status**: USED (minimal)
**Path**: `cudf/lists/`
**Headers we include**: `cudf/lists/count_elements.hpp`

## Summary

Minimal usage — only `count_elements` is used for counting elements in list columns during grouped aggregate merge operations.

## API Reference

### `cudf::lists::count_elements`

**Header**: `cudf/lists/count_elements.hpp`
```cpp
std::unique_ptr<column> count_elements(lists_column_view const& input, ...);
```

**Our usage**:
- `src/op/sirius_physical_grouped_aggregate_merge.cpp:24` — Counting elements in intermediate list aggregation results during merge

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::lists::contains()` | `contains.hpp` | Check if list contains value |
| `cudf::lists::extract_list_element()` | `extract.hpp` | Extract element from list |
| `cudf::lists::explode_outer()` | `explode.hpp` | Unnest lists to rows |
| `cudf::lists::sort_lists()` | `sorting.hpp` | Sort elements within lists |
