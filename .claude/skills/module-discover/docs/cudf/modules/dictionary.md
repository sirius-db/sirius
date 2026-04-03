# Dictionary

**Status**: USED
**Path**: `cudf/dictionary/`
**Headers we include**: `cudf/dictionary/dictionary_column_view.hpp`, `cudf/dictionary/encode.hpp`

## Summary

Dictionary encoding is used as an optimization in merge and aggregate operations on string columns, reducing memory usage and improving comparison performance.

## API Reference

### `cudf::dictionary_column_view`

**Header**: `cudf/dictionary/dictionary_column_view.hpp`
```cpp
class dictionary_column_view : public column_view {
    column_view keys() const;     // Unique values
    column_view indices() const;  // Index into keys
    size_type keys_size() const;
};
```

**Our usage**:
- `src/op/merge/gpu_merge_impl.cpp:24` — Inspecting dictionary-encoded string columns
- `src/op/aggregate/gpu_aggregate_impl.cpp:23` — Dictionary column handling in aggregation

### `cudf::dictionary::encode`

**Header**: `cudf/dictionary/encode.hpp`
```cpp
std::unique_ptr<column> encode(column_view const& input, ...);
```

**Our usage**:
- `src/op/merge/gpu_merge_impl.cpp:25` — Encoding strings to dictionary before merge for performance
- `src/op/aggregate/gpu_aggregate_impl.cpp:24` — Dictionary encoding for groupby
