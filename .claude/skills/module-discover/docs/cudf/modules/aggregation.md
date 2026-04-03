# Aggregation

**Status**: USED
**Path**: `cudf/aggregation.hpp`, `cudf/groupby.hpp`, `cudf/reduction.hpp`, `cudf/reduction/`, `cudf/detail/aggregation/`
**Headers we include**: `cudf/aggregation.hpp`, `cudf/groupby.hpp`, `cudf/reduction.hpp`, `cudf/reduction/distinct_count.hpp`, `cudf/reduction/approx_distinct_count.hpp`, `cudf/detail/aggregation/aggregation.hpp`

## Summary

Sirius uses cuDF's aggregation module for both grouped (GROUP BY) and ungrouped aggregations. The groupby API processes multiple aggregation requests in a single pass. Factory functions create typed aggregation objects (SUM, COUNT, MIN, MAX, MEAN, NUNIQUE). Detail headers are used for version-compatibility.

## API Reference

### Aggregation Factory Functions

**Header**: `cudf/aggregation.hpp`
```cpp
template <typename Base = aggregation>
std::unique_ptr<Base> make_sum_aggregation();
std::unique_ptr<Base> make_mean_aggregation();
std::unique_ptr<Base> make_min_aggregation();
std::unique_ptr<Base> make_max_aggregation();
std::unique_ptr<Base> make_count_aggregation(null_policy policy = null_policy::EXCLUDE);
std::unique_ptr<Base> make_nunique_aggregation(null_policy policy = null_policy::EXCLUDE);
std::unique_ptr<Base> make_any_aggregation();
```

**Description**: Template parameter `Base` is `cudf::groupby_aggregation` for groupby or `cudf::reduce_aggregation` for reductions.

**Our usage**:
- `src/cuda/cudf/cudf_groupby.cu` — Creates groupby aggregation objects: `make_sum_aggregation<groupby_aggregation>()`
- `src/cuda/cudf/cudf_aggregate.cu` — Creates reduction aggregation objects
- `src/op/sirius_physical_ungrouped_aggregate.cpp` — `make_sum_aggregation<reduce_aggregation>()`

### `cudf::groupby::groupby`

**Header**: `cudf/groupby.hpp`
```cpp
class groupby {
    groupby(table_view const& keys, null_policy include_null_keys = null_policy::EXCLUDE,
            sorted keys_are_sorted = sorted::NO, ...);

    std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
    aggregate(host_span<aggregation_request const> requests, ...);
};

struct aggregation_request {
    column_view values;
    std::vector<std::unique_ptr<groupby_aggregation>> aggregations;
};

struct aggregation_result {
    std::vector<std::unique_ptr<column>> results;
};
```

**Description**: Groups rows by key columns, then applies multiple aggregations per value column. Returns (unique_keys_table, vector_of_results).

**Our usage**:
- `src/cuda/cudf/cudf_groupby.cu` — Primary grouped aggregation. Constructs `aggregation_request` per value column, calls `aggregate()`.
- `src/op/aggregate/gpu_aggregate_impl.cpp` — Groupby operator implementation

### `cudf::reduce`

**Header**: `cudf/reduction.hpp`
```cpp
std::unique_ptr<scalar> reduce(column_view const& col,
                                reduce_aggregation const& agg,
                                data_type output_dtype, ...);
```

**Description**: Reduces an entire column to a single scalar value.

**Our usage**:
- `src/op/sirius_physical_ungrouped_aggregate.cpp:30` — Ungrouped aggregations (e.g., `SELECT SUM(x) FROM t`)
- `src/expression_executor/specializations/gpu_execute_case.cpp:24` — Reduction in CASE expression evaluation

### `cudf::distinct_count`

**Header**: `cudf/reduction/distinct_count.hpp` (26.04+)
```cpp
size_type distinct_count(column_view const& input, null_policy null_handling, nan_policy nan_handling, ...);
```

**Our usage**:
- `src/include/cudf/cudf_utils.hpp:42` — Conditionally included for COUNT DISTINCT

### `cudf::approx_distinct_count`

**Header**: `cudf/reduction/approx_distinct_count.hpp`
```cpp
size_type approx_distinct_count(column_view const& input, ...);
```

**Our usage**:
- `src/op/merge/gpu_merge_impl.cpp:27` — Approximate distinct count for merge optimization decisions
- `src/op/aggregate/gpu_aggregate_impl.cpp:25` — Cardinality estimation

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::make_variance_aggregation()` | `aggregation.hpp` | Variance computation |
| `cudf::make_std_aggregation()` | `aggregation.hpp` | Standard deviation |
| `cudf::make_median_aggregation()` | `aggregation.hpp` | Median value |
| `cudf::make_quantile_aggregation()` | `aggregation.hpp` | Quantile computation |
| `cudf::make_collect_list_aggregation()` | `aggregation.hpp` | Collect values into list |
| `cudf::make_rank_aggregation()` | `aggregation.hpp` | Row ranking |
| `cudf::groupby::scan()` | `groupby.hpp` | Grouped scan (running aggregation) |
| `cudf::segmented_reduce()` | `reduction.hpp` | Segmented reduction |
| `cudf::scan()` | `reduction.hpp` | Prefix scan (cumulative aggregation) |
