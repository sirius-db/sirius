# cuDF (libcudf) — Module Reference

**Version**: 26.04+ (with 25.04 backward compatibility via `CUDF_VERSION_NUM` macro)
**Location**: `/home/felipe/miniconda3/envs/libcudf-env/include/cudf/`
**Namespace**: `cudf`
**Build Integration**: Linked via CMake as `cudf::cudf`

## Module Map

| Module | Status | Description | Key APIs Used |
|--------|--------|-------------|---------------|
| [types_core](modules/types_core.md) | USED | Core type system: type_id, data_type, size_type, bitmask_type | `type_id`, `data_type`, `size_type` |
| [table](modules/table.md) | USED | Table/column data structures (owning + views) | `table`, `table_view`, `column_view`, `column_factories` |
| [join](modules/join.md) | USED | Hash, conditional, mixed, and distinct joins | `hash_join`, `inner_join`, `conditional_join`, `mixed_join` |
| [aggregation](modules/aggregation.md) | USED | Groupby, reduction, and aggregation operations | `groupby`, `reduce`, `make_sum_aggregation` |
| [sorting](modules/sorting.md) | USED | Sort, merge, and search operations | `sorted_order`, `merge`, `lower_bound` |
| [copying](modules/copying.md) | USED | Gather, scatter, slice, concatenate | `gather`, `slice`, `concatenate`, `scatter` |
| [unary_binary](modules/unary_binary.md) | USED | Unary/binary element-wise operations, type casting | `binary_operation`, `cast`, `unary_operation` |
| [strings](modules/strings.md) | USED | GPU string operations (regex, find, slice, etc.) | `contains`, `like`, `slice_strings`, `regex_program` |
| [io](modules/io.md) | USED | Parquet I/O, datasource abstraction, hybrid scan | `parquet_reader_options`, `datasource`, `hybrid_scan_reader` |
| [ast](modules/ast.md) | USED | Expression trees for conditional joins | `tree`, `operation`, `column_reference`, `ast_operator` |
| [scalar](modules/scalar.md) | USED | Device-side scalar values for expressions | `numeric_scalar`, `string_scalar`, `fixed_point_scalar` |
| [stream_compaction](modules/stream_compaction.md) | USED | Duplicate elimination, filtering | `drop_duplicates` |
| [utilities](modules/utilities.md) | USED | Stream, memory resource, type dispatch | `get_default_stream`, `set_current_device_resource`, `type_dispatcher` |
| [dictionary](modules/dictionary.md) | USED | Dictionary-encoded columns for merge optimization | `dictionary_column_view`, `encode` |
| [fixed_point](modules/fixed_point.md) | USED | DECIMAL type support | `fixed_point`, `scale_type` |
| [lists](modules/lists.md) | USED | List column operations (count_elements only) | `count_elements` |
| [datetime](modules/datetime.md) | USED | Date/time extraction and arithmetic | `extract_year`, `extract_month` |
| [partitioning](modules/partitioning.md) | USED | Hash-based table partitioning | `hash_partition` |
| [transform](modules/transform.md) | USED | Element-wise transforms with expressions | `compute_column` |
| [hashing](modules/hashing.md) | UNUSED | Hash function implementations | — |
| [rolling](modules/rolling.md) | UNUSED | Rolling window aggregations | — |
| [json](modules/json.md) | UNUSED | JSON column support | — |
| [structs](modules/structs.md) | UNUSED | Struct/record column support | — |
| [tdigest](modules/tdigest.md) | UNUSED | T-Digest approximate percentiles | — |
| [labeling](modules/labeling.md) | UNUSED | Bin labeling/categorization | — |

## Our Usage Summary

We use **19 of 25** modules. Primary integration points:

- **Query execution**: Binary/unary ops for filter/projection expressions, AST for conditional joins
- **Join processing**: Hash joins (inner, left, right, full), conditional joins via AST, mixed joins
- **Aggregation**: Groupby with SUM/COUNT/MIN/MAX/MEAN/NUNIQUE, ungrouped reductions
- **Sort/Order**: Lexicographic sorting with gather, merge for sorted streams, top-N via sort+slice
- **Data I/O**: Parquet reading via hybrid_scan experimental API, custom datasource implementations
- **String processing**: LIKE, regex matching, substring, length operations
- **Memory management**: RMM device resource control, pinned memory configuration, stream management
- **Type system**: cuDF type_id mapping to/from DuckDB types, DECIMAL via fixed_point

## Version Compatibility

Sirius handles cuDF API differences between 25.04 and 26.04+ via a version macro:
```cpp
#include <cudf/version_config.hpp>
#define CUDF_VERSION_NUM (CUDF_VERSION_MAJOR * 100 + CUDF_VERSION_MINOR)
```

Key version differences:
- **Join headers**: Unified `cudf/join.hpp` in ≤25.04, split into `cudf/join/*.hpp` in 26.04+
- **Aggregation headers**: `cudf/aggregation.hpp` in ≤25.04, split headers in 26.04+
- **Stream compaction**: Detail API access patterns differ
- **Reduction**: `distinct_count` moved to `cudf/reduction/distinct_count.hpp` in 26.04+

## Files That Reference cuDF

| Source File | Modules Used | Key APIs |
|-------------|-------------|----------|
| `src/include/cudf/cudf_utils.hpp` | types, table, join, aggregation, sorting, copying, scalar, stream_compaction | Central cuDF include hub |
| `src/cuda/cudf/cudf_join.cu` | join, copying | `hash_join`, `gather`, inner/left/right/full joins |
| `src/cuda/cudf/cudf_groupby.cu` | aggregation, stream_compaction | `groupby`, aggregation factories |
| `src/cuda/cudf/cudf_orderby.cu` | sorting, copying | `sorted_order`, `gather` |
| `src/cuda/cudf/cudf_aggregate.cu` | aggregation | Reduction aggregation |
| `src/expression_executor/specializations/gpu_execute_operator.cpp` | unary_binary, search | `binary_operation`, `cast` |
| `src/expression_executor/specializations/gpu_execute_comparison.cpp` | unary_binary, scalar | `binary_operation`, scalar construction |
| `src/expression_executor/specializations/gpu_execute_function.cpp` | strings, unary_binary, scalar, datetime | String functions, type casting |
| `src/expression_executor/gpu_expression_translator.cpp` | ast, types | AST tree construction |
| `src/op/sirius_physical_hash_join.cpp` | join, copying, unary_binary | `hash_join`, `gather`, `cast` |
| `src/op/sirius_physical_nested_loop_join.cpp` | join, ast, copying | `conditional_join`, AST expressions |
| `src/op/sirius_physical_ungrouped_aggregate.cpp` | aggregation, scalar, fixed_point | `reduce`, scalar extraction |
| `src/op/scan/parquet_scan_task.cpp` | io | Parquet reading, hybrid_scan |
| `src/op/aggregate/gpu_aggregate_impl.cpp` | aggregation, dictionary | Groupby, dictionary encoding |
| `src/op/merge/gpu_merge_impl.cpp` | sorting, aggregation, dictionary, lists, copying | `merge`, dictionary, `concatenate` |
| `src/op/sirius_physical_top_n.cpp` | sorting, copying | `sorted_order`, `gather`, `concatenate` |
| `src/op/sirius_physical_sort_partition.cpp` | sorting, search | `lower_bound`, `upper_bound` |
| `src/op/partition/gpu_partition_impl.cpp` | partitioning, unary_binary | `hash_partition` |
| `src/memory/sirius_memory_reservation_manager.cpp` | utilities | `set_current_device_resource` |
| `src/sirius_context.cpp` | utilities | Pinned memory configuration |
| `src/include/data/host_parquet_representation.hpp` | io | `datasource`, `hybrid_scan_reader` |
| `src/include/memory/host_table_utils.hpp` | table, utilities | `contiguous_split`, `host_span` |
