# Join

**Status**: USED
**Path**: `cudf/join/` (26.04+) or `cudf/join.hpp` (≤25.04)
**Headers we include**: `cudf/join/hash_join.hpp`, `cudf/join/join.hpp`, `cudf/join/conditional_join.hpp`, `cudf/join/mixed_join.hpp`, `cudf/join/distinct_hash_join.hpp`, `cudf/join/filtered_join.hpp`, `cudf/join.hpp`

## Summary

The join module is critical to Sirius — it implements all SQL join operations on GPU. Sirius uses hash joins for equi-joins (most common), conditional joins for non-equi predicates via AST expressions, and mixed joins combining both. Version-conditional includes handle the 25.04→26.04 header reorganization.

## API Reference

### `cudf::hash_join`

**Header**: `cudf/join/hash_join.hpp`
```cpp
class hash_join {
    hash_join(cudf::table_view const& build, null_equality compare_nulls,
              rmm::cuda_stream_view stream = {});

    // Probe methods return gather maps
    std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
              std::unique_ptr<rmm::device_uvector<size_type>>>
    inner_join(cudf::table_view const& probe, ...) const;

    left_join(cudf::table_view const& probe, ...) const;
    full_join(cudf::table_view const& probe, ...) const;
};
```

**Description**: Build-once, probe-multiple-times hash join. Returns pairs of gather-map vectors (left indices, right indices). `JoinNoMatch` sentinel marks unmatched rows.

**Our usage**:
- `src/cuda/cudf/cudf_join.cu` — Primary join implementation. Builds hash table on smaller relation, probes with larger.
- `src/op/sirius_physical_hash_join.cpp` — Hash join operator wrapping cudf_join

### `cudf::inner_join` / `cudf::left_join` / `cudf::full_join`

**Header**: `cudf/join/join.hpp`
```cpp
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
inner_join(table_view const& left_keys, table_view const& right_keys,
           null_equality compare_nulls = null_equality::EQUAL, ...);
```

**Description**: Convenience free functions (internally create hash_join). Sirius uses the `hash_join` class directly for better control over build/probe phases.

**Our usage**:
- `src/cuda/cudf/cudf_join.cu` — Fallback path for simple joins

### `cudf::conditional_inner_join` / `cudf::conditional_left_join` / `cudf::conditional_full_join`

**Header**: `cudf/join/conditional_join.hpp`
```cpp
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
conditional_inner_join(table_view const& left, table_view const& right,
                       ast::expression const& binary_predicate, ...);

std::unique_ptr<rmm::device_uvector<size_type>>
conditional_left_semi_join(table_view const& left, table_view const& right,
                           ast::expression const& binary_predicate, ...);
```

**Description**: Joins using arbitrary AST expressions as predicates. Used for non-equi joins (e.g., `a.x > b.y AND a.z < b.w`).

**Our usage**:
- `src/op/sirius_physical_nested_loop_join.cpp:34` — Nested loop join with conditional predicates translated to cuDF AST

### `cudf::mixed_inner_join` / `cudf::mixed_left_join`

**Header**: `cudf/join/mixed_join.hpp`
```cpp
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
mixed_inner_join(table_view const& left_equality, table_view const& right_equality,
                 table_view const& left_conditional, table_view const& right_conditional,
                 ast::expression const& binary_predicate, null_equality compare_nulls, ...);
```

**Description**: Combines hash-based equality matching with conditional predicates for optimal performance on complex join conditions.

**Our usage**:
- `src/op/sirius_physical_hash_join.cpp` — Used when join has both equality and inequality conditions

### `cudf::distinct_hash_join`

**Header**: `cudf/join/distinct_hash_join.hpp`
```cpp
class distinct_hash_join {
    distinct_hash_join(cudf::table_view const& build, null_equality compare_nulls, ...);
    // Methods for semi/anti joins with distinct semantics
};
```

**Our usage**:
- `src/include/cudf/cudf_utils.hpp:27` — Included for availability, used in semi/anti join paths

### `cudf::left_semi_join` / `cudf::left_anti_join`

**Header**: `cudf/join/filtered_join.hpp`
```cpp
std::unique_ptr<rmm::device_uvector<size_type>>
left_semi_join(table_view const& left_keys, table_view const& right_keys, ...);
```

**Our usage**:
- `src/cuda/cudf/cudf_join.cu` — Semi and anti join implementations

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::cross_join()` | `join.hpp` | Cartesian product of two tables |
| `cudf::sort_merge_join()` | `sort_merge_join.hpp` | Sort-merge join algorithm |
