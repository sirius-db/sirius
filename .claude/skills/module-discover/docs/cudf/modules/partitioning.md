# Partitioning

**Status**: USED
**Path**: `cudf/partitioning.hpp`
**Headers we include**: `cudf/partitioning.hpp`

## Summary

Hash-based table partitioning for parallel pipeline execution. Distributes rows across partitions based on hash of key columns.

## API Reference

### `cudf::hash_partition`

**Header**: `cudf/partitioning.hpp`
```cpp
std::pair<std::unique_ptr<table>, std::vector<size_type>>
hash_partition(table_view const& input,
               std::vector<size_type> const& columns_to_hash,
               int num_partitions, ...);
```

**Description**: Returns (partitioned_table, partition_offsets). Rows with same hash key end up in same partition.

**Our usage**:
- `src/op/partition/gpu_partition_impl.cpp:21` — Repartitioning data for parallel join/aggregate execution
