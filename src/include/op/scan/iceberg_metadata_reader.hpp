/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "io/types.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table.hpp>

#include <duckdb/main/client_context.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class sirius_ioctx;
}  // namespace sirius::io

namespace sirius::op::scan {

/**
 * @brief Fully materialized Iceberg delete data for one table.
 *
 * All I/O (positional deletes, deletion vectors, equality deletes) is
 * performed once during sirius_engine::initialize() under a single
 * InternalQueryGuard.  This struct is immutable after construction and
 * is shared (via shared_ptr<const>) across the operator, task creator,
 * and delete filters.
 */
/// One group of equality-delete files sharing the same key column schema.
struct EqualityDeleteGroup {
  /// GPU-resident deduplicated key table.
  std::unique_ptr<cudf::table> delete_table;
  /// Column names (parallel to table columns).
  std::vector<std::string> key_names;
  /// Iceberg field IDs for each key column (populated when available).
  std::vector<std::optional<int32_t>> key_field_ids;
  /// Pre-built GPU hash join (build side = delete_table).
  std::unique_ptr<cudf::distinct_hash_join> hash_join;
  /// Sequence number of the delete file(s) in this group.
  /// Per Iceberg spec, this group only applies to data files with
  /// data_file_sequence_number < this value (strictly lower).
  int64_t sequence_number{0};
};

struct IcebergDeleteData {
  /// V2 positional deletes + V3 deletion vectors (merged, sorted).
  /// Key: data_file_path, Value: sorted deleted row positions.
  /// Stored on CPU (tiny metadata).
  std::unordered_map<std::string, std::vector<int64_t>> positional_deletes;

  /// V2 equality-delete groups (one per unique key-column schema).
  /// Supports heterogeneous delete files (e.g., delete by "name" vs "name+bir").
  std::vector<EqualityDeleteGroup> equality_delete_groups;

  /// Per-data-file sequence numbers (for equality delete seq filtering).
  /// Key: data_file_path, Value: sequence number from manifest entry.
  std::unordered_map<std::string, int64_t> data_file_sequence_numbers;

  /// True if there are no deletes to apply (V1 table or empty manifests).
  [[nodiscard]] bool empty() const
  {
    return positional_deletes.empty() && equality_delete_groups.empty();
  }
};

/**
 * @brief Read and fully materialize Iceberg delete data for the given table.
 *
 * Consolidates all delete-related I/O into one call:
 *   1. Discovers delete file paths via iceberg_metadata(), plus a read_avro pass
 *      over a manifest when a V3 deletion vector needs its Puffin offsets.
 *   2. Reads V2 positional-delete parquet files (CPU via DuckDB).
 *   3. Reads V3 deletion vectors from Puffin files (CPU).
 *   4. Reads V2 equality-delete parquet files (GPU via cuDF).
 *   5. Deduplicates equality deletes and builds the GPU hash join.
 *
 * Caller must ensure DuckDB side-effects are suppressed (InternalQueryGuard).
 *
 * THROWS on any failure to read the manifests or delete files. An empty result therefore
 * means "this table has no deletes", never "the deletes could not be read" — the two must not
 * be confused, because acting on the second as if it were the first drops deletes silently and
 * returns rows the table logically removed.
 *
 * @param context        DuckDB client context for running iceberg_metadata()
 *                       and reading positional-delete parquet files.
 * @param table_path     The Iceberg table path passed to iceberg_scan().
 * @param metadata_ioctx Non-owning sirius_ioctx for routing parquet reads
 *                       (V2 equality-delete files + footer extraction). A
 *                       single GPU's ioctx is sufficient — these are
 *                       planning-time reads, not on the multi-GPU column-
 *                       chunk hot path. Multi-GPU residency for iceberg
 *                       metadata is deferred. Must outlive the call and be
 *                       non-null; nullptr throws.
 * @param snapshot_id    Optional Iceberg snapshot id (latest if omitted).
 * @return Shared pointer to immutable delete data.
 */
std::shared_ptr<const IcebergDeleteData> read_iceberg_delete_data(
  duckdb::ClientContext& context,
  std::string const& table_path,
  sirius::io::sirius_ioctx* metadata_ioctx,
  std::optional<uint64_t> snapshot_id = std::nullopt);

/**
 * @brief Drop everything the per-query delete-data cache is holding.
 *
 * MUST be called from the query lifecycle (QueryBegin and QueryEnd). Two reasons, and the
 * second is not optional:
 *
 * 1. Correctness. The cache exists only to collapse the repeated reads WITHIN one query
 *    (see read_iceberg_delete_data). An entry that outlives its query could answer for a
 *    table that has since been committed to, and returning a previous snapshot's deletes
 *    means returning rows the table has removed.
 * 2. Resources. IcebergDeleteData owns EqualityDeleteGroups, and those hold a GPU key table
 *    and a prebuilt hash join. Holding them past the query pins GPU memory.
 */
void clear_iceberg_delete_data_cache();

/**
 * @brief Count of delete-data reads that actually walked the manifests (cache MISSES).
 *
 * The point of the cache is that one query performs this work once rather than repeatedly.
 * Release builds compile INFO/DEBUG logging out, so a log line cannot demonstrate that, and
 * a passing test only shows the memo did not corrupt results — not that it eliminated any
 * work. This counter makes the claim measurable, and turns "the memo silently stopped
 * hitting" into a test failure instead of a cost that quietly returns.
 *
 * Monotonic for the process lifetime; tests take a delta around a query.
 */
uint64_t iceberg_delete_data_uncached_read_count();

/**
 * @brief Extract a column_name → field_id map from a parquet FileMetaData.
 *
 * Walks the flattened schema depth-first and collects field IDs for leaf
 * columns (num_children == 0).  Columns without a field_id are omitted.
 *
 * @param file_meta  Parquet file metadata (from read_parquet_footers or
 *                   parquet_scan_task_global_state::_file_metadatas).
 * @return Map of column name to Iceberg field ID.
 */
std::unordered_map<std::string, int32_t> extract_field_id_map(
  cudf::io::parquet::FileMetaData const& file_meta);

}  // namespace sirius::op::scan
