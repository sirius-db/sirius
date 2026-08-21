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

/// One group of equality-delete files sharing the same key column schema.
struct EqualityDeleteGroup {
  /// GPU-resident deduplicated key table.
  std::unique_ptr<cudf::table> delete_table;
  std::vector<std::string> key_names;
  /// Populated when the delete file's footer carries them.
  std::vector<std::optional<int32_t>> key_field_ids;
  /// Build side = delete_table.
  std::unique_ptr<cudf::distinct_hash_join> hash_join;
  /// Applies only to data files with a STRICTLY lower DATA sequence number.
  ///
  /// ⚠️ Holds the MANIFEST's number, which is not that. Per the inheritance rule an entry takes
  /// its manifest's number only when its own is null AND its status is ADDED, so after a manifest
  /// rewrite these are inflated and the strict-inequality test flips both ways. This is one of the
  /// reasons read_iceberg_delete_data() refuses live equality entries.
  int64_t sequence_number{0};
};

/**
 * @brief Fully materialized Iceberg delete data for one table.
 *
 * All delete I/O happens at PLAN time, on internal connections each bracketed by their own
 * InternalQueryGuard — which is also why the memo is cleared on QueryEnd rather than inside the
 * execution window; see clear_iceberg_delete_data_cache(). Immutable after construction.
 */
struct IcebergDeleteData {
  /// V2 positional deletes and V3 deletion vectors merged: data_file_path -> sorted positions.
  std::unordered_map<std::string, std::vector<int64_t>> positional_deletes;

  /// One per unique (key-column schema, sequence number).
  std::vector<EqualityDeleteGroup> equality_delete_groups;

  /// Keyed as the manifest wrote the path. NOT the file's data sequence number -- see above.
  std::unordered_map<std::string, int64_t> data_file_manifest_sequence_numbers;

  [[nodiscard]] bool empty() const
  {
    return positional_deletes.empty() && equality_delete_groups.empty();
  }
};

/**
 * @brief Concatenate the delete files' rows, deduplicate them, and stand up the GPU hash join the
 *        scan probes. All @p views must share @p key_names.
 *
 * Exported only so this and the anti-join mask can be tested directly: the SQL route that reaches
 * them is declined at plan time, so no fixture can, and an inverted mask would stay green until
 * the route is switched on.
 */
EqualityDeleteGroup build_equality_group(std::vector<std::string> key_names,
                                         std::vector<std::optional<int32_t>> key_field_ids,
                                         std::vector<cudf::table_view> const& views);

/**
 * @brief Read and fully materialize every kind of delete for one table.
 *
 * Caller must suppress DuckDB side-effects (InternalQueryGuard).
 *
 * THROWS on any failure to read the manifests or delete files, so an empty result means "this
 * table has no deletes" and never "the deletes could not be read" — treating the second as the
 * first drops deletes silently and returns rows the table removed.
 *
 * @param metadata_ioctx Routes the equality-delete parquet and footer reads. Single-GPU is
 *                       sufficient (planning-time reads). Must outlive the call; nullptr throws.
 * @param snapshot_id    Latest if omitted.
 */
std::shared_ptr<const IcebergDeleteData> read_iceberg_delete_data(
  duckdb::ClientContext& context,
  std::string const& table_path,
  sirius::io::sirius_ioctx* metadata_ioctx,
  std::optional<uint64_t> snapshot_id = std::nullopt);

/**
 * @brief Drop everything the per-query delete-data cache is holding.
 *
 * MUST be called from the QueryEnd hook, not the execution window: a table declined at plan
 * time never opens one, and those entries are exactly the ones that would go stale. An entry
 * outliving its query could serve a previous snapshot's deletes; it also pins the GPU key table
 * and hash join its EqualityDeleteGroups own.
 */
void clear_iceberg_delete_data_cache();

/// Cache MISSES: reads that actually walked the manifests. Release builds compile the logging
/// out, so this is what lets a test assert the memo still collapses the repeat reads rather than
/// merely not corrupting them. Monotonic; tests take a delta around a query.
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
