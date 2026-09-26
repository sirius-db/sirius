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
#include "op/scan/table_scan/scan_contract.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table.hpp>

#include <duckdb/main/client_context.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class ioctx;
}  // namespace sirius::io

namespace sirius::op::scan {

/// One file entry from an Iceberg manifest: equality-delete files and V3 deletion vectors.
struct IcebergDeleteFileEntry {
  std::string file_path;
  int content{0};                     // 0=DATA, 1=POSITION_DELETES, 2=EQUALITY_DELETES
  std::string file_format;            // "parquet" or "puffin" (always lowercase)
  std::string referenced_data_file;   // data file this DV applies to (V3, empty if absent)
  int64_t content_offset{-1};         // byte offset in Puffin file (V3, -1 if absent)
  int64_t content_size_in_bytes{-1};  // byte length of DV blob (V3, -1 if absent)
  int64_t sequence_number{0};         // manifest entry sequence number (for eq delete filtering)
  int64_t record_count{-1};           // deleted positions the manifest claims (V3, -1 if absent)

  /// Requires file_format already lowercased by the reader. Format alone: an entry that IS a
  /// deletion vector but describes itself incompletely must be rejected, not reclassified as
  /// something other than a deletion vector and skipped.
  [[nodiscard]] bool is_deletion_vector() const { return file_format == "puffin"; }

  /// Whether the manifest gave this vector a locatable blob.
  [[nodiscard]] bool has_complete_descriptor() const
  {
    return content_offset >= 0 && content_size_in_bytes > 0;
  }

  /// Whether this vector's decode is bounded. `record_count` is a required manifest field: absent,
  /// both cardinality cross-checks compare against nothing and the Roaring expansion is unbounded.
  [[nodiscard]] bool has_decodable_record_count() const;
};

struct iceberg_delete_discovery {
  std::vector<std::string> positional_delete_files;
  std::vector<IcebergDeleteFileEntry> equality_delete_entries;
  std::vector<IcebergDeleteFileEntry> deletion_vector_entries;
  /// From the data manifests; equality deletes need them to test applicability.
  std::unordered_map<std::string, int64_t> data_file_manifest_sequence_numbers;
};

struct inventory_result {
  std::optional<iceberg_delete_inventory> inventory;
  uint64_t equality_count = 0;
};
struct inventory_query_error : std::runtime_error {
  inventory_query_error(std::string const& message, bool missing)
    : std::runtime_error(message), missing_result(missing)
  {
  }
  bool missing_result;
};
inventory_result read_delete_inventory(duckdb::ClientContext&,
                                       std::string const&,
                                       std::optional<int64_t>,
                                       std::string_view injection = {});
iceberg_delete_discovery discover_from_manifests(duckdb::ClientContext&,
                                                 std::string const&,
                                                 iceberg_delete_inventory&&);

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
  /// reasons load_delete_payload() refuses live equality entries.
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
 * @brief Load or reuse immutable delete payloads from this scan's supplied discovery.
 *
 * Caller must suppress DuckDB side-effects (InternalQueryGuard).
 *
 * THROWS on any failure to read the manifests or delete files, so an empty result means "this
 * table has no deletes" and never "the deletes could not be read" — treating the second as the
 * first drops deletes silently and returns rows the table removed.
 *
 * @param metadata_ioctx Routes the equality-delete parquet and footer reads. Single-GPU is
 *                       sufficient (planning-time reads). Must outlive the call; nullptr throws.
 * @param snapshot_id    The snapshot the SCAN was bound to. Callers on the GPU path always pass
 *                       one: an unpinned iceberg_scan is declined at plan time precisely so that
 *                       this pass cannot resolve "current" independently and pair one snapshot's
 *                       data files with another's deletes. It identifies the payload cache entry;
 *                       this function never queries the inventory.
 */
std::shared_ptr<const IcebergDeleteData> load_delete_payload(
  duckdb::ClientContext& context,
  std::string const& table_path,
  sirius::io::ioctx* metadata_ioctx,
  std::optional<uint64_t> snapshot_id,
  iceberg_delete_discovery const& discovery);

/**
 * @brief Drop everything the per-query delete-data cache is holding.
 *
 * MUST be called from the QueryEnd hook, not the execution window: a table declined at plan
 * time never opens one, and those entries are exactly the ones that would go stale. An entry
 * outliving its query could serve a previous snapshot's deletes; it also pins the GPU key table
 * and hash join its EqualityDeleteGroups own.
 */
void clear_iceberg_delete_data_cache();

/// Payload cache misses, independent of inventory walks. Release builds compile the logging
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
