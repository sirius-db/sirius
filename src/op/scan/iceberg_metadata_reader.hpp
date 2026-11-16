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

#include "op/scan/table_scan/scan_contract.hpp"

#include <cudf/io/parquet_schema.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table.hpp>

#include <cucascade/io/types.hpp>
#include <duckdb/main/client_context.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cucascade::io {
class ioctx;
}  // namespace cucascade::io

namespace sirius::op::scan {

/// Independent input for a DV preparation task; contains no borrowed context/provider.
struct dv_descriptor {
  std::string referenced_data_file;
  int64_t content_offset, content_size_in_bytes, record_count, file_size_in_bytes;
  std::string puffin_path;
};

/// One file entry from an Iceberg manifest: equality-delete files and V3 deletion vectors.
struct IcebergDeleteFileEntry {
  std::string file_path;
  int content{0};                     // 0=DATA, 1=POSITION_DELETES, 2=EQUALITY_DELETES
  std::string file_format;            // "parquet" or "puffin" (always lowercase)
  std::string referenced_data_file;   // data file this DV applies to (V3, empty if absent)
  int64_t content_offset{-1};         // byte offset in Puffin file (V3, -1 if absent)
  int64_t content_size_in_bytes{-1};  // byte length of DV blob (V3, -1 if absent)
  int64_t sequence_number{0};         // inherited data sequence number (equality deletes)
  int64_t file_size_in_bytes{-1};
  int64_t record_count{-1};  // deleted positions the manifest claims (V3, -1 if absent)

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
  /// Live equality-delete entries of the snapshot. Non-empty selects eager loading.
  std::vector<IcebergDeleteFileEntry> equality_delete_entries;
  std::vector<IcebergDeleteFileEntry> deletion_vector_entries;
  /// Present whenever @ref equality_delete_entries is non-empty on an admitted scan.
  std::optional<iceberg_equality_plan> equality_plan;
};

void validate_deletion_vector_claims(iceberg_delete_discovery const&);

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
// Inspect only Avro manifest schemas (LIMIT 0), once per data manifest. Empty
// partition records prove missing columns cannot be supplied by partition metadata.
std::unordered_set<std::string> iceberg_files_without_partition_values(
  duckdb::ClientContext&, iceberg_delete_inventory const&);

/// @p expected_table_uuid is checked by iceberg_metadata() when the loaded extension accepts it.
inventory_result read_delete_inventory(duckdb::ClientContext&,
                                       std::string const&,
                                       std::optional<int64_t>,
                                       std::string_view injection           = {},
                                       std::string_view expected_table_uuid = {});

/// Ceilings on the equality deletes a statement loads, shaped like the deletion-vector ceilings in
/// puffin_reader.hpp: one per delete file, one per statement, charged from the manifests'
/// `record_count` before any delete file is opened. A table above them is declined.
///
/// Sized to retain what the deletion-vector statement ceiling retains, about 2 GiB (256M
/// positions x 8 B). A built group retains 24 B per INT64 key row, 32 B for two INT64 keys and
/// about 31 B for a 10-character string key (measured, libcudf 26.10), so 64M rows is 1.5-2 GiB.
/// Unlike decoded positions this memory is on the device and outside the scan's reservations.
/// `record_count` bounds rows, not bytes; the byte ceiling on delete files covers long keys.
inline constexpr int64_t kMaxEqualityDeleteRowsPerFile       = 32LL * 1024 * 1024;
inline constexpr int64_t kMaxEqualityDeleteRowsPerStatement  = 64LL * 1024 * 1024;
inline constexpr int64_t kMaxEqualityDeleteBytesPerStatement = 1024LL * 1024 * 1024;
/// Not a memory bound: every batch is probed once per applicable delete group.
inline constexpr std::size_t kMaxEqualityDeleteFilesPerDataFile = 64;

/// Running total of the equality deletes a statement has agreed to load.
struct equality_delete_charge {
  int64_t rows  = 0;
  int64_t bytes = 0;
};
/// Adds every delete file of @p plan to @p total. False when a file is above the per-file
/// ceiling or the total would pass a statement ceiling; @p total is then unspecified.
[[nodiscard]] bool charge_equality_deletes(
  iceberg_equality_plan const& plan,
  equality_delete_charge& total,
  int64_t statement_row_limit = kMaxEqualityDeleteRowsPerStatement);

/// A scan plan that contradicts the inventory, or that this reader cannot interpret.
struct equality_plan_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Read which equality-delete files apply to which data file, from `iceberg_scan_plan`.
 *
 * Applicability is the extension's answer, checked against @p inventory: both must list the same
 * live data files and equality deletes, and every (data file, delete) pair must have a strictly
 * lower data sequence number on the data file. Any disagreement throws @ref equality_plan_error.
 * `keys` is left empty; the plan gate resolves field IDs against the bound schema.
 */
iceberg_equality_plan read_equality_plan(duckdb::ClientContext&,
                                         std::string const& table_path,
                                         int64_t snapshot_id,
                                         iceberg_delete_inventory const& inventory);
iceberg_delete_discovery discover_from_manifests(duckdb::ClientContext&,
                                                 std::string const&,
                                                 iceberg_delete_inventory&&);

/// Equality-delete files that share their key columns and apply to exactly the same data files.
struct EqualityDeleteGroup {
  /// GPU-resident deduplicated keys: one column per key, in @ref key_field_ids order, in the
  /// bound key types.
  std::unique_ptr<cudf::table> delete_table;
  /// Iceberg field IDs of the keys, ascending. From the manifest's `equality_ids`.
  std::vector<int32_t> key_field_ids;
  /// Bound-schema (P-space) index of each key, parallel to @ref key_field_ids.
  std::vector<std::size_t> key_primary_indices;
  /// The delete files merged into this group; diagnostics only.
  std::vector<std::string> delete_files;
  /// Build side = delete_table. NULL keys compare equal.
  std::unique_ptr<cudf::distinct_hash_join> hash_join;
};

/// Ascending indices into @ref IcebergDeleteData::equality_delete_groups.
using equality_group_list = std::vector<uint32_t>;

/**
 * @brief Fully materialized Iceberg delete data for one table.
 *
 * Legacy producer output, materialized at plan time. Manifest queries use internal
 * connections; Parquet position deletes use the original context and native reader. Each scan owns
 * its supplied discovery and result; no global memo.
 */
struct IcebergDeleteData {
  /// V2 positional deletes and V3 deletion vectors merged: data_file_path -> sorted positions.
  std::unordered_map<std::string, std::vector<int64_t>> positional_deletes;

  /// Built on the planning thread's device; the plan gate declines multi-GPU configurations.
  std::vector<EqualityDeleteGroup> equality_delete_groups;

  /// Every data file of the snapshot (keyed as the manifest wrote the path) -> the groups that
  /// apply to it. Files with the same applicability share one list object. Empty when the table
  /// has no equality deletes; otherwise a scanned file missing from it must be refused.
  std::unordered_map<std::string, std::shared_ptr<equality_group_list const>>
    equality_groups_by_data_file;

  [[nodiscard]] bool empty() const
  {
    return positional_deletes.empty() && equality_delete_groups.empty() &&
           equality_groups_by_data_file.empty();
  }
};

/**
 * @brief Concatenate the delete files' key rows, deduplicate them, and stand up the GPU hash join
 *        the scan probes. Every view holds exactly the key columns, in @p key_field_ids order.
 *
 * Exported so this and the anti-join mask can be tested without a table.
 */
EqualityDeleteGroup build_equality_group(std::vector<int32_t> key_field_ids,
                                         std::vector<std::size_t> key_primary_indices,
                                         std::vector<cudf::table_view> const& views);

/**
 * @brief Load immutable legacy delete payloads from this scan's supplied discovery.
 *
 * Caller must suppress DuckDB side-effects (InternalQueryGuard).
 *
 * THROWS on any failure to read the manifests or delete files, so an empty result means "this
 * table has no deletes" and never "the deletes could not be read" — treating the second as the
 * first drops deletes silently and returns rows the table removed.
 *
 * @param metadata_ioctx Routes the equality-delete parquet and footer reads. Single-GPU is
 *                       sufficient (planning-time reads). Shared because every datasource
 *                       opened on it co-owns it; nullptr throws.
 * The caller supplies discovery from the scan's snapshot-scoped inventory.
 */
std::shared_ptr<const IcebergDeleteData> load_delete_payload(
  duckdb::ClientContext& context,
  std::string const& table_path,
  std::shared_ptr<cucascade::io::ioctx> const& metadata_ioctx,
  iceberg_delete_discovery const& discovery);

/// Legacy payload builds. Monotonic; retained for the existing scan-local read oracle.
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
