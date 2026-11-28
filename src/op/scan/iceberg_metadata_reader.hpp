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

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cucascade/io/types.hpp>
#include <duckdb/main/client_context.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cucascade::io {
class ioctx;
}  // namespace cucascade::io

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

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
  inventory_query_error(std::string const& message, bool missing, bool unconfirmed = false)
    : std::runtime_error(message), missing_result(missing), unconfirmed_table(unconfirmed)
  {
  }
  bool missing_result;
  /// The lookup did not confirm that it reads the table the scan bound: it resolved another
  /// table, or one it could not identify. what() says which.
  bool unconfirmed_table;
};
// Inspect only Avro manifest schemas (LIMIT 0), once per data manifest. Empty
// partition records prove missing columns cannot be supplied by partition metadata.
std::unordered_set<std::string> iceberg_files_without_partition_values(
  duckdb::ClientContext&, iceberg_delete_inventory const&);

/// @p expected_table_uuid is checked by iceberg_metadata() when the loaded extension accepts it.
/// When it does not and @p unconfirmed_table is set, that runs first on the connection that
/// reads the inventory, in the same transaction, and says why the table resolved there is not
/// confirmed to be the bound one (empty when it is the bound one); a reason fails the read.
inventory_result read_delete_inventory(
  duckdb::ClientContext&,
  std::string const&,
  std::optional<int64_t>,
  std::string_view injection                                                  = {},
  std::string_view expected_table_uuid                                        = {},
  std::function<std::string(duckdb::ClientContext&)> const& unconfirmed_table = {});

/// Ceilings on the equality deletes one statement loads, charged from the manifests' record
/// counts and file sizes before any delete file is opened; a table above them is declined. The
/// byte ceiling exists because a row count does not bound variable-width keys.
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

/// The cuDF type an equality-delete key bound as @p type is compared in, or nullopt for a type
/// this path does not compare: a data file and a delete file must decode the key identically.
[[nodiscard]] std::optional<cudf::data_type> equality_key_cudf_type(
  duckdb::LogicalType const& type);

/// Equality-delete manifests that cannot be read or interpreted, or that contradict the inventory.
struct equality_plan_error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Decide which equality-delete files apply to which data file of @p inventory.
 *
 * `equality_ids`, sizes and partitions come from the manifests the inventory names, read with
 * `read_avro`; the table path is not resolved again. A delete applies to a data file with a
 * strictly lower data sequence number, in the same partition unless the delete's spec does not
 * partition. A manifest whose live entries are not the inventory's throws
 * @ref equality_plan_error. `keys` is left empty; the plan gate resolves them.
 *
 * @param max_files_per_data_file A data file's list stops growing one past this.
 */
iceberg_equality_plan read_equality_plan(duckdb::ClientContext&,
                                         iceberg_delete_inventory const& inventory,
                                         std::size_t max_files_per_data_file);
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
  /// Build side = delete_table.
  std::unique_ptr<cudf::distinct_hash_join> hash_join;
};

/// Ascending indices into the groups of an @ref equality_replica.
using equality_group_list = std::vector<uint32_t>;

/// A table's equality-delete groups as built on one device. Every replica of a table holds the
/// same groups in the same order.
struct equality_replica {
  rmm::cuda_device_id device;
  std::vector<EqualityDeleteGroup> groups;
};

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

  /// One per GPU the query can be admitted to, none when no group applies to a data file. Each
  /// is released with its device current. Mutable: admission releases the ones it leaves out
  /// through the scan's const payload.
  mutable std::vector<equality_replica> equality_replicas;
  ~IcebergDeleteData();

  /// Releases the tables on every device outside @p devices. Not synchronized: call before a
  /// batch probes.
  void release_equality_replicas_outside(std::span<int const> devices) const;

  /// The groups built on @p device. Throws when there are none: skipping the probe would return
  /// deleted rows, and another device's tables need not be addressable from this one.
  [[nodiscard]] std::vector<EqualityDeleteGroup> const& equality_groups_on(
    rmm::cuda_device_id device) const;

  /// Every data file of the snapshot (keyed as the manifest wrote the path) -> the groups that
  /// apply to it. Files with the same applicability share one list object. Empty when the table
  /// has no equality deletes; otherwise a scanned file missing from it must be refused.
  std::unordered_map<std::string, std::shared_ptr<equality_group_list const>>
    equality_groups_by_data_file;

  [[nodiscard]] bool empty() const
  {
    return positional_deletes.empty() && equality_replicas.empty() &&
           equality_groups_by_data_file.empty();
  }
};

/**
 * @brief Concatenate the delete files' key rows, deduplicate them, and stand up the GPU hash join
 *        the scan probes. Every view holds exactly the key columns, in @p key_field_ids order.
 *
 * Exported so this and the anti-join mask can be tested without a table.
 *
 * @param stream A stream of the current device.
 * @param mr     Allocates everything the group retains.
 */
EqualityDeleteGroup build_equality_group(std::vector<int32_t> key_field_ids,
                                         std::vector<std::size_t> key_primary_indices,
                                         std::vector<cudf::table_view> const& views,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

/// A device equality-delete tables are built on, with a stream and an allocator of it.
struct equality_build_resources {
  rmm::cuda_device_id device;
  ::cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;
  /// The context an `s3://` delete file is opened on, with its statement's credentials. Unset
  /// or returning null, such a file fails the load. The first entry's is the one used.
  std::function<std::shared_ptr<cucascade::io::ioctx>(std::string const& path)> remote_ioctx = {};
  /// The GPU memory space @ref mr allocates from. Keys are copied between the devices of two
  /// entries through it; null stages such a copy on the host, which then needs a space as well.
  cucascade::memory::memory_space const* space = nullptr;
};

/**
 * @brief Load immutable legacy delete payloads from this scan's supplied discovery.
 *
 * Caller must suppress DuckDB side-effects (InternalQueryGuard).
 *
 * THROWS on any failure to read the manifests or delete files, so an empty result means "this
 * table has no deletes" and never "the deletes could not be read" — treating the second as the
 * first drops deletes silently and returns rows the table removed.
 *
 * @param metadata_ioctx Routes the local equality-delete parquet and footer reads. Shared
 *                       because every datasource opened on it co-owns it; nullptr throws.
 * @param equality_resources One per GPU a scan task can run on; the delete tables are built
 *                       once on each. The first reads the delete files and the others copy its
 *                       keys. Required when @p discovery has equality deletes; empty throws.
 * The caller supplies discovery from the scan's snapshot-scoped inventory.
 */
std::shared_ptr<const IcebergDeleteData> load_delete_payload(
  duckdb::ClientContext& context,
  std::string const& table_path,
  std::shared_ptr<cucascade::io::ioctx> const& metadata_ioctx,
  iceberg_delete_discovery const& discovery,
  std::vector<equality_build_resources> equality_resources = {});

/// Legacy payload builds. Monotonic; retained for the existing scan-local read oracle.
uint64_t iceberg_delete_data_uncached_read_count();

}  // namespace sirius::op::scan
