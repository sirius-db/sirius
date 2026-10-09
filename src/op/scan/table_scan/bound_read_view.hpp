/*
 * Copyright 2026, Sirius Contributors.
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

#include "transparent/plan_source_policy.hpp"

#include <duckdb/common/types.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace duckdb {
class ClientContext;
class DataTable;
class LogicalGet;
class LogicalOperator;
class PhysicalOperator;
class PhysicalTableScan;
}  // namespace duckdb
namespace sirius::op {
class sirius_physical_table_scan;
}
namespace sirius::op::scan {
/// Storage or streaming source category used by scan admission.
enum class source_kind : uint8_t { duckdb_native, parquet_local, parquet_s3, stream_source };
/// Detail available in captured file observations, not a snapshot guarantee.
enum class evidence_depth : uint8_t { path, path_and_size, path_size_and_tag };

/// Verified scan implementation and the registry profile used to identify it.
struct verified_source_identity {
  std::string function_name;
  source_kind kind = source_kind::duckdb_native;
  /// Versioned registry identifier for the verified scan implementation.
  std::string registry_profile;
};
/// Number of bound files; their paths are encoded in the canonical identity.
struct file_inventory {
  uint32_t count = 0;
};
/// Optional per-file observations in canonical path order, separate from read identity.
struct file_evidence_arrays {
  /// File sizes in bytes; valid only where size_present is nonzero.
  std::vector<int64_t> size;
  /// Captured modification timestamps; valid only where last_modified_present is nonzero.
  std::vector<int64_t> last_modified;
  /// Per-file presence flags distinguish missing observations from zero values.
  std::vector<uint8_t> size_present;
  std::vector<uint8_t> last_modified_present;
  /// Captured object tags; an empty string means no tag was observed.
  std::vector<std::string> etag;
};
/// Catalog and table incarnation identifying a bound native table.
struct native_table_identity {
  std::string catalog_name;
  duckdb::idx_t catalog_oid = 0;
  std::string schema_name;
  std::string table_name;
  duckdb::idx_t table_oid = 0;
  std::string db_path;
};
/// Identity of a bound streaming input.
struct stream_identity {
  uint64_t stream_id = 0;
};
/// Non-owning, statement-bounded pointers for the existing provider execution lanes.
struct provider_borrow {
  /// Active DuckDB query ID at capture time; does not extend provider lifetime.
  uint64_t statement_id          = 0;
  uint64_t transaction_id        = 0;
  duckdb::ClientContext* context = nullptr;
  duckdb::DataTable* table       = nullptr;
};
/// Canonical identity bytes and a hash used to accelerate comparisons.
struct read_view_fingerprint {
  std::string canonical;
  /// Lookup accelerator only; equality compares the full canonical bytes.
  uint64_t hash = 0;
  bool operator==(read_view_fingerprint const& other) const { return canonical == other.canonical; }
};

/// Capture storage capacities in bytes, except for the file count; not allocator peak usage.
struct read_view_capture_metrics {
  std::size_t file_count = 0;
  /// Canonical identity string capacity, including its terminator.
  std::size_t canonical_capacity = 0;
  /// Evidence vector buffers and tag string capacities.
  std::size_t evidence_capacity = 0;
  /// Temporary path containers and string capacities used during capture.
  std::size_t transient_path_capacity = 0;
  /// Temporary index capacity used to order the file inventory.
  std::size_t sort_index_capacity = 0;
};

/// Stable read identity shared after an equal comparison; evidence remains capture-specific.
struct bound_read_identity {
  verified_source_identity source;
  std::variant<native_table_identity, file_inventory, stream_identity> data_view;
  duckdb::vector<duckdb::LogicalType> bound_types;
  duckdb::vector<std::string> bound_names;
  /// Canonical bound reader options participating in stable identity.
  std::string selector;
  read_view_fingerprint fingerprint;
};
/// One captured read identity with its evidence, provider context, and replay policy.
struct bound_read_view {
  /// Captured replay permission, excluded from stable read identity.
  transparent::scan_source_policy replay_policy;
  std::shared_ptr<bound_read_identity const> identity;
  std::optional<provider_borrow> provider;
  uint64_t transaction_id = 0;
  std::shared_ptr<file_evidence_arrays const> evidence;
  /// Available file observation detail; does not strengthen identity equality.
  evidence_depth depth = evidence_depth::path;
  read_view_capture_metrics metrics;
  /// Requires correspondence to prove the evaluated logical selector, even if its evidence
  /// is missing. This requirement is separate from stable identity.
  bool selector_evidence_required = false;
  /// Evaluated selector captured from logical binding, such as an Iceberg snapshot choice.
  std::optional<std::string> logical_selector_evidence;
};

/// Logical scan binding paired with its captured read view.
struct logical_bound_read_view {
  /// DuckDB logical binding index used to match scans across a copied plan.
  duckdb::idx_t table_index;
  bound_read_view view;
};

/// Logical read views captured in one planning generation.
struct logical_bound_read_view_capture {
  /// Registry generation to which these original logical bindings belong.
  uint64_t planning_generation = 0;
  std::vector<logical_bound_read_view> views;
};

// Maps original file order to evidence order without retaining another path inventory.
// An empty result means the paths are already sorted; use the original file position.
std::vector<std::size_t> make_read_view_evidence_index(std::span<std::string const> paths);

// Paths are borrowed only while encoding and are not retained beside the canonical text.
std::shared_ptr<bound_read_identity const> make_bound_read_identity(
  bound_read_identity,
  std::span<std::string const> paths,
  read_view_capture_metrics* metrics = nullptr);
bound_read_view capture_bound_read_view(duckdb::LogicalGet const&, duckdb::ClientContext&);
std::vector<logical_bound_read_view> capture_bound_read_views(duckdb::LogicalOperator const&,
                                                              duckdb::ClientContext&);
bound_read_view capture_bound_read_view(duckdb::PhysicalTableScan const&, duckdb::ClientContext&);
std::vector<bound_read_view> capture_bound_read_views(duckdb::PhysicalOperator const&,
                                                      duckdb::ClientContext&);
bound_read_view capture_bound_read_view(sirius::op::sirius_physical_table_scan const&,
                                        duckdb::ClientContext&,
                                        std::span<std::string const> resolved_paths = {});
std::string canonical_read_view_text(bound_read_view const&);
std::string canonical_value_text(duckdb::Value const&);
}  // namespace sirius::op::scan
