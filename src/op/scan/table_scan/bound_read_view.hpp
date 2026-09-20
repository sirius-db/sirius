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
enum class source_kind : uint8_t { duckdb_native, parquet_local, parquet_s3, stream_source };
enum class evidence_depth : uint8_t { path, path_and_size, path_size_and_tag };

struct verified_source_identity {
  std::string function_name;
  source_kind kind = source_kind::duckdb_native;
  std::string registry_profile;
};
struct file_inventory {
  uint32_t count = 0;
};
struct file_evidence_arrays {
  std::vector<int64_t> size;
  std::vector<int64_t> last_modified;
  std::vector<uint8_t> size_present;
  std::vector<uint8_t> last_modified_present;
  std::vector<std::string> etag;
};
struct native_table_identity {
  std::string catalog_name;
  duckdb::idx_t catalog_oid = 0;
  std::string schema_name;
  std::string table_name;
  duckdb::idx_t table_oid = 0;
  std::string db_path;
};
struct stream_identity {
  uint64_t stream_id = 0;
};
// Non-owning, statement-bounded pointers; use only on the existing provider execution lanes.
struct provider_borrow {
  uint64_t statement_id          = 0;
  uint64_t transaction_id        = 0;
  duckdb::ClientContext* context = nullptr;
  duckdb::DataTable* table       = nullptr;
};
struct read_view_fingerprint {
  std::string canonical;
  uint64_t hash = 0;
  bool operator==(read_view_fingerprint const& other) const { return canonical == other.canonical; }
};

struct read_view_capture_metrics {
  std::size_t file_count              = 0;
  std::size_t canonical_capacity      = 0;
  std::size_t evidence_capacity       = 0;
  std::size_t transient_path_capacity = 0;
  std::size_t sort_index_capacity     = 0;
};

// Only this content is shared after an equal comparison; each capture retains its evidence.
struct bound_read_identity {
  verified_source_identity source;
  std::variant<native_table_identity, file_inventory, stream_identity> data_view;
  duckdb::vector<duckdb::LogicalType> bound_types;
  duckdb::vector<std::string> bound_names;
  std::string selector;
  read_view_fingerprint fingerprint;
};
struct bound_read_view {
  // Diagnostic policy is observation state, never part of stable read identity.
  transparent::scan_source_policy replay_policy;
  std::shared_ptr<bound_read_identity const> identity;
  std::optional<provider_borrow> provider;
  uint64_t transaction_id = 0;
  std::shared_ptr<file_evidence_arrays const> evidence;
  evidence_depth depth = evidence_depth::path;
  read_view_capture_metrics metrics;
  // Kept outside the stable identity: this says whether correspondence must also prove the
  // evaluated logical selector, even if capture of that evidence is unexpectedly absent.
  bool selector_evidence_required = false;
  std::optional<std::string> logical_selector_evidence;
};

struct logical_bound_read_view {
  duckdb::idx_t table_index;
  bound_read_view view;
};

struct logical_bound_read_view_capture {
  uint64_t planning_generation = 0;
  std::vector<logical_bound_read_view> views;
};

// Maps original file order to evidence order without retaining another path inventory.
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
