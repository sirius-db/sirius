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
#include <duckdb/common/open_file_info.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::op::scan {
struct bound_read_identity;
}

namespace sirius::ducklake {
enum class unit_kind : uint8_t { parquet, inline_table, transaction_local_inline };
enum class delete_format : uint8_t { none, parquet, puffin, unknown };

// Provider discovery metadata, not a shared execution unit or an eligibility
// certificate. Lowering must still seal mapping/visibility/physical dependencies.
struct unit_descriptor {
  uint64_t original_file_index = 0;
  std::shared_ptr<op::scan::bound_read_identity const> read_view;
  unit_kind kind = unit_kind::parquet;
  // Preserve the provider's options, including inline schema and local markers.
  duckdb::OpenFileInfo input;
  uint64_t bound_snapshot = 0, bound_schema_version = 0;
  std::optional<uint64_t> file_id, data_file_id, mapping_id, row_id_start, insertion_snapshot;
  std::optional<uint64_t> snapshot_min, snapshot_max, inline_schema_version;
  uint64_t file_size = 0;
  std::optional<uint64_t> footer_size;
  bool encrypted = false;
  std::string delete_path;
  delete_format deletes     = delete_format::none;
  uint64_t delete_file_size = 0;
  std::optional<uint64_t> delete_footer_size;
  bool delete_encrypted         = false;
  bool transaction_local_delete = false;
  // Positions remain with the original list and can be copied into a bounded
  // caller buffer. Discovery never duplicates the complete position set.
  uint64_t inline_delete_count = 0;
};

enum class unit_gate_reason : uint8_t {
  unknown_unit_kind,
  remote_data,
  remote_delete,
  invalid_locator,
  encrypted_data,
  encrypted_delete,
  invalid_snapshot_range,
  unknown_delete_format,
  parquet_delete_semantics_required,
  dv_metadata_required
};
struct unit_gate {
  // These gates only identify local scan scope and unresolved delete evidence.
  // Empty reasons are never authorization to materialize this descriptor.
  std::vector<unit_gate_reason> reasons;
  bool outside_local_scope = false;
};
unit_gate assess_local_scope(unit_descriptor const&);

struct position_page {
  std::size_t size = 0;
  bool more        = false;
};
}  // namespace sirius::ducklake
