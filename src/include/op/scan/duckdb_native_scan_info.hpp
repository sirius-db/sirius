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

#include "helper/logical_type.hpp"
#include "op/scan/duckdb_native_metadata.hpp"
#include "op/scan/scan_info.hpp"

#include <duckdb/common/types/column/column_data_collection.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::io {
class sirius_ioctx;
}  // namespace sirius::io

namespace sirius::scan_manager {
class split_provider;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

/// DuckDB-native bind-data. Single-use: `make_provider()` moves the whole
/// object into the split_provider's owned scan_info, leaving `storage` /
/// `context` null on the source. A second call would trip the non-null
/// preconditions in `duckdb_native_split_provider`'s ctor.
struct duckdb_native_scan_info : public scan_info {
  duckdb::DataTable* storage     = nullptr;
  duckdb::ClientContext* context = nullptr;

  // projected_cols / projected_types are sized to the FULL scan output (includes filter-only
  // columns appended by sirius_plan_get.cpp). Filter application + projection down to
  // output_types happens inside decode_duckdb_native_split.
  std::vector<projected_column> projected_cols;
  std::vector<sirius::logical_type> projected_types;

  /// Output types after filter-only-column projection. Set by the pipeline converter
  /// from the original scan's `types`; scan operator projects down to these post-filter.
  duckdb::vector<sirius::logical_type> output_types;

  std::string db_path;

  /// Single call only — moves *this into the returned split_provider.
  std::unique_ptr<scan_manager::split_provider> make_provider(
    scan_manager::sirius_scan_manager& manager,
    std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs) override;
};

}  // namespace sirius::op::scan
