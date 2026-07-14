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

// sirius
#include <helper/logical_type.hpp>
#include <op/scan/duckdb_native_decoder.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <sirius_config.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace duckdb {
class SingleFileBlockManager;
}

namespace sirius::op {
class sirius_dynamic_filter_set;
}  // namespace sirius::op

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// duckdb_native_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Bind data for a duckdb-native scan; builds the @c duckdb_native_gpu_ingestible.
 */
class duckdb_native_ingestible_table_info : public op::scan::ingestible_table_info {
 public:
  duckdb::vector<sirius::logical_type> returned_types;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> names;
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  /// Sirius-side dynamic join filters published by a build-side hash join. Null when none are
  /// wired. The ingestible installs the consumer column remap on the channel; the downstream
  /// dynamic-filter operator applies the filters post-decode (the native scan has no read-time
  /// dynamic path).
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> sirius_dynamic_filters;
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;

  duckdb::DataTable* storage     = nullptr;
  duckdb::ClientContext* context = nullptr;
  std::vector<projected_column> projected_cols;
  std::vector<sirius::logical_type> projected_types;
  duckdb::vector<sirius::logical_type> output_types;
  std::string db_path;
  /// Qualified table reference (catalog / schema / table) derived from the resolved
  /// DuckTableEntry. The pin cache (@c cache_entry_info) keys on these for
  /// duckdb-native identity instead of the @c storage pointer, so the pin-time and
  /// query-time derivation MUST use the same DuckTableEntry accessors.
  std::string catalog_name;
  std::string schema_name;
  std::string table_name;

  duckdb_native_ingestible_table_info() = default;

  [[nodiscard]] std::span<std::string const> column_names() const override { return names; }

  /// db_path-as-span (contract-keeping for the base interface). duckdb-native
  /// cache matching keys on the catalog/schema/table name in @c cache_entry_info,
  /// not on these paths.
  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    if (db_path.empty()) { return {}; }
    return std::span<std::string const>(&db_path, 1);
  }
};

//===----------------------------------------------------------------------===//
// duckdb_native_scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief One unit of duckdb-native scan work. A metadata-scan task emits one per row-group range;
 * the batch coalescer merges them into decode-batch-sized units the scan operator decodes.
 */
class duckdb_native_scan_info : public op::scan::scan_info {
 public:
  /// Row-group metadata for this unit.
  std::vector<duckdb_row_group_metadata> row_groups;
  /// Read handle for the .db file; prefetched by the sequencer and decoded by materialize.
  std::shared_ptr<sirius::io::sirius_datasource> datasource;
  /// Resolves block ids to file offsets when deriving the on-disk ranges below.
  duckdb::SingleFileBlockManager const* block_manager = nullptr;

  /// On-disk byte ranges this unit reads, derived from @ref row_groups so they always match the row
  /// groups currently held. The scan sequencer fadvises these to prefetch.
  [[nodiscard]] std::vector<fadvise_entry> fadvise_entries() const override
  {
    if (!datasource || block_manager == nullptr) { return {}; }
    fadvise_entry entry;
    entry.datasource = datasource;
    for (auto const& rg : row_groups) {
      auto ranges = row_group_file_ranges(*block_manager, rg);
      entry.ranges.insert(entry.ranges.end(), ranges.begin(), ranges.end());
    }
    return {std::move(entry)};
  }

  /// Decoded (GPU) byte budget for this unit; drives memory reservation.
  [[nodiscard]] std::size_t estimated_bytes() const noexcept override
  {
    std::size_t total = 0;
    for (auto const& rg : row_groups) {
      total += rg.decoded_bytes_budget;
    }
    return total;
  }
};

//===----------------------------------------------------------------------===//
// duckdb_native_gpu_ingestible
//===----------------------------------------------------------------------===//
class duckdb_native_gpu_ingestible : public op::scan::gpu_ingestible {
 public:
  duckdb_native_gpu_ingestible(std::unique_ptr<op::scan::duckdb_native_ingestible_table_info> info);

  ~duckdb_native_gpu_ingestible() override;

  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;

  [[nodiscard]] bool has_processed_all_metadata() const override;

  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve) override;

  op::scan::filtered_table materialize_metadata_to_table(
    scan_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) override;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept override { return *_info; }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const override;

 private:
  std::unique_ptr<op::scan::duckdb_native_ingestible_table_info> _info;
  duckdb_native_walk_plan _plan;
  std::shared_ptr<duckdb::Expression> _filter_expression;
  duckdb::SingleFileBlockManager const* _block_manager = nullptr;

  //===----------RG Range Slicing----------===//
  std::size_t _chunk_row_groups =
    1;  ///< The number of row groups to chunk together for each metadata scan task.
  std::size_t _num_ranges = 0;
  std::atomic<std::size_t> _next_range_idx{0};
};

std::shared_ptr<duckdb_native_gpu_ingestible> make_ingestible(
  std::unique_ptr<duckdb_native_ingestible_table_info> info);

}  // namespace sirius::op::scan
