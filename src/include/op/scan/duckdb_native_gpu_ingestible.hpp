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
#include <unordered_map>
#include <vector>

namespace duckdb {
class SingleFileBlockManager;
}

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
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;

  duckdb::DataTable* storage     = nullptr;
  duckdb::ClientContext* context = nullptr;
  std::vector<projected_column> projected_cols;
  std::vector<sirius::logical_type> projected_types;
  duckdb::vector<sirius::logical_type> output_types;
  std::string db_path;

  duckdb_native_ingestible_table_info() = default;

  /// db_path-as-span. The cache match in @c sirius_scan_manager never
  /// matches duckdb-native ingestibles (pinned-cache key is parquet file
  /// paths), so this is purely contract-keeping.
  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    if (db_path.empty()) { return {}; }
    return std::span<std::string const>(&db_path, 1);
  }

  /// Can serve @p other iff it is the same duckdb table (same DataTable*) and every column @p other
  /// requests is also read by this scan — i.e. this scan's (cached) data is a superset that can
  /// serve @p other. Columns are matched by storage column id, not by name or position.
  ///
  /// Returns, for each column @p other requests (in @p other's @c column_ids order), the position
  /// of that column within THIS scan's @c column_ids — a gather index into this scan's (cached)
  /// materialized columns that reproduces @p other's requested layout (the index space
  /// @c cached_databatch_provider slices). Empty when @p other is a different table or requests a
  /// column this scan does not read.
  [[nodiscard]] std::vector<std::size_t> can_serve_with_columns(
    const ingestible_table_info& other) const override
  {
    auto const* o = dynamic_cast<duckdb_native_ingestible_table_info const*>(&other);
    if (o == nullptr || storage != o->storage) { return {}; }

    std::unordered_map<duckdb::idx_t, std::size_t> this_pos;
    this_pos.reserve(column_ids.size());
    for (std::size_t i = 0; i < column_ids.size(); ++i) {
      this_pos.emplace(column_ids[i].GetPrimaryIndex(), i);
    }
    std::vector<std::size_t> projection;
    projection.reserve(o->column_ids.size());
    for (auto const& c : o->column_ids) {
      auto it = this_pos.find(c.GetPrimaryIndex());
      if (it == this_pos.end()) { return {}; }  // this scan lacks a requested column
      projection.push_back(it->second);
    }
    return projection;
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

  std::unique_ptr<batch_coalecer> create_batch_coalecer() const override;

  [[nodiscard]] bool has_processed_all_metadata() const override;

  metadata_scan_task_t next_split_provider(std::shared_ptr<io::sirius_ioctx> io_ctx) override;

  op::scan::filtered_table materialize_metadata_to_table(
    scan_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& input,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) override;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept override { return *_info; }

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
