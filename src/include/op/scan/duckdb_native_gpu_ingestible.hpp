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
#include <io/gpu_ingestible.hpp>
#include <op/scan/duckdb_native_decoder.hpp>  // duckdb_native_split_payload
#include <op/scan/duckdb_native_metadata.hpp>
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
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::scan_manager {
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// duckdb_native_ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief DuckDB-native bind-data carrier; factory for
 *        @c duckdb_native_gpu_ingestible.
 *
 * Mirror of the legacy @c duckdb_native_scan_info field set, but rooted in
 * @c io::ingestible_table_info instead of @c op::scan::scan_info.
 */
class duckdb_native_ingestible_table_info : public io::ingestible_table_info {
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

  std::shared_ptr<io::gpu_ingestible> make_ingestible(
    std::unique_ptr<io::ingestible_table_info> self,
    scan_manager::sirius_scan_manager const& mgr) override;

  /// db_path-as-span. The cache match in @c sirius_scan_manager never
  /// matches duckdb-native ingestibles (pinned-cache key is parquet file
  /// paths), so this is purely contract-keeping.
  [[nodiscard]] std::span<std::string const> file_paths() const override
  {
    if (db_path.empty()) { return {}; }
    return std::span<std::string const>(&db_path, 1);
  }
};

//===----------------------------------------------------------------------===//
// duckdb_native_split_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split scan metadata for a duckdb-native row-group batch.
 *
 * Owns the @c split_payload built from a contiguous run of the walker's
 * metadata. @c materialize_table forwards the payload to
 * @c decode_duckdb_native_split unchanged.
 */
class duckdb_native_split_info : public io::scan_info {
 public:
  duckdb_native_split_payload payload;

  [[nodiscard]] std::size_t estimated_bytes() const noexcept override
  {
    std::size_t total = 0;
    for (auto const& rg : payload.row_groups) {
      total += rg.decoded_bytes_budget;
    }
    return total;
  }
};

//===----------------------------------------------------------------------===//
// duckdb_native_post_filter_and_projection_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split post-decode filter + projection description.
 *
 * Emitted by @c duckdb_native_gpu_ingestible whenever the @c table_filters
 * survived translation OR the scan emitted more columns than
 * @c output_types requested (trailing filter-only columns).
 * @c apply_filter triggers the @c gpu_expression_executor pass against the
 * ingestible's shared filter expression; @c output_arity drives the
 * projection-down step.
 */
class duckdb_native_post_filter_and_projection_info : public io::post_filter_and_projection_info {
 public:
  bool apply_filter        = false;
  std::size_t output_arity = 0;
};

//===----------------------------------------------------------------------===//
// duckdb_native_gpu_ingestible
//===----------------------------------------------------------------------===//
class duckdb_native_gpu_ingestible : public io::gpu_ingestible {
 public:
  duckdb_native_gpu_ingestible(std::unique_ptr<io::ingestible_table_info> info,
                               scan_manager::sirius_scan_manager const& mgr);

  ~duckdb_native_gpu_ingestible() override;

  [[nodiscard]] bool has_more_splits() const override;
  std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider() override;

  io::filtered_table materialize_table(io::scan_info const& info,
                                       ::cucascade::memory::memory_space const& mem_space,
                                       rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    std::unique_ptr<cudf::table> input,
    io::post_filter_and_projection_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) override;

 private:
  struct row_group_batch {
    std::size_t first_idx;
    std::size_t count;
  };

  duckdb_native_metadata _metadata;
  std::shared_ptr<sirius::io::sirius_ioctx> _io_ctx;
  std::shared_ptr<sirius::io::sirius_io_object> _db_io_object;
  /// Pre-built coalesced filter expression. Empty when no translatable
  /// filters survived. Reused across every emitted split.
  std::shared_ptr<duckdb::Expression> _filter_expression;
  bool _projection_required = false;
  std::size_t _output_arity = 0;

  std::vector<row_group_batch> _batches;
  std::atomic<std::size_t> _next_batch_idx{0};
};

}  // namespace sirius::op::scan
