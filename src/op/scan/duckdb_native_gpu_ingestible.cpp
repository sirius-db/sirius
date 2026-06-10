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

// sirius
#include <expression/ast/from_duckdb.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <io/io_context.hpp>
#include <log/logging.hpp>
#include <op/scan/duckdb_native_decoder.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

// cudf
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

// Mirrors duckdb_native_split_provider::partition_row_groups_into_batches.
struct row_group_batch_local {
  std::size_t first_idx;
  std::size_t count;
};

std::vector<row_group_batch_local> partition_row_groups_into_batches(
  const std::vector<duckdb_row_group_metadata>& row_groups,
  std::size_t approximate_batch_size,
  scan_plan const& plan)
{
  std::vector<row_group_batch_local> batches;
  if (row_groups.empty()) return batches;

  const std::size_t num_cols = plan.data_columns.size();
  std::vector<bool> is_varchar(num_cols, false);
  bool any_varchar = false;
  for (std::size_t c = 0; c < num_cols; ++c) {
    is_varchar[c] = plan.data_columns[c].reader_info->type.is_varchar();
    any_varchar   = any_varchar || is_varchar[c];
  }

  if (approximate_batch_size == 0 && !any_varchar) {
    batches.push_back({0, row_groups.size()});
    return batches;
  }

  std::size_t batch_first = 0;
  std::size_t batch_bytes = 0;
  std::vector<std::size_t> col_bytes(num_cols, 0);

  for (std::size_t i = 0; i < row_groups.size(); ++i) {
    const auto& rg                  = row_groups[i];
    const std::size_t this_rg_bytes = rg.decoded_bytes_budget;

    if (i > batch_first) {
      const bool would_exceed_total =
        (approximate_batch_size > 0) && (batch_bytes + this_rg_bytes > approximate_batch_size);
      bool would_exceed_varchar = false;
      if (any_varchar) {
        for (std::size_t c = 0; c < num_cols; ++c) {
          if (is_varchar[c] &&
              col_bytes[c] + rg.varchar_bytes_per_col[c] >= kCudfInt32StringsThreshold) {
            would_exceed_varchar = true;
            break;
          }
        }
      }
      if (would_exceed_total || would_exceed_varchar) {
        batches.push_back({batch_first, i - batch_first});
        batch_first = i;
        batch_bytes = 0;
        std::fill(col_bytes.begin(), col_bytes.end(), 0);
      }
    }

    batch_bytes += this_rg_bytes;
    if (any_varchar) {
      for (std::size_t c = 0; c < num_cols; ++c) {
        col_bytes[c] += rg.varchar_bytes_per_col[c];
      }
    }
  }
  if (batch_first < row_groups.size()) {
    batches.push_back({batch_first, row_groups.size() - batch_first});
  }
  return batches;
}

}  // namespace

//===----------------------------------------------------------------------===//
// duckdb_native_ingestible_table_info::make_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<io::gpu_ingestible> duckdb_native_ingestible_table_info::make_ingestible(
  std::unique_ptr<io::ingestible_table_info> self,
  scan_manager::sirius_scan_manager const& mgr,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs)
{
  return std::make_shared<duckdb_native_gpu_ingestible>(std::move(self), mgr, gpu_ioctxs);
}

//===----------------------------------------------------------------------===//
// duckdb_native_gpu_ingestible — construction
//===----------------------------------------------------------------------===//
duckdb_native_gpu_ingestible::duckdb_native_gpu_ingestible(
  std::unique_ptr<io::ingestible_table_info> info,
  scan_manager::sirius_scan_manager const& mgr,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& /*gpu_ioctxs*/)
  : io::gpu_ingestible(std::move(info))
{
  auto const& bind = static_cast<duckdb_native_ingestible_table_info const&>(table_info());

  if (bind.storage == nullptr) {
    throw std::invalid_argument(
      "[duckdb_native_gpu_ingestible] table_info.storage must be non-null");
  }
  if (bind.context == nullptr) {
    throw std::invalid_argument(
      "[duckdb_native_gpu_ingestible] table_info.context must be non-null");
  }
  if (!bind.plan) {
    throw std::invalid_argument("[duckdb_native_gpu_ingestible] table_info.plan must be non-null");
  }
  _plan = bind.plan;

  // Walk metadata once. Pushed-down filters drive row-group pruning in the walk;
  // column_ids maps each filter to its storage index.
  _metadata = walk_duckdb_native_metadata(
    *bind.storage, *bind.context, *_plan, bind.table_filters.get(), &bind.column_ids);
  if (!_metadata.viable) {
    SPDLOG_DEBUG("[duckdb_native_gpu_ingestible] non-viable: {}",
                 _metadata.viability_failure_reason);
    throw std::runtime_error("duckdb-native scan rejected query: " +
                             _metadata.viability_failure_reason);
  }

  // Mint the .db io_object once per query when the manager exposes a
  // backend for db_path.
  _io_ctx = !bind.db_path.empty() ? mgr.io_ctx_shared_for(bind.db_path) : nullptr;
  if (_io_ctx && !bind.db_path.empty()) { _db_io_object = _io_ctx->create_io_object(bind.db_path); }

  // Pre-build the coalesced filter expression once. The plan's C → D map resolves
  // each filter column to its batch position (rowid included).
  if (bind.table_filters && !bind.table_filters->filters.empty()) {
    auto filter_expr_duckdb =
      sirius::op::convert_table_filters_to_expression(*bind.table_filters,
                                                      bind.column_ids,
                                                      bind.returned_types,
                                                      _plan->batch_position_by_column_id);
    if (filter_expr_duckdb) {
      _filter_expression = std::shared_ptr<duckdb::Expression>(filter_expr_duckdb.release());
    }
  }

  _needs_assembly = needs_output_assembly(*_plan);

  auto batches =
    partition_row_groups_into_batches(_metadata.row_groups, bind.approximate_batch_size, *_plan);
  _batches.reserve(batches.size());
  for (auto const& b : batches) {
    _batches.push_back({b.first_idx, b.count});
  }
}

duckdb_native_gpu_ingestible::~duckdb_native_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool duckdb_native_gpu_ingestible::has_more_splits() const
{
  return _next_batch_idx.load(std::memory_order_relaxed) < _batches.size();
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
duckdb_native_gpu_ingestible::next_split_provider()
{
  std::size_t idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _batches.size()) { return {}; }
  row_group_batch claimed = _batches[idx];

  bool const apply_filter        = static_cast<bool>(_filter_expression);
  bool const needs_assembly      = _needs_assembly;
  bool const has_post_processing = apply_filter || needs_assembly;

  return [this, claimed, apply_filter, needs_assembly, has_post_processing]()
           -> std::vector<std::unique_ptr<op::operator_data>> {
    auto split_info = std::make_unique<duckdb_native_split_info>();
    split_info->payload.table_info =
      &static_cast<duckdb_native_ingestible_table_info const&>(table_info());
    split_info->payload.io_ctx       = _io_ctx;
    split_info->payload.db_io_object = _db_io_object;
    split_info->payload.row_groups.reserve(claimed.count);
    for (std::size_t i = claimed.first_idx; i < claimed.first_idx + claimed.count; ++i) {
      split_info->payload.row_groups.push_back(std::move(_metadata.row_groups[i]));
    }

    std::unique_ptr<io::post_filter_and_projection_info> filter_info;
    if (has_post_processing) {
      auto pf            = std::make_unique<duckdb_native_post_filter_and_projection_info>();
      pf->apply_filter   = apply_filter;
      pf->needs_assembly = needs_assembly;
      filter_info        = std::move(pf);
    }

    auto metadata =
      std::make_unique<io::scan_and_filter_metadata>(std::move(split_info), std::move(filter_info));

    std::vector<std::unique_ptr<op::operator_data>> out;
    out.push_back(std::make_unique<scan_operator_input>(std::move(metadata)));
    return out;
  };
}

//===----------------------------------------------------------------------===//
// materialize_table
//===----------------------------------------------------------------------===//
io::filtered_table duckdb_native_gpu_ingestible::materialize_table(
  io::scan_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const& split = static_cast<duckdb_native_split_info const&>(info);
  // Decoder takes mem_space by non-const ref; the const-ref input here is a
  // formality (the ingestible interface preserves immutability conceptually,
  // but the decoder mutates the allocator state on the space).
  auto& mem_space_mut = const_cast<::cucascade::memory::memory_space&>(mem_space);
  auto table          = decode_duckdb_native_split(split.payload, mem_space_mut, stream);
  SIRIUS_LOG_DEBUG(
    "[duckdb_native_gpu_ingestible::materialize_table] decoded split: row_groups={} rows={} "
    "cols={}",
    split.payload.row_groups.size(),
    table->num_rows(),
    table->num_columns());
  // duckdb-native applies filter + projection inside post_filter_and_project,
  // never during materialization — always UNFILTERED here.
  return io::filtered_table{std::move(table), io::filter_state::UNFILTERED};
}

//===----------------------------------------------------------------------===//
// post_filter_and_project — filter eval + projection to output arity
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> duckdb_native_gpu_ingestible::post_filter_and_project(
  std::unique_ptr<cudf::table> input,
  io::post_filter_and_projection_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const& pf = static_cast<duckdb_native_post_filter_and_projection_info const&>(info);

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());

  if (pf.apply_filter && _filter_expression) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_filter_expression);
    sirius::gpu_expression_executor exec(sirius_filter_ast.get(), mr_ref, stream);
    auto src = std::move(input);
    input    = exec.select(src->view());
  }

  // Reshape the decoder's D-order batch to the plan's output layout: drop
  // pure-filter columns and reorder to the operator's expected output.
  if (pf.needs_assembly) {
    input = assemble_scan_output(*_plan, std::move(input), /*partition_values=*/{}, stream);
  }

  return input;
}

}  // namespace sirius::op::scan
