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

#include "op/sirius_physical_window.hpp"

#include "data/data_batch_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_window::sirius_physical_window(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> window_exprs,
  std::size_t estimated_cardinality)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::WINDOW, std::move(types), estimated_cardinality)
{
  auto defs     = convert_duckdb_window_to_cudf(window_exprs);
  partition_idx = std::move(defs.partition_idx);
  order_idx     = std::move(defs.order_idx);
  order_dirs    = std::move(defs.order_dirs);
  order_null    = std::move(defs.order_null);
  ranks         = std::move(defs.ranks);
}

// Drain all batches of one partition per task, mirroring sirius_physical_grouped_aggregate_merge.
std::unique_ptr<operator_data> sirius_physical_window::get_next_task_input_data()
{
  std::lock_guard<std::mutex> lg(lock);
  if (current_partition_index < ports.begin()->second->repo->num_partitions()) {
    std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
    bool found_batch = true;
    while (found_batch) {
      auto batch = ports.begin()->second->repo->pop_data_batch(
        ::cucascade::batch_state::task_created, current_partition_index);
      if (batch) {
        input_batch.push_back(std::move(batch));
      } else {
        found_batch = false;
      }
    }
    current_partition_index++;
    if (input_batch.empty()) { return nullptr; }
    return std::make_unique<pipelineable_operator_data>(input_batch);
  }
  return nullptr;
}

std::unique_ptr<operator_data> sirius_physical_window::execute(const operator_data& input_data,
                                                               rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_window::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_data_batches();

  // Collect the partition's non-empty batches and their (shared) memory space.
  cucascade::memory::memory_space* space = nullptr;
  std::vector<cudf::table_view> views;
  for (const auto& batch : input_batches) {
    if (!batch) { continue; }
    auto* batch_space = batch->get_memory_space();
    if (!batch_space) { continue; }
    if (!space) { space = batch_space; }
    views.push_back(get_cudf_table_view(*batch));
  }
  if (space == nullptr || views.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<::cucascade::data_batch>>{});
  }
  auto mr = space->get_default_allocator();

  // 1. View the partition's rows as one table. Multiple batches are concatenated into an owned
  //    table; a lone batch is used in place (no copy), kept alive by input_batches.
  std::unique_ptr<cudf::table> concatenated;
  cudf::table_view source_view;
  if (views.size() == 1) {
    source_view = views[0];
  } else {
    concatenated = cudf::concatenate(views, stream, mr);
    source_view  = concatenated->view();
  }

  // 2. Stable sort by (PARTITION BY ASC, ORDER BY dirs), then gather into the owned `working`
  // table.
  //    Stable keeps output run-to-run deterministic. With neither PARTITION BY nor ORDER BY there
  //    is nothing to sort, so just take ownership (copying a lone input batch).
  std::vector<cudf::column_view> sort_cols;
  std::vector<cudf::order> sort_orders;
  std::vector<cudf::null_order> sort_null;
  for (int idx : partition_idx) {
    sort_cols.push_back(source_view.column(idx));
    sort_orders.push_back(cudf::order::ASCENDING);
    sort_null.push_back(cudf::null_order::AFTER);
  }
  for (size_t i = 0; i < order_idx.size(); ++i) {
    sort_cols.push_back(source_view.column(order_idx[i]));
    sort_orders.push_back(order_dirs[i]);
    sort_null.push_back(order_null[i]);
  }
  std::unique_ptr<cudf::table> working;
  if (sort_cols.empty()) {
    working = concatenated ? std::move(concatenated)
                           : std::make_unique<cudf::table>(source_view, stream, mr);
  } else {
    auto perm =
      cudf::stable_sorted_order(cudf::table_view(sort_cols), sort_orders, sort_null, stream, mr);
    working =
      cudf::gather(source_view, perm->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    concatenated.reset();  // free the pre-sort concatenation; `working` owns the sorted rows
  }
  auto working_view = working->view();
  auto num_rows     = working_view.num_rows();

  // 3. Grouped rank scan over the sorted rows. Group keys = PARTITION BY columns (a single constant
  //    group when there is no PARTITION BY). null_policy::INCLUDE treats a NULL key as its own
  //    group.
  std::unique_ptr<cudf::column> const_key;
  std::vector<cudf::column_view> key_cols;
  std::vector<cudf::order> key_order;
  std::vector<cudf::null_order> key_null;
  for (int idx : partition_idx) {
    key_cols.push_back(working_view.column(idx));
    key_order.push_back(cudf::order::ASCENDING);
    key_null.push_back(cudf::null_order::AFTER);
  }
  if (key_cols.empty()) {
    auto zero = cudf::numeric_scalar<int8_t>(0, true, stream);
    const_key = cudf::make_column_from_scalar(zero, num_rows, stream, mr);
    key_cols.push_back(const_key->view());
    key_order.push_back(cudf::order::ASCENDING);
    key_null.push_back(cudf::null_order::AFTER);
  }

  // Rank values: RANK / DENSE_RANK need the ORDER BY tuple so tied rows compare equal; ROW_NUMBER
  // (rank_method::FIRST) ignores values because row order is already fixed by the sort above. Build
  // the struct-of-order-columns only when a tie-sensitive rank is present (the Phase 1 guard
  // ensures RANK/DENSE_RANK always carry ORDER BY); otherwise a constant column suffices and we
  // skip copying the order columns.
  bool needs_order_values = false;
  for (auto kind : ranks) {
    if (kind == window_rank_kind::RANK || kind == window_rank_kind::DENSE_RANK) {
      needs_order_values = true;
      break;
    }
  }
  std::unique_ptr<cudf::column> value_col;
  if (needs_order_values && !order_idx.empty()) {
    std::vector<std::unique_ptr<cudf::column>> value_children;
    value_children.reserve(order_idx.size());
    for (int idx : order_idx) {
      value_children.push_back(
        std::make_unique<cudf::column>(working_view.column(idx), stream, mr));
    }
    value_col = cudf::make_structs_column(
      num_rows, std::move(value_children), 0, rmm::device_buffer{}, stream, mr);
  } else {
    auto zero = cudf::numeric_scalar<int8_t>(0, true, stream);
    value_col = cudf::make_column_from_scalar(zero, num_rows, stream, mr);
  }

  // sorted::YES marks the keys (and values) as already grouped/sorted. Under this "presorted" path
  // cuDF's grouped rank scan uses an identity value-order, so the rank order comes entirely from
  // the (PARTITION BY, ORDER BY) pre-sort above and the rank aggregation's
  // column_order/null_precedence below are ignored. ASC/DESC and NULLS FIRST/LAST (including the
  // DESC null flip) are therefore honored by the pre-sort, not by these aggregation arguments.
  cudf::groupby::groupby grpby(
    cudf::table_view(key_cols), cudf::null_policy::INCLUDE, cudf::sorted::YES, key_order, key_null);

  std::vector<std::unique_ptr<cudf::groupby_scan_aggregation>> aggs;
  aggs.reserve(ranks.size());
  for (auto kind : ranks) {
    cudf::rank_method method = cudf::rank_method::FIRST;
    switch (kind) {
      case window_rank_kind::ROW_NUMBER: method = cudf::rank_method::FIRST; break;
      case window_rank_kind::RANK: method = cudf::rank_method::MIN; break;
      case window_rank_kind::DENSE_RANK: method = cudf::rank_method::DENSE; break;
    }
    // column_order / null_precedence here are inert under sorted::YES (see note above); the rows
    // are already arranged by the pre-sort. Tie detection uses equality on the ORDER BY value
    // tuple.
    aggs.push_back(cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
      method, cudf::order::ASCENDING, cudf::null_policy::INCLUDE, cudf::null_order::AFTER));
  }

  std::vector<cudf::groupby::scan_request> requests;
  cudf::groupby::scan_request request;
  request.values       = value_col->view();
  request.aggregations = std::move(aggs);
  requests.push_back(std::move(request));

  auto scan_result = grpby.scan(requests, stream, mr);

  // 4. Output = sorted child columns ++ one rank column per window expression, cast to BIGINT.
  auto output_cols   = working->release();
  auto& rank_results = scan_result.second[0].results;
  for (auto& rank_col : rank_results) {
    output_cols.push_back(
      cudf::cast(rank_col->view(), cudf::data_type(cudf::type_id::INT64), stream, mr));
  }

  auto output_table = std::make_unique<cudf::table>(std::move(output_cols), stream, mr);
  std::vector<std::shared_ptr<::cucascade::data_batch>> out_batches{
    make_data_batch(std::move(output_table), *space)};
  return std::make_unique<pipelineable_operator_data>(out_batches);
}

}  // namespace op
}  // namespace sirius
