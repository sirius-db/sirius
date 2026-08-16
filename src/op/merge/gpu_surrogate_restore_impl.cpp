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

#include "op/merge/gpu_surrogate_restore_impl.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <utility>

namespace sirius::op {

bool gpu_surrogate_restore_impl::is_recomposable_aggregate(cudf::aggregation::Kind kind) noexcept
{
  switch (kind) {
    case cudf::aggregation::Kind::MIN:
    case cudf::aggregation::Kind::MAX:
    case cudf::aggregation::Kind::SUM:
    case cudf::aggregation::Kind::COUNT_ALL:
    case cudf::aggregation::Kind::COUNT_VALID: return true;
    default: return false;
  }
}

std::unique_ptr<cudf::groupby_aggregation> gpu_surrogate_restore_impl::make_recompose_aggregation(
  cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::MIN:
      return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::MAX:
      return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::Kind::SUM:
    case cudf::aggregation::Kind::COUNT_ALL:
    case cudf::aggregation::Kind::COUNT_VALID:
      // Partial counts re-combine by summation.
      return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    default:
      throw sirius::internal_exception(
        "gpu_surrogate_restore_impl::make_recompose_aggregation: unsupported aggregate kind for "
        "re-group (planner invariant violated): {}",
        static_cast<int>(kind));
  }
}

bool gpu_surrogate_restore_impl::tuples_proven_distinct(
  std::vector<cudf::column_view> const& real_keys,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream)
{
  for (auto const& key : real_keys) {
    auto const type_id = key.type().id();
    if (type_id == cudf::type_id::FLOAT32 || type_id == cudf::type_id::FLOAT64) { return false; }
  }
  auto const distinct =
    cudf::distinct_count(cudf::table_view(real_keys), cudf::null_equality::EQUAL, stream);
  return distinct == num_rows;
}

void gpu_surrogate_restore_impl::restore_deferred_keys(
  std::vector<std::unique_ptr<cudf::column>>& cols,
  surrogate_restore_plan::restore_group const& group,
  surrogate_deferral_store const& store,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = cols.empty() ? 0 : cols.front()->size();
  if (num_rows == 0) {
    for (auto const& key : group.keys()) {
      cols.at(static_cast<std::size_t>(key.key_slot)) =
        cudf::make_empty_column(sirius::get_cudf_type(key.original_type));
    }
    return;
  }

  std::vector<cudf::size_type> source_cols;
  source_cols.reserve(group.keys().size());
  for (auto const& key : group.keys()) {
    source_cols.push_back(key.source_col);
  }

  auto const sources = store.snapshot(group.side());
  std::vector<cudf::table_view> source_views;
  source_views.reserve(sources.size());
  for (auto const& src : sources) {
    // Order this task's stream after each source's writer event before reading its memory.
    if (auto const writer_event = src.batch.get_writer_event(); writer_event != nullptr) {
      CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.value(), writer_event, 0));
    }
    source_views.push_back(sirius::get_cudf_table_view(src.batch).select(source_cols));
  }
  if (source_views.empty()) {
    throw sirius::internal_exception(
      "gpu_surrogate_restore_impl::restore_deferred_keys: merged rows reference deferred string "
      "sources but none were registered by the deferral join on the {} side",
      to_string(group.side()));
  }
  std::unique_ptr<cudf::table> src_owned;
  cudf::table_view src_view;
  if (source_views.size() == 1) {
    src_view = source_views[0];
  } else {
    src_owned = cudf::concatenate(source_views, stream, mr);
    src_view  = src_owned->view();
  }

  // The BIGINT rowids are absolute addresses into the concatenated source view. The INT32
  // narrowing cast is lossless (reserve() refuses address spaces beyond int32) and every rowid
  // names a row an INNER-join gather actually produced, so DONT_CHECK is safe.
  auto const& rowid_col = cols.at(static_cast<std::size_t>(group.rowid_key_slot()));
  auto rowid_gather_map =
    cudf::cast(rowid_col->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto gathered = cudf::gather(
    src_view, rowid_gather_map->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto gathered_cols = gathered->release();
  for (std::size_t i = 0; i < group.keys().size(); ++i) {
    cols.at(static_cast<std::size_t>(group.keys()[i].key_slot)) = std::move(gathered_cols[i]);
  }
}

std::vector<std::unique_ptr<cudf::column>> gpu_surrogate_restore_impl::regroup_full_tuple(
  std::vector<std::unique_ptr<cudf::column>> cols,
  std::size_t num_key_cols,
  std::vector<cudf::aggregation::Kind> const& kinds,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<cudf::column_view> key_views;
  key_views.reserve(num_key_cols);
  for (std::size_t i = 0; i < num_key_cols; ++i) {
    key_views.push_back(cols[i]->view());
  }
  // null_policy::INCLUDE matches the main HASH_GROUP_BY and MERGE_GROUP_BY group-bys: SQL
  // GROUP BY gives NULL keys their own group.
  cudf::groupby::groupby grpby_obj(cudf::table_view(key_views), cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.reserve(kinds.size());
  for (std::size_t i = 0; i < kinds.size(); ++i) {
    cudf::groupby::aggregation_request request;
    request.values = cols.at(num_key_cols + i)->view();
    request.aggregations.push_back(make_recompose_aggregation(kinds[i]));
    requests.push_back(std::move(request));
  }
  auto groupby_result = grpby_obj.aggregate(requests, stream, mr);
  auto new_cols       = groupby_result.first->release();
  for (auto& aggregation_result : groupby_result.second) {
    new_cols.push_back(std::move(aggregation_result.results[0]));
  }
  return new_cols;
}

}  // namespace sirius::op
