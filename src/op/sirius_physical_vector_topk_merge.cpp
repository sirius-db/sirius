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

#include "op/sirius_physical_vector_topk_merge.hpp"

#include "cuda/vss/knn_merge.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_vector_topk_join.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <raft/core/device_resources.hpp>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sirius {
namespace op {

namespace {

//! [left types..., right types..., FLOAT ranking value].
duckdb::vector<sirius::logical_type> merged_types(const sirius_physical_vector_topk_join& join)
{
  auto types        = join.children[0]->get_types();
  auto const& right = join.children[1]->get_types();
  types.insert(types.end(), right.begin(), right.end());
  types.push_back(sirius::from_duckdb(duckdb::LogicalType::FLOAT));
  return types;
}

}  // namespace

sirius_physical_vector_topk_merge::sirius_physical_vector_topk_merge(
  const sirius_physical_vector_topk_join& join, uint64_t batch_bytes)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::VECTOR_TOPK_MERGE,
                                                merged_types(join),
                                                join.estimated_cardinality),
    k(join.k),
    is_similarity(join.is_similarity),
    join_type(join.join_type),
    batch_bytes(batch_bytes),
    right_types_(join.children[1]->get_types())
{
}

std::unique_ptr<operator_data> sirius_physical_vector_topk_merge::get_next_task_input_data()
{
  // One task per left batch: its left columns first, then all of its partials.
  std::scoped_lock lg(drain_mutex_);
  auto* repo = ports.begin()->second->repo;

  auto const ordinal = next_left_ordinal_;
  if (topk_left_partition(ordinal) >= repo->num_partitions()) { return nullptr; }
  next_left_ordinal_++;

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  while (auto batch = repo->pop_next_data_batch(topk_left_partition(ordinal))) {
    batches.push_back(std::move(batch));
  }
  if (batches.size() != 1) {
    throw std::runtime_error(
      "sirius_physical_vector_topk_merge: expected one left part for left "
      "batch " +
      std::to_string(ordinal) + ", got " + std::to_string(batches.size()));
  }
  while (auto batch = repo->pop_next_data_batch(topk_partials_partition(ordinal))) {
    batches.push_back(std::move(batch));
  }
  return std::make_unique<partitioned_operator_data>(std::move(batches), ordinal);
}

std::unique_ptr<operator_data> sirius_physical_vector_topk_merge::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_topk_merge::execute"};
  auto const& input         = dynamic_cast<const partitioned_operator_data&>(input_data);
  auto const& input_batches = input.get_read_only_batches();
  auto const partition_idx  = input.get_partition_idx().value_or(0);
  if (input_batches.empty()) {
    throw std::runtime_error("sirius_physical_vector_topk_merge expects the left part first");
  }

  auto const& left_batch = input_batches[0];
  cudf::table_view left  = get_cudf_table_view(left_batch);
  auto* space            = left_batch.get_memory_space();
  auto mr                = space->get_default_allocator();
  auto const n_left      = static_cast<std::int64_t>(left.num_rows());
  auto const n_parts     = static_cast<std::int64_t>(input_batches.size()) - 1;

  std::vector<std::shared_ptr<cucascade::data_batch>> out_batches;
  auto finish = [&]() {
    return std::make_unique<partitioned_operator_data>(std::move(out_batches), partition_idx);
  };
  auto emit_table = [&](std::unique_ptr<cudf::table> table) {
    out_batches.push_back(make_data_batch(std::move(table), *space, stream, batch_telemetry()));
  };

  if (n_left == 0 || (n_parts == 0 && join_type != duckdb::JoinType::LEFT)) {
    emit_table(make_empty_table(types));
    return finish();
  }

  // No right rows at all: LEFT keeps every left row with NULL right columns and a NULL ranking.
  if (n_parts == 0) {
    auto const n     = static_cast<cudf::size_type>(n_left);
    auto empty_right = make_empty_table(right_types_);
    cudf::numeric_scalar<cudf::size_type> zero(0, true, stream);
    // Every index is out of range of the empty right table, so NULLIFY yields all-NULL rows.
    auto pad = cudf::make_column_from_scalar(zero, n, stream, mr);
    auto right_cols =
      cudf::gather(
        empty_right->view(), pad->view(), cudf::out_of_bounds_policy::NULLIFY, stream, mr)
        ->release();
    auto cols = std::make_unique<cudf::table>(left, stream, mr)->release();
    for (auto& c : right_cols) {
      cols.push_back(std::move(c));
    }
    cols.push_back(cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT32}, n, cudf::mask_state::ALL_NULL, stream, mr));
    emit_table(std::make_unique<cudf::table>(std::move(cols)));
    return finish();
  }

  // Stack the partials part-major; each is [right cols..., distance] with n_left * k rows.
  auto const part_rows = n_left * k;
  std::vector<cudf::table_view> payload_views;
  std::vector<cudf::column_view> distance_views;
  std::size_t partial_bytes = 0;
  for (std::size_t i = 1; i < input_batches.size(); ++i) {
    auto const tv = get_cudf_table_view(input_batches[i]);
    if (tv.num_rows() != part_rows) {
      throw std::runtime_error("sirius_physical_vector_topk_merge: a partial has " +
                               std::to_string(tv.num_rows()) + " rows, expected " +
                               std::to_string(part_rows));
    }
    std::vector<cudf::size_type> payload_cols(tv.num_columns() - 1);
    std::iota(payload_cols.begin(), payload_cols.end(), 0);
    payload_views.push_back(tv.select(payload_cols));
    distance_views.push_back(tv.column(tv.num_columns() - 1));
    partial_bytes += input_batches[i].get_data()->get_size_in_bytes();
  }
  auto stacked_payload   = cudf::concatenate(payload_views, stream, mr);
  auto stacked_distances = cudf::concatenate(distance_views, stream, mr);

  // Ids are row positions in the stacked payload, so the merged ids gather it directly.
  cudf::numeric_scalar<std::int64_t> zero64(0, true, stream);
  cudf::numeric_scalar<std::int64_t> one64(1, true, stream);
  auto stacked_ids =
    cudf::sequence(static_cast<cudf::size_type>(n_parts * part_rows), zero64, one64, stream, mr);

  raft::device_resources res{stream};
  auto merged = vss::knn_merge_parts_topk(
    res, stacked_distances->view(), stacked_ids->view(), n_left, n_parts, k, stream, mr);

  // Merged results are k per left row, nearest first: left row i repeated k times.
  cudf::numeric_scalar<cudf::size_type> zero(0, true, stream);
  cudf::numeric_scalar<cudf::size_type> one(1, true, stream);
  auto left_rows = cudf::sequence(static_cast<cudf::size_type>(n_left), zero, one, stream, mr);
  auto left_map  = cudf::repeat(
    cudf::table_view({left_rows->view()}), static_cast<cudf::size_type>(k), stream, mr);

  // When the whole right side has fewer than k rows, padding rows (at +inf) survive the merge;
  // drop them.
  cudf::numeric_scalar<float> pad_distance(std::numeric_limits<float>::max(), true, stream);
  auto is_real = cudf::binary_operation(merged.distances->view(),
                                        pad_distance,
                                        cudf::binary_operator::LESS,
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);
  auto kept    = cudf::apply_boolean_mask(
    cudf::table_view(
      {left_map->get_column(0).view(), merged.neighbors->view(), merged.distances->view()}),
    is_real->view(),
    stream,
    mr);
  auto kept_cols       = kept->release();
  auto const left_idx  = kept_cols[0]->view();
  auto const right_idx = kept_cols[1]->view();
  auto ranking         = std::move(kept_cols[2]);
  if (is_similarity) {
    cudf::numeric_scalar<float> one_f(1.0F, true, stream);
    ranking = cudf::binary_operation(one_f,
                                     ranking->view(),
                                     cudf::binary_operator::SUB,
                                     cudf::data_type{cudf::type_id::FLOAT32},
                                     stream,
                                     mr);
  }

  // Emit within the byte budget.
  auto const total = static_cast<std::size_t>(left_idx.size());
  auto const left_row_bytes =
    left_batch.get_data()->get_size_in_bytes() / static_cast<std::size_t>(n_left);
  auto const right_row_bytes = partial_bytes / static_cast<std::size_t>(n_parts * part_rows);
  auto const max_rows        = std::max<std::size_t>(
    1, batch_bytes / std::max<std::size_t>(1, left_row_bytes + right_row_bytes));
  for (std::size_t start = 0; start < total; start += max_rows) {
    auto const s = static_cast<cudf::size_type>(start);
    auto const e = static_cast<cudf::size_type>(std::min(total, start + max_rows));
    auto cols    = cudf::gather(left,
                             cudf::slice(left_idx, {s, e}).front(),
                             cudf::out_of_bounds_policy::DONT_CHECK,
                             stream,
                             mr)
                  ->release();
    auto right_cols = cudf::gather(stacked_payload->view(),
                                   cudf::slice(right_idx, {s, e}).front(),
                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                   stream,
                                   mr)
                        ->release();
    for (auto& c : right_cols) {
      cols.push_back(std::move(c));
    }
    cols.push_back(
      std::make_unique<cudf::column>(cudf::slice(ranking->view(), {s, e}).front(), stream, mr));
    emit_table(std::make_unique<cudf::table>(std::move(cols)));
  }
  if (out_batches.empty()) { emit_table(make_empty_table(types)); }
  return finish();
}

}  // namespace op
}  // namespace sirius
