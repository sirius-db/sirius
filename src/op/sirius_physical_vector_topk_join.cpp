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

#include "op/sirius_physical_vector_topk_join.hpp"

#include "cuda/vss/brute_force_search.hpp"
#include "cuda/vss/cudf_raft_interop.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/exception.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_concat.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "vss/distance_metric.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <raft/core/device_resources.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sirius {
namespace op {

namespace {

//! Per-row partials are [right types..., FLOAT distance];
//! Global rows are [left types..., right types..., FLOAT ranking value].
duckdb::vector<sirius::logical_type> output_types(const sirius_physical_operator& left,
                                                  const sirius_physical_operator& right,
                                                  topk_scope scope)
{
  duckdb::vector<sirius::logical_type> types;
  if (scope == topk_scope::global) { types = left.get_types(); }
  types.insert(types.end(), right.get_types().begin(), right.get_types().end());
  types.push_back(sirius::from_duckdb(duckdb::LogicalType::FLOAT));
  return types;
}

//! A right batch with fewer than k rows only yields k_eff (< k) neighbors per left row, but the
//! merge needs exactly k per row from every part. Widen each row's k_eff results to k, filling the
//! tail with @p filler.
std::unique_ptr<cudf::column> pad_rows_to_k(cudf::column_view const& src,
                                            std::int64_t n_rows,
                                            std::int64_t k_eff,
                                            std::int64_t k,
                                            cudf::scalar const& filler,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref const& mr)
{
  auto const out_size = static_cast<cudf::size_type>(n_rows * k);
  auto out =
    cudf::make_numeric_column(src.type(), out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto out_view = out->mutable_view();
  cudf::fill_in_place(out_view, 0, out_size, filler, stream);

  // Copy each row's k_eff real results to the front of its k-wide slot.
  auto const elem = cudf::size_of(src.type());
  CUDF_CUDA_TRY(cudaMemcpy2DAsync(out_view.head<std::uint8_t>(),
                                  static_cast<std::size_t>(k) * elem,
                                  src.head<std::uint8_t>(),
                                  static_cast<std::size_t>(k_eff) * elem,
                                  static_cast<std::size_t>(k_eff) * elem,
                                  static_cast<std::size_t>(n_rows),
                                  cudaMemcpyDeviceToDevice,
                                  stream.get()));
  return out;
}

}  // namespace

sirius_physical_vector_topk_join::sirius_physical_vector_topk_join(
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  std::size_t left_vector_col_idx,
  std::size_t right_vector_col_idx,
  std::int64_t k,
  std::string metric,
  bool is_similarity,
  std::int64_t dim,
  duckdb::JoinType join_type,
  std::size_t estimated_cardinality,
  topk_scope scope)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::VECTOR_TOPK_JOIN,
                                                output_types(*left, *right, scope),
                                                estimated_cardinality),
    left_vector_col_idx(left_vector_col_idx),
    right_vector_col_idx(right_vector_col_idx),
    k(k),
    metric(std::move(metric)),
    is_similarity(is_similarity),
    dim(dim),
    join_type(join_type),
    scope(scope)
{
  if (scope == topk_scope::global && join_type != duckdb::JoinType::INNER) {
    throw duckdb::NotImplementedException("Vector global top-k join: only INNER is supported");
  }
  if (join_type != duckdb::JoinType::INNER && join_type != duckdb::JoinType::LEFT) {
    throw duckdb::NotImplementedException("Vector top-k join: only INNER and LEFT are supported");
  }
  children.push_back(std::move(left));
  children.push_back(std::move(right));
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_vector_topk_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Per-row: its own single-operator pipeline sinking into the merge.
  // Global: the head of the pipeline that streams into TOP_N. Either way, build then probe inputs.
  pipeline::sirius_meta_pipeline* host_meta;
  pipeline::sirius_pipeline* host_current;
  if (is_sink()) {
    auto& sink_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
    host_meta       = &sink_meta;
    host_current    = sink_meta.get_base_pipeline().get();
  } else {
    meta_pipeline.get_state().add_pipeline_operator(current, *this);
    host_meta    = &meta_pipeline;
    host_current = &current;
  }

  D_ASSERT(children.size() == 2);
  auto& build_child = *children[1];
  D_ASSERT(build_child.is_sink());
  D_ASSERT(!build_child.children.empty());
  auto& build_meta = host_meta->create_child_meta_pipeline(*host_current, build_child);
  build_meta.build(*build_child.children[0]);

  auto& probe_child = *children[0];
  D_ASSERT(probe_child.is_sink());
  D_ASSERT(!probe_child.children.empty());
  auto& probe_meta = host_meta->create_child_meta_pipeline(*host_current, probe_child);
  probe_meta.build(*probe_child.children[0]);
}

partition_strategy sirius_physical_vector_topk_join::get_partition_strategy(
  const partition_sizing_input& /*in*/)
{
  return {/*num_partitions=*/1, /*broadcast=*/false, /*build_probe=*/false};
}

std::string_view sirius_physical_vector_topk_join::input_port_for(
  sirius_physical_operator const& producer) const
{
  if (producer.type == SiriusPhysicalOperatorType::CONCAT) {
    return producer.Cast<sirius_physical_concat>().is_build_concat() ? "build" : "default";
  }
  return sirius_physical_operator::input_port_for(producer);
}

std::unique_ptr<operator_data> sirius_physical_vector_topk_join::get_next_task_input_data()
{
  // One task per (left batch, right batch) pair, or per left batch when the right side is empty.
  std::scoped_lock lg(batches_to_processed_mutex);

  auto* default_port = get_port("default");
  auto* build_port   = get_port("build");

  // Moves the cursor past partitions with no left batch. Their right batches are popped and dropped
  // so the build repository still drains and the pipeline can finish.
  auto skip_empty_left = [&]() {
    while (cursor_partition_ < left_batch_ids.size() && left_batch_ids[cursor_partition_].empty()) {
      for (auto id : right_batch_ids[cursor_partition_]) {
        build_port->repo->pop_data_batch_by_id(id, cursor_partition_);
      }
      cursor_partition_++;
    }
  };

  if (!ids_initialized_) {
    if (!default_port || !default_port->repo || !build_port || !build_port->repo) {
      return nullptr;
    }
    if (default_port->repo->num_partitions() != build_port->repo->num_partitions()) {
      throw std::runtime_error(
        "sirius_physical_vector_topk_join: number of partitions for default and build ports must "
        "match");
    }
    for (size_t i = 0; i < default_port->repo->num_partitions(); i++) {
      left_batch_ids.push_back(default_port->repo->get_batch_ids(i));
      right_batch_ids.push_back(build_port->repo->get_batch_ids(i));
    }
    ids_initialized_ = true;
    skip_empty_left();
  }

  if (cursor_partition_ >= left_batch_ids.size()) { return nullptr; }

  auto const p     = cursor_partition_;
  auto const li    = cursor_left_;
  auto const ri    = cursor_right_;
  auto const& lids = left_batch_ids[p];
  auto const& rids = right_batch_ids[p];

  auto [it, inserted] = left_ordinal_.try_emplace(lids[li], next_left_ordinal_);
  if (inserted) { next_left_ordinal_++; }
  auto const ordinal = it->second;

  std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
  bool emit_left_part = false;
  if (rids.empty()) {
    input_batch.push_back(default_port->repo->pop_data_batch_by_id(lids[li], p));
    emit_left_part = true;
    cursor_left_++;
  } else {
    bool const last_right = ri + 1 == rids.size();
    bool const last_left  = li + 1 == lids.size();
    // A left batch is reused across every right batch, so release it only on the last right; and
    // a right batch likewise only on the last left.
    input_batch.push_back(last_right ? default_port->repo->pop_data_batch_by_id(lids[li], p)
                                     : default_port->repo->get_data_batch_by_id(lids[li], p));
    input_batch.push_back(last_left ? build_port->repo->pop_data_batch_by_id(rids[ri], p)
                                    : build_port->repo->get_data_batch_by_id(rids[ri], p));
    emit_left_part = ri == 0;
    cursor_right_++;
    if (cursor_right_ >= rids.size()) {
      cursor_right_ = 0;
      cursor_left_++;
    }
  }
  if (cursor_left_ >= lids.size()) {
    cursor_left_ = 0;
    cursor_partition_++;
    skip_empty_left();
  }

  return std::make_unique<topk_pair_input>(std::move(input_batch), ordinal, emit_left_part);
}

std::unique_ptr<operator_data> sirius_physical_vector_topk_join::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_topk_join::execute"};
  auto const* input = dynamic_cast<const topk_pair_input*>(&input_data);
  if (!input) {
    throw std::runtime_error("sirius_physical_vector_topk_join expects a topk_pair_input");
  }
  const auto& input_batches = input->get_read_only_batches();
  if (input_batches.empty() || input_batches.size() > 2) {
    throw std::runtime_error(
      "sirius_physical_vector_topk_join expects 1 or 2 input batches (left[, right]), got " +
      std::to_string(input_batches.size()));
  }

  auto const& left_batch = input_batches[0];
  cudf::table_view left  = get_cudf_table_view(left_batch);

  cucascade::memory::memory_space* space = left_batch.get_memory_space();
  if (!space) {
    return std::make_unique<topk_partial_output>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{}, std::vector<std::size_t>{});
  }
  auto mr = space->get_default_allocator();

  std::vector<std::shared_ptr<cucascade::data_batch>> out_batches;
  std::vector<std::size_t> out_partitions;

  if (scope == topk_scope::global) { return execute_global(*input, stream); }

  // Forward the left columns once per left batch; the merge repeats them per neighbor.
  if (input->emit_left_part) {
    out_batches.push_back(make_data_batch(
      std::make_unique<cudf::table>(left, stream, mr), *space, stream, batch_telemetry()));
    out_partitions.push_back(topk_left_partition(input->left_ordinal));
  }

  auto const n_left = static_cast<std::int64_t>(left.num_rows());
  if (input_batches.size() == 1 || n_left == 0) {
    return std::make_unique<topk_partial_output>(std::move(out_batches), std::move(out_partitions));
  }
  auto const& right_batch = input_batches[1];
  cudf::table_view right  = get_cudf_table_view(right_batch);
  auto const n_right      = static_cast<std::int64_t>(right.num_rows());
  if (n_right == 0) {
    return std::make_unique<topk_partial_output>(std::move(out_batches), std::move(out_partitions));
  }

  // A NULL vector has no distance to rank by; not supported.
  if (left.column(left_vector_col_idx).null_count() > 0) {
    throw std::runtime_error("Vector top-k join: the left input has NULL vectors");
  }
  if (right.column(right_vector_col_idx).null_count() > 0) {
    throw std::runtime_error("Vector top-k join: the right input has NULL vectors");
  }
  if (n_left * k > std::numeric_limits<cudf::size_type>::max()) {
    throw std::runtime_error(
      "Vector top-k join: left batch rows * k exceeds a cuDF column; lower the batch size");
  }

  auto const dataset = vss::list_column_as_dataset_view(right.column(right_vector_col_idx), dim);
  auto const queries = vss::list_column_as_dataset_view(left.column(left_vector_col_idx), dim);
  raft::device_resources res{stream};
  auto const metric_type =
    vss::join_selection_distance_type_from_metric(metric, /*exact_unexpanded=*/false);

  auto const k_eff = std::min(k, n_right);
  auto knn         = vss::brute_force_knn(res, dataset, queries, k_eff, metric_type, mr);
  auto neighbors   = std::move(knn.neighbors);
  auto distances   = std::move(knn.distances);
  if (k_eff < k) {
    // Padding rows: an out-of-range index gathers as NULL, at a distance that never wins the merge
    // over a real neighbor.
    cudf::numeric_scalar<std::int64_t> const id_filler(n_right, true, stream);
    cudf::numeric_scalar<float> const dist_filler(std::numeric_limits<float>::max(), true, stream);
    neighbors = pad_rows_to_k(neighbors->view(), n_left, k_eff, k, id_filler, stream, mr);
    distances = pad_rows_to_k(distances->view(), n_left, k_eff, k, dist_filler, stream, mr);
  }

  // Gather the right columns now, while this right batch is at hand.
  auto cols =
    cudf::gather(right, neighbors->view(), cudf::out_of_bounds_policy::NULLIFY, stream, mr)
      ->release();
  cols.push_back(std::move(distances));
  out_batches.push_back(make_data_batch(
    std::make_unique<cudf::table>(std::move(cols)), *space, stream, batch_telemetry()));
  out_partitions.push_back(topk_partials_partition(input->left_ordinal));

  return std::make_unique<topk_partial_output>(std::move(out_batches), std::move(out_partitions));
}

std::unique_ptr<operator_data> sirius_physical_vector_topk_join::execute_global(
  const topk_pair_input& input, ::cuda::stream_ref stream)
{
  const auto& input_batches = input.get_read_only_batches();
  auto const& left_batch    = input_batches[0];
  cudf::table_view left     = get_cudf_table_view(left_batch);
  auto* space               = left_batch.get_memory_space();
  auto mr                   = space->get_default_allocator();

  std::vector<std::shared_ptr<cucascade::data_batch>> out_batches;
  auto finish = [&](std::unique_ptr<cudf::table> table) {
    out_batches.push_back(make_data_batch(std::move(table), *space, stream, batch_telemetry()));
    return std::make_unique<pipelineable_operator_data>(std::move(out_batches));
  };

  // No pairs: still emit one (empty) batch so TOP_N downstream sees input and can finish.
  auto const n_left = static_cast<std::int64_t>(left.num_rows());
  if (input_batches.size() == 1 || n_left == 0) { return finish(make_empty_table(types)); }
  cudf::table_view right = get_cudf_table_view(input_batches[1]);
  auto const n_right     = static_cast<std::int64_t>(right.num_rows());
  if (n_right == 0) { return finish(make_empty_table(types)); }

  // A NULL vector has no distance to rank by; not supported.
  if (left.column(left_vector_col_idx).null_count() > 0) {
    throw std::runtime_error("Vector top-k join: the left input has NULL vectors");
  }
  if (right.column(right_vector_col_idx).null_count() > 0) {
    throw std::runtime_error("Vector top-k join: the right input has NULL vectors");
  }
  auto const k_eff = std::min(k, n_right);
  if (n_left * k_eff > std::numeric_limits<cudf::size_type>::max()) {
    throw std::runtime_error(
      "Vector top-k join: left batch rows * k exceeds a cuDF column; lower the batch size");
  }

  auto const dataset = vss::list_column_as_dataset_view(right.column(right_vector_col_idx), dim);
  auto const queries = vss::list_column_as_dataset_view(left.column(left_vector_col_idx), dim);
  raft::device_resources res{stream};
  auto const metric_type =
    vss::join_selection_distance_type_from_metric(metric, /*exact_unexpanded=*/false);
  auto knn = vss::brute_force_knn(res, dataset, queries, k_eff, metric_type, mr);

  // Candidates are k_eff per left row: left row i repeated k_eff times.
  cudf::numeric_scalar<cudf::size_type> zero(0, true, stream);
  cudf::numeric_scalar<cudf::size_type> one(1, true, stream);
  auto left_rows = cudf::sequence(static_cast<cudf::size_type>(n_left), zero, one, stream, mr);
  auto candidates =
    cudf::repeat(
      cudf::table_view({left_rows->view()}), static_cast<cudf::size_type>(k_eff), stream, mr)
      ->release();
  candidates.push_back(std::move(knn.neighbors));
  candidates.push_back(std::move(knn.distances));
  auto kept = std::make_unique<cudf::table>(std::move(candidates));

  // Only this pair's k closest candidates can reach the global top-k; drop the rest before
  // gathering any columns.
  if (kept->num_rows() > k) {
    auto order = cudf::top_k_order(kept->get_column(2).view(),
                                   static_cast<cudf::size_type>(k),
                                   cudf::order::ASCENDING,
                                   stream,
                                   mr);
    kept =
      cudf::gather(kept->view(), order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  }
  auto kept_cols = kept->release();

  auto cols =
    cudf::gather(left, kept_cols[0]->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr)
      ->release();
  auto right_cols =
    cudf::gather(right, kept_cols[1]->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr)
      ->release();
  for (auto& c : right_cols) {
    cols.push_back(std::move(c));
  }
  auto ranking = std::move(kept_cols[2]);
  if (is_similarity) {
    // cuVS ranks cosine by distance; a similarity query reports `1 - distance`.
    cudf::numeric_scalar<float> one_f(1.0F, true, stream);
    ranking = cudf::binary_operation(one_f,
                                     ranking->view(),
                                     cudf::binary_operator::SUB,
                                     cudf::data_type{cudf::type_id::FLOAT32},
                                     stream,
                                     mr);
  }
  cols.push_back(std::move(ranking));
  return finish(std::make_unique<cudf::table>(std::move(cols)));
}

void sirius_physical_vector_topk_join::sink(const operator_data& output_data,
                                            ::cuda::stream_ref stream)
{
  if (scope == topk_scope::global) {
    sirius_physical_partition_consumer_operator::sink(output_data, stream);
    return;
  }
  auto const& output  = dynamic_cast<const topk_partial_output&>(output_data);
  auto const& batches = output.get_data_batches();
  for (std::size_t i = 0; i < batches.size(); ++i) {
    for (auto& next_port_info : next_port_after_sink) {
      auto* consumer =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (!consumer) {
        throw std::runtime_error(
          "sirius_physical_vector_topk_join::sink: next operator is not a partition consumer");
      }
      consumer->push_data_batch_partitioned(
        next_port_info.next_operator_port_name, batches[i], output.partitions[i]);
    }
  }
}

}  // namespace op
}  // namespace sirius
