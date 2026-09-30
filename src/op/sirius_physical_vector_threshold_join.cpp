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

#include "op/sirius_physical_vector_threshold_join.hpp"

#include "cuda/vss/brute_force_threshold.hpp"
#include "cuda/vss/cudf_raft_interop.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
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
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <raft/core/device_resources.hpp>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <vector>

namespace sirius {
namespace op {

sirius_physical_vector_threshold_join::sirius_physical_vector_threshold_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  std::size_t left_vector_col_idx,
  std::size_t right_vector_col_idx,
  float cutoff,
  std::string metric,
  std::int64_t dim,
  duckdb::JoinType join_type,
  std::size_t estimated_cardinality,
  uint64_t batch_bytes)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::VECTOR_THRESHOLD_JOIN,
                                                sirius::from_duckdb_vec(op.types),
                                                estimated_cardinality),
    left_vector_col_idx(left_vector_col_idx),
    right_vector_col_idx(right_vector_col_idx),
    cutoff(cutoff),
    metric(std::move(metric)),
    dim(dim),
    join_type(join_type),
    batch_bytes(batch_bytes)
{
  children.push_back(std::move(left));
  children.push_back(std::move(right));
  auto const n_left  = children[0]->get_types().size();
  auto const n_right = children[1]->get_types().size();
  auto const& join   = op.Cast<duckdb::LogicalJoin>();
  auto fill          = [](const duckdb::vector<duckdb::idx_t>& projection_map,
                 std::size_t n_cols,
                 duckdb::vector<std::size_t>& out) {
    if (projection_map.empty()) {
      out.reserve(n_cols);
      for (std::size_t i = 0; i < n_cols; i++) {
        out.push_back(i);
      }
    } else {
      out.reserve(projection_map.size());
      for (auto idx : projection_map) {
        out.push_back(static_cast<std::size_t>(idx));
      }
    }
  };
  fill(join.left_projection_map, n_left, left_output_col_idxs);
  fill(join.right_projection_map, n_right, right_output_col_idxs);
}

void sirius_physical_vector_threshold_join::enable_distance_output(bool as_similarity)
{
  // Idempotent: the planner may match more than one distance reference in the same SELECT list.
  if (emit_distance_) { return; }
  emit_distance_               = true;
  emit_distance_as_similarity_ = as_similarity;
  // The join carries no physical-type overrides, so the logical types fully describe the output
  // batches. Append one FLOAT column for the distance; it lands at output index n_left + n_right,
  // which is what the planner rewrites its distance reference to.
  D_ASSERT(!has_physical_overrides());
  types.push_back(sirius::from_duckdb(duckdb::LogicalType::FLOAT));
}

void sirius_physical_vector_threshold_join::set_output_row_count_only()
{
  if (output_row_count_only_) { return; }
  // Only an ungrouped count_star sets this, and it never also reuses the distance column.
  D_ASSERT(!emit_distance_);
  output_row_count_only_ = true;
  // execute() emits number of result rows, so shrink the declared output schema to match.
  types.clear();
  // cuDF INT8 for DuckDB TINYINT.
  types.push_back(sirius::from_duckdb(duckdb::LogicalType::TINYINT));
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_vector_threshold_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Mirrors sirius_physical_nested_loop_join::build_pipelines.
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

partition_strategy sirius_physical_vector_threshold_join::get_partition_strategy(
  const partition_sizing_input& /*in*/)
{
  // Streams both sides through per-batch-pair threshold joins on a single partition; the answer is
  // the union of the per-pair edge lists, so it never hash-partitions, broadcasts, or build-probes.
  return {/*num_partitions=*/1, /*broadcast=*/false, /*build_probe=*/false};
}

std::string_view sirius_physical_vector_threshold_join::input_port_for(
  sirius_physical_operator const& producer) const
{
  if (producer.type == SiriusPhysicalOperatorType::CONCAT) {
    return producer.Cast<sirius_physical_concat>().is_build_concat() ? "build" : "default";
  }
  return sirius_physical_operator::input_port_for(producer);
}

std::unique_ptr<operator_data> sirius_physical_vector_threshold_join::get_next_task_input_data()
{
  // Enumerate every (left batch, right batch) pair, popping the last consumer of each batch with
  // a persistent cursor advances one pair per call.
  std::scoped_lock lg(batches_to_processed_mutex);

  auto* default_port = get_port("default");
  auto* build_port   = get_port("build");

  if (!ids_initialized_) {
    if (!default_port || !default_port->repo || !build_port || !build_port->repo) {
      return nullptr;
    }
    if (default_port->repo->num_partitions() != build_port->repo->num_partitions()) {
      throw std::runtime_error(
        "sirius_physical_vector_threshold_join: number of partitions for default and build ports "
        "must match");
    }
    left_batch_ids.reserve(default_port->repo->num_partitions());
    right_batch_ids.reserve(build_port->repo->num_partitions());
    for (size_t i = 0; i < default_port->repo->num_partitions(); i++) {
      left_batch_ids.push_back(default_port->repo->get_batch_ids(i));
      right_batch_ids.push_back(build_port->repo->get_batch_ids(i));
    }
    ids_initialized_ = true;
    // Park the cursor on the first partition that actually has a pair to process.
    while (
      cursor_partition_ < left_batch_ids.size() &&
      (left_batch_ids[cursor_partition_].empty() || right_batch_ids[cursor_partition_].empty())) {
      cursor_partition_++;
    }
  }

  if (cursor_partition_ >= left_batch_ids.size()) { return nullptr; }

  auto const p          = cursor_partition_;
  auto const li         = cursor_left_;
  auto const ri         = cursor_right_;
  auto const& lids      = left_batch_ids[p];
  auto const& rids      = right_batch_ids[p];
  bool const last_right = ri + 1 == rids.size();
  bool const last_left  = li + 1 == lids.size();

  std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
  input_batch.reserve(2);
  // Left batch is reused across every right batch, so release it only on the last right.
  input_batch.push_back(last_right ? default_port->repo->pop_data_batch_by_id(lids[li], p)
                                   : default_port->repo->get_data_batch_by_id(lids[li], p));
  // Right batch is reused across every left batch, so release it only on the last left.
  input_batch.push_back(last_left ? build_port->repo->pop_data_batch_by_id(rids[ri], p)
                                  : build_port->repo->get_data_batch_by_id(rids[ri], p));

  // Advance one pair: inner over right, then left, then skip to the next non-empty partition.
  cursor_right_++;
  if (cursor_right_ >= rids.size()) {
    cursor_right_ = 0;
    cursor_left_++;
  }
  if (cursor_left_ >= lids.size()) {
    cursor_left_ = 0;
    cursor_partition_++;
    while (
      cursor_partition_ < left_batch_ids.size() &&
      (left_batch_ids[cursor_partition_].empty() || right_batch_ids[cursor_partition_].empty())) {
      cursor_partition_++;
    }
  }

  return std::make_unique<pipelineable_operator_data>(std::move(input_batch));
}

std::unique_ptr<operator_data> sirius_physical_vector_threshold_join::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_threshold_join::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (input_batches.size() != 2) {
    throw std::runtime_error(
      "sirius_physical_vector_threshold_join expects 2 input batches (left, right), got " +
      std::to_string(input_batches.size()));
  }

  auto const& left_batch  = input_batches[0];
  auto const& right_batch = input_batches[1];

  cudf::table_view left  = get_cudf_table_view(left_batch);
  cudf::table_view right = get_cudf_table_view(right_batch);

  cucascade::memory::memory_space* space = left_batch.get_memory_space();
  if (!space) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }
  auto mr = space->get_default_allocator();

  // Collects the output batches this call emits.
  std::vector<std::shared_ptr<cucascade::data_batch>> out_batches;

  // Assemble one output table from a left gather-map and an equal-length right gather-map.
  auto assemble = [&](cudf::column_view const& left_map,
                      cudf::column_view const& right_map,
                      cudf::out_of_bounds_policy right_policy,
                      std::unique_ptr<cudf::column> distance_col) {
    if (output_row_count_only_) {
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT8},
                                               left_map.size(),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               mr));
      return std::make_unique<cudf::table>(std::move(cols));
    }
    auto left_gathered =
      cudf::gather(left, left_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    auto right_gathered = cudf::gather(right, right_map, right_policy, stream, mr);
    auto left_released  = left_gathered->release();
    auto right_released = right_gathered->release();
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(left_output_col_idxs.size() + right_output_col_idxs.size() +
                 (emit_distance_ ? 1 : 0));
    for (std::size_t idx : left_output_col_idxs) {
      if (idx < left_released.size()) { cols.push_back(std::move(left_released[idx])); }
    }
    for (std::size_t idx : right_output_col_idxs) {
      if (idx < right_released.size()) { cols.push_back(std::move(right_released[idx])); }
    }
    if (emit_distance_) { cols.push_back(std::move(distance_col)); }
    return std::make_unique<cudf::table>(std::move(cols));
  };

  // Emit one schema-correct zero-row batch. The join must always produce at least one batch: a
  // downstream aggregate pipeline (e.g. count(*)/sum over the join) waits on the source's batches
  // to finalize, so a source that emits nothing never completes and the query deadlocks. This
  // mirrors how the scan path signals end-of-input with an empty batch.
  auto emit_empty_output = [&]() {
    auto empty_map = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
    std::unique_ptr<cudf::column> distance_col;
    if (emit_distance_) {
      distance_col = cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32});
    }
    out_batches.push_back(make_data_batch(assemble(empty_map->view(),
                                                   empty_map->view(),
                                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                                   std::move(distance_col)),
                                          *space,
                                          stream,
                                          batch_telemetry()));
  };

  bool const is_left     = join_type == duckdb::JoinType::LEFT;
  auto const n_left      = static_cast<cudf::size_type>(left.num_rows());
  bool const right_empty = right.num_rows() == 0;

  // No left rows -> no output. An empty right side yields no matches: INNER emits nothing, while
  // LEFT still emits every left row padded with NULLs (handled by the unmatched pass below).
  // Either way emit one empty batch so a downstream aggregate can finalize.
  if (n_left == 0 || (right_empty && !is_left)) {
    emit_empty_output();
    return std::make_unique<pipelineable_operator_data>(std::move(out_batches));
  }

  // Sizes its own output batches and caps each emitted batch by the engine's byte budget, so
  // a large edge list becomes several batches.
  auto const left_row_bytes =
    (left_batch.get_data() && n_left > 0)
      ? left_batch.get_data()->get_size_in_bytes() / static_cast<std::size_t>(n_left)
      : std::size_t{0};
  auto const right_row_bytes =
    (right_batch.get_data() && right.num_rows() > 0)
      ? right_batch.get_data()->get_size_in_bytes() / static_cast<std::size_t>(right.num_rows())
      : std::size_t{0};
  // Row-count-only emits a 1-byte carrier, so its per-row size and column-length cap are set by the
  // carrier, not by the (unused) vector row width.
  auto const bytes_per_row =
    output_row_count_only_
      ? std::size_t{1}
      : std::max<std::size_t>(
          1, left_row_bytes + right_row_bytes + (emit_distance_ ? sizeof(float) : 0));
  auto const budget_rows = std::max<std::size_t>(1, batch_bytes / bytes_per_row);
  auto const size_type_cap =
    (!output_row_count_only_ && dim > 0)
      ? static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max() / dim)
      : static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  auto const max_rows = std::min(budget_rows, size_type_cap);

  auto emit = [&](cudf::column_view const& left_map,
                  cudf::column_view const& right_map,
                  cudf::out_of_bounds_policy right_policy,
                  std::unique_ptr<cudf::column> distance_col) {
    auto const total = static_cast<std::size_t>(left_map.size());
    if (total <= max_rows) {
      out_batches.push_back(
        make_data_batch(assemble(left_map, right_map, right_policy, std::move(distance_col)),
                        *space,
                        stream,
                        batch_telemetry()));
      return;
    }
    for (std::size_t start = 0; start < total; start += max_rows) {
      auto const s = static_cast<cudf::size_type>(start);
      auto const e = static_cast<cudf::size_type>(std::min(total, start + max_rows));
      auto lm      = cudf::slice(left_map, {s, e}).front();
      auto rm      = cudf::slice(right_map, {s, e}).front();
      std::unique_ptr<cudf::column> dcol;
      if (emit_distance_ && distance_col) {
        dcol = std::make_unique<cudf::column>(
          cudf::slice(distance_col->view(), {s, e}).front(), stream, mr);
      }
      out_batches.push_back(make_data_batch(
        assemble(lm, rm, right_policy, std::move(dcol)), *space, stream, batch_telemetry()));
    }
  };

  // LEFT join needs to know which left rows matched. Build the flag up front, even when the right
  // side is empty, so every left row falls through the unmatched pass below. It is filled in per
  // query tile as edges are produced.
  std::unique_ptr<cudf::column> matched_flag;
  if (is_left) {
    cudf::numeric_scalar<bool> false_scalar(false, true, stream);
    matched_flag = cudf::make_column_from_scalar(false_scalar, n_left, stream, mr);
  }

  if (!right_empty) {
    auto const dataset = vss::list_column_as_dataset_view(right.column(right_vector_col_idx), dim);
    raft::device_resources res{stream};
    auto const metric_type =
      vss::join_selection_distance_type_from_metric(metric, /*exact_unexpanded=*/false);

    // Bound each threshold call so its edge list can never exceed a cudf column's int32 length.
    // A single query row produces at most n_right edges, so keeping query_tile * n_right under the
    // cap keeps every call's edge columns representable.
    auto const n_left_sz = static_cast<std::size_t>(n_left);
    auto const n_right   = static_cast<std::size_t>(right.num_rows());
    auto const edge_cap  = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
    auto query_tile      = std::max<std::size_t>(1, edge_cap / std::max<std::size_t>(1, n_right));
    // Test hook
    if (const char* env = std::getenv("SIRIUS_VSS_QUERY_TILE_ROWS")) {
      auto const forced = std::strtoull(env, nullptr, 10);
      if (forced > 0) { query_tile = std::min<std::size_t>(query_tile, forced); }
    }

    // The queries are a window into the full left column.
    auto const queries_full =
      vss::list_column_as_dataset_view(left.column(left_vector_col_idx), dim);

    for (std::size_t s = 0; s < n_left_sz; s += query_tile) {
      auto const qs      = std::min(query_tile, n_left_sz - s);
      auto const queries = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        queries_full.data_handle() + static_cast<int64_t>(s) * dim, static_cast<int64_t>(qs), dim);
      auto tj = vss::brute_force_threshold(res, dataset, queries, cutoff, metric_type, mr);
      if (tj.query_rows->size() == 0) { continue; }

      // The kernel numbers query rows within the tile (0..qs); shift them to global left indices so
      // the left gather map addresses the whole left batch.
      std::unique_ptr<cudf::column> left_map_global;
      if (s == 0) {
        left_map_global = std::move(tj.query_rows);
      } else {
        cudf::numeric_scalar<int64_t> off(static_cast<int64_t>(s), true, stream);
        left_map_global = cudf::binary_operation(tj.query_rows->view(),
                                                 off,
                                                 cudf::binary_operator::ADD,
                                                 cudf::data_type{cudf::type_id::INT64},
                                                 stream,
                                                 mr);
      }

      std::unique_ptr<cudf::column> distance_col;
      if (emit_distance_) {
        if (emit_distance_as_similarity_) {
          cudf::numeric_scalar<float> one(1.0F, true, stream);
          distance_col = cudf::binary_operation(one,
                                                tj.distances->view(),
                                                cudf::binary_operator::SUB,
                                                cudf::data_type{cudf::type_id::FLOAT32},
                                                stream,
                                                mr);
        } else {
          distance_col = std::move(tj.distances);
        }
      }

      emit(left_map_global->view(),
           tj.neighbors->view(),
           cudf::out_of_bounds_policy::DONT_CHECK,
           std::move(distance_col));

      if (is_left) {
        cudf::numeric_scalar<bool> true_scalar(true, true, stream);
        auto scattered = cudf::scatter({std::ref(static_cast<cudf::scalar const&>(true_scalar))},
                                       left_map_global->view(),
                                       cudf::table_view({matched_flag->view()}),
                                       stream);
        matched_flag   = std::move(scattered->release()[0]);
      }
    }
  }

  if (is_left) {
    // matched_flag was accumulated across the query tiles above; a left row is unmatched wherever
    // it stayed false.
    auto unmatched_mask =
      cudf::unary_operation(matched_flag->view(), cudf::unary_operator::NOT, stream, mr);
    // Local left row indices that survived (unmatched).
    cudf::numeric_scalar<cudf::size_type> zero(0, true, stream);
    cudf::numeric_scalar<cudf::size_type> one(1, true, stream);
    auto seq = cudf::sequence(n_left, zero, one, stream, mr);
    auto unmatched_idx_tbl =
      cudf::apply_boolean_mask(cudf::table_view({seq->view()}), unmatched_mask->view(), stream, mr);
    auto unmatched_idx = unmatched_idx_tbl->get_column(0).view();
    // All null right columns: gather every right row by an out-of-range index under NULLIFY.
    cudf::numeric_scalar<cudf::size_type> oob(
      static_cast<cudf::size_type>(right.num_rows()), true, stream);
    auto pad = cudf::make_column_from_scalar(
      oob, static_cast<cudf::size_type>(unmatched_idx.size()), stream, mr);
    // Unmatched left rows have no partner, so their distance is NULL.
    std::unique_ptr<cudf::column> distance_col;
    if (emit_distance_) {
      distance_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32},
                                               static_cast<cudf::size_type>(unmatched_idx.size()),
                                               cudf::mask_state::ALL_NULL,
                                               stream,
                                               mr);
    }
    emit(unmatched_idx, pad->view(), cudf::out_of_bounds_policy::NULLIFY, std::move(distance_col));
  }

  // An INNER join with no matches produces no edges above, so emit one empty batch to signal
  // end-of-input to a downstream aggregate.
  if (out_batches.empty()) { emit_empty_output(); }

  return std::make_unique<pipelineable_operator_data>(std::move(out_batches));
}

}  // namespace op
}  // namespace sirius
