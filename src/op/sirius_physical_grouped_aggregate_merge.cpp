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
#include "op/sirius_physical_grouped_aggregate_merge.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

// Helpers create_group_chunk_types / copy_expressions were used by the original grouping-sets
// initialization path (now dead) and by the merge clone-from-parent ctor (which now takes pre-
// converted cuDF definitions instead of DuckDB expressions). Both helpers have no remaining
// callers in Super Sirius and have been removed.

// Helper to convert vector<vector<idx_t>> to vector<unsafe_vector<idx_t>>
[[maybe_unused]] static duckdb::vector<duckdb::unsafe_vector<std::size_t>>
convert_grouping_functions(const duckdb::vector<duckdb::vector<std::size_t>>& src)
{
  duckdb::vector<duckdb::unsafe_vector<std::size_t>> result;
  result.reserve(src.size());
  for (const auto& inner : src) {
    duckdb::unsafe_vector<std::size_t> converted;
    for (auto val : inner) {
      converted.push_back(val);
    }
    result.push_back(std::move(converted));
  }
  return result;
}

void sirius_physical_grouped_aggregate_merge::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // The child sink still creates the upstream pipeline boundary.
  if (fuse_into_parent()) {
    D_ASSERT(children.size() == 1);
    meta_pipeline.get_state().add_pipeline_operator(current, *this);
    children[0]->build_pipelines(current, meta_pipeline);
    return;
  }
  sirius_physical_operator::build_pipelines(current, meta_pipeline);
}

sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  sirius_physical_grouped_aggregate* grouped_aggregate, uint64_t hash_partition_bytes)
  : sirius_physical_grouped_aggregate_merge(grouped_aggregate->types,
                                            grouped_aggregate->group_idx,
                                            grouped_aggregate->cudf_aggregates,
                                            grouped_aggregate->cudf_aggregate_idx,
                                            grouped_aggregate->cudf_aggregate_struct_col_indices,
                                            grouped_aggregate->aggregate_slots,
                                            grouped_aggregate->has_avg,
                                            grouped_aggregate->has_count_distinct,
                                            grouped_aggregate->estimated_cardinality)
{
  child_op              = grouped_aggregate;
  _hash_partition_bytes = hash_partition_bytes;
  if (grouped_aggregate->surrogate_spec) {
    // Surrogate-key deferral: the wrapped aggregate emits rowid/dummy key carriers, but this
    // merge finalizes them back to the original string keys, so it declares (and produces)
    // the original schema.
    surrogate_spec = grouped_aggregate->surrogate_spec;
    types          = surrogate_spec->original_output_types;
  }
}

sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  duckdb::vector<sirius::logical_type> types,
  std::vector<int> group_idx,
  std::vector<cudf::aggregation::Kind> cudf_aggregates,
  std::vector<int> cudf_aggregate_idx,
  std::vector<std::vector<int>> cudf_aggregate_struct_col_indices,
  std::vector<AggregateSlot> aggregate_slots,
  bool has_avg,
  bool has_count_distinct,
  std::size_t estimated_cardinality)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::MERGE_GROUP_BY, std::move(types), estimated_cardinality),
    group_idx(std::move(group_idx)),
    cudf_aggregates(std::move(cudf_aggregates)),
    cudf_aggregate_idx(std::move(cudf_aggregate_idx)),
    cudf_aggregate_struct_col_indices(std::move(cudf_aggregate_struct_col_indices)),
    aggregate_slots(std::move(aggregate_slots)),
    has_avg(has_avg),
    has_count_distinct(has_count_distinct)
{
}

sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  std::size_t estimated_cardinality)
  : sirius_physical_grouped_aggregate_merge(std::move(types),
                                            std::move(expressions),
                                            std::move(groups_p),
                                            {},
                                            {},
                                            estimated_cardinality,
                                            duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                            duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)
{
}

// expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression
// to a column groups_p is the list of group by columns. Each group by column is a bound_ref
// expression to a column grouping_sets_p is the list of grouping set. Each grouping set is a set of
// indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets
// and for every grouping set there is one radix_table grouping_functions_p is a list of indexes to
// the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the
// grouping set and the second level is the indexes to the groupby expression for that set.
sirius_physical_grouped_aggregate_merge::sirius_physical_grouped_aggregate_merge(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<std::size_t>> /*grouping_functions_p*/,
  std::size_t estimated_cardinality,
  duckdb::TupleDataValidityType /*group_validity*/,
  duckdb::TupleDataValidityType /*distinct_validity*/)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::MERGE_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  // Convert input parameters to cudf compute definitions BEFORE moving them
  auto cudf_defs                    = convert_duckdb_aggregates_to_cudf(groups_p, expressions);
  group_idx                         = std::move(cudf_defs.group_idx);
  cudf_aggregates                   = std::move(cudf_defs.cudf_aggregates);
  cudf_aggregate_idx                = std::move(cudf_defs.cudf_aggregate_idx);
  cudf_aggregate_struct_col_indices = std::move(cudf_defs.cudf_aggregate_struct_col_indices);
  aggregate_slots                   = std::move(cudf_defs.aggregate_slots);
  has_avg                           = cudf_defs.has_avg;
  has_count_distinct                = cudf_defs.has_count_distinct;
}

partition_strategy sirius_physical_grouped_aggregate_merge::get_partition_strategy(
  const partition_sizing_input& in)
{
  int const natural = natural_num_partitions(in.total_bytes, _hash_partition_bytes, _num_gpus);
  // Pre-size this merge's single input repository so every partition slot exists before batches
  // arrive (grouping is never broadcast / build-probe). Guarded on strictly-greater to respect the
  // repository's set_num_partitions contract.
  if (natural > 1) {
    std::lock_guard<std::mutex> lg(lock);
    if (!ports.empty()) {
      auto& repo = ports.begin()->second->repo;
      if (repo != nullptr && static_cast<std::size_t>(natural) > repo->num_partitions()) {
        repo->set_num_partitions(static_cast<std::size_t>(natural));
      }
    }
  }
  return {natural, /*broadcast=*/false, /*build_probe=*/false};
}

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate_merge::get_next_task_input_data()
{
  // we need to lock, then pull all the batches from one partition and return them, and increment
  // the partition index
  std::lock_guard<std::mutex> lg(lock);
  if (current_partition_index < ports.begin()->second->repo->num_partitions()) {
    std::vector<::std::shared_ptr<::cucascade::data_batch>> input_batch;
    bool found_batch       = true;
    auto this_partition_id = current_partition_index;
    while (found_batch) {
      auto batch = ports.begin()->second->repo->pop_next_data_batch(current_partition_index);
      if (batch) {
        input_batch.push_back(std::move(batch));
      } else {
        found_batch = false;
      }
    }
    current_partition_index++;
    if (input_batch.empty()) { return nullptr; }
    // Tag with the source partition index so the scheduler pins this task to
    // partition_idx % num_gpus. merge_group_by materializes a cuco hash table
    // to combine its input batches, so — like hash_join — every task of a
    // given partition must stay on a single GPU.
    return std::make_unique<partitioned_operator_data>(std::move(input_batch), this_partition_id);
  } else {
    return nullptr;
  }
}

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate_merge::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_grouped_aggregate_merge::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (input_batches.size() == 0) {
    throw std::runtime_error(
      "We expect at least one input batch for grouped aggregate merge operator");
  }

  // Fast path: single batch with no post-processing needed (surrogate-key deferral always
  // needs the finalization below, so it never takes this exit).
  if (input_batches.size() == 1 && !has_avg && !has_count_distinct && !surrogate_spec) {
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

  // Merge multiple batches, or use single batch directly if only one
  std::shared_ptr<::cucascade::data_batch> merged;
  if (input_batches.size() == 1) {
    const auto clone_batch_id = sirius::get_next_batch_id();
    merged                    = input_batches[0].clone(
      clone_batch_id,
      stream,
      telemetry::quent_data_batch_probe::create(batch_telemetry(), clone_batch_id));
  } else {
    merged = gpu_merge_impl::merge_grouped_aggregate(input_batches,
                                                     group_idx.size(),
                                                     cudf_aggregates,
                                                     stream,
                                                     *input_batches[0].get_memory_space(),
                                                     batch_telemetry());
  }

  // Surrogate-key deferral: materialize the deferred string keys and restore the original
  // schema before any AVG / COUNT DISTINCT post-processing (which only touches aggregate
  // slots, never key slots).
  if (surrogate_spec) { merged = finalize_surrogate_groupby(std::move(merged), stream); }

  // If no post-processing needed, return merged result directly
  if (!has_avg && !has_count_distinct) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<::cucascade::data_batch>>{merged});
  }

  // Post-merge projection: handle AVG (SUM/COUNT) and COUNT DISTINCT (list element count).
  // Release ownership of the merged table's columns so we can move (not copy) them.
  // Acquire EXCLUSIVE lock since release_table() is a mutating operation
  auto merged_mut    = merged->to_mutable();
  auto* space        = merged_mut.get_memory_space();
  auto mr            = space->get_default_allocator();
  auto& gpu_rep      = merged_mut.get_data()->cast<cucascade::gpu_table_representation>();
  auto merged_cols   = gpu_rep.release_table(stream)->release();
  int num_group_cols = static_cast<int>(group_idx.size());

  std::vector<std::unique_ptr<cudf::column>> output_cols;

  // Move group key columns (zero-copy)
  for (int i = 0; i < num_group_cols; ++i) {
    output_cols.push_back(std::move(merged_cols[i]));
  }

  // Process each original aggregate
  for (auto const& slot : aggregate_slots) {
    if (slot.is_avg) {
      int sum_col_idx   = num_group_cols + static_cast<int>(slot.cudf_idx);
      int count_col_idx = num_group_cols + static_cast<int>(slot.cudf_idx) + 1;

      auto sum_view   = merged_cols[sum_col_idx]->view();
      auto count_view = merged_cols[count_col_idx]->view();

      std::unique_ptr<cudf::column> avg_col;
      bool is_decimal = sirius::IsCudfTypeDecimal(slot.output_type);
      if (is_decimal) {
        // DECIMAL: divide directly in fixed-point to preserve precision
        avg_col = cudf::binary_operation(
          sum_view, count_view, cudf::binary_operator::DIV, slot.output_type, stream, mr);
      } else {
        // Non-DECIMAL: cast to FLOAT64 and divide
        auto sum_f64 = cudf::cast(sum_view, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
        auto count_f64 =
          cudf::cast(count_view, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
        avg_col = cudf::binary_operation(sum_f64->view(),
                                         count_f64->view(),
                                         cudf::binary_operator::DIV,
                                         cudf::data_type{cudf::type_id::FLOAT64},
                                         stream,
                                         mr);
      }

      output_cols.push_back(std::move(avg_col));
    } else if (slot.is_count_distinct) {
      // The merged column is a LIST column (output of MERGE_SETS). Count elements per row to
      // produce the final distinct count, then cast to INT64.
      int col_idx      = num_group_cols + static_cast<int>(slot.cudf_idx);
      auto list_view   = cudf::lists_column_view(merged_cols[col_idx]->view());
      auto count_int32 = cudf::lists::count_elements(list_view, stream, mr);
      auto count_int64 =
        cudf::cast(count_int32->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
      output_cols.push_back(std::move(count_int64));
    } else {
      // Move non-AVG, non-count-distinct aggregate columns directly (zero-copy)
      int col_idx = num_group_cols + static_cast<int>(slot.cudf_idx);
      output_cols.push_back(std::move(merged_cols[col_idx]));
    }
  }

  auto output_table = std::make_unique<cudf::table>(std::move(output_cols));
  auto result = sirius::make_data_batch(std::move(output_table), *space, stream, batch_telemetry());
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{result});
}

void sirius_physical_grouped_aggregate_merge::on_finalize_operator()
{
  // All merge tasks are done: drop the surrogate store's retained source accessors so the
  // batches become downgradable/freeable for whatever remains of the query.
  if (surrogate_spec && surrogate_spec->store) {
    auto const [count, bytes] = surrogate_spec->store->release();
    if (count > 0) {
      SIRIUS_LOG_INFO(
        "groupby_surrogate_keys: released {} retained source batch accessor(s) ({} bytes) after "
        "merge finalization",
        count,
        bytes);
    }
  }
}

std::shared_ptr<::cucascade::data_batch>
sirius_physical_grouped_aggregate_merge::finalize_surrogate_groupby(
  std::shared_ptr<::cucascade::data_batch> merged, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"grouped_aggregate_merge::finalize_surrogate"};
  auto const& spec = *surrogate_spec;

  auto merged_mut = merged->to_mutable();
  auto* space     = merged_mut.get_memory_space();
  auto mr         = space->get_default_allocator();
  auto& gpu_rep   = merged_mut.get_data()->cast<cucascade::gpu_table_representation>();
  auto cols       = gpu_rep.release_table(stream)->release();

  auto const num_rows = cols.empty() ? 0 : cols[0]->size();

  // Fast path proof: if the real (non-deferred) key columns are already distinct across the
  // merged rows, every full tuple is distinct, so grouping by surrogate produced exactly the
  // tuple grouping and no re-group is needed. The check is EXACT (cudf::distinct_count with
  // nulls equal, matching groupby null_policy::INCLUDE semantics), so the fast path is safe by
  // construction; when the proof fails we fall through to the conservative full-tuple re-group.
  //
  // NaN parity guard: SQL GROUP BY treats all NaNs as one group, but distinct_count's row
  // comparator's NaN semantics are not contractually documented, so the proof comparator might
  // be FINER than the grouping's on floating-point keys (two NaN rows counted distinct would
  // fake a proof and leak a duplicate group). Floating-point real keys therefore always take
  // the conservative re-group.
  bool tuples_proven_distinct = (num_rows == 0);
  if (!tuples_proven_distinct && spec.unique_fastpath && !spec.real_key_slots.empty()) {
    bool has_floating_point_key = false;
    std::vector<cudf::column_view> check_cols;
    check_cols.reserve(spec.real_key_slots.size());
    for (int slot : spec.real_key_slots) {
      auto view = cols.at(static_cast<std::size_t>(slot))->view();
      if (view.type().id() == cudf::type_id::FLOAT32 ||
          view.type().id() == cudf::type_id::FLOAT64) {
        has_floating_point_key = true;
        break;
      }
      check_cols.push_back(view);
    }
    if (!has_floating_point_key) {
      auto const distinct =
        cudf::distinct_count(cudf::table_view(check_cols), cudf::null_equality::EQUAL, stream);
      tuples_proven_distinct = (distinct == num_rows);
    }
  }

  // Materialize the deferred string key columns: per deferral-join side, gather the retained
  // source columns at the merged rowids. Sources are concatenated in base order, which
  // reproduces the absolute rowid address space exactly (bases are contiguous by construction
  // and reserve/commit is deduplicated per batch id).
  for (auto const& group : spec.groups) {
    if (num_rows == 0) {
      for (std::size_t i = 0; i < group.restore_key_slots.size(); ++i) {
        cols.at(static_cast<std::size_t>(group.restore_key_slots[i])) =
          cudf::make_empty_column(sirius::get_cudf_type(group.restored_types[i]));
      }
      continue;
    }

    auto const sources = spec.store->snapshot(group.from_left);
    std::vector<cudf::table_view> pieces;
    pieces.reserve(sources.size());
    for (auto const& src : sources) {
      // STREAM-LINEAGE: the retained batches were written on the deferral join's streams; order
      // this task's stream after each writer event before reading their memory.
      if (auto const writer_event = src.batch.get_writer_event(); writer_event != nullptr) {
        auto const status = cudaStreamWaitEvent(stream.value(), writer_event, 0);
        if (status != cudaSuccess) {
          throw std::runtime_error(
            std::string("finalize_surrogate_groupby: writer-event wait failed: ") +
            cudaGetErrorString(status));
        }
      }
      pieces.push_back(sirius::get_cudf_table_view(src.batch).select(group.source_input_cols));
    }
    if (pieces.empty()) {
      throw std::runtime_error(
        "finalize_surrogate_groupby: merged rows reference deferred string sources but none "
        "were registered by the deferral join");
    }
    std::unique_ptr<cudf::table> src_owned;
    cudf::table_view src_view;
    if (pieces.size() == 1) {
      src_view = pieces[0];
    } else {
      src_owned = cudf::concatenate(pieces, stream, mr);
      src_view  = src_owned->view();
    }

    auto const& rowid_col = cols.at(static_cast<std::size_t>(group.rowid_key_slot));
    auto map32 = cudf::cast(rowid_col->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);
    auto gathered =
      cudf::gather(src_view, map32->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    auto gathered_cols = gathered->release();
    for (std::size_t i = 0; i < group.restore_key_slots.size(); ++i) {
      cols.at(static_cast<std::size_t>(group.restore_key_slots[i])) = std::move(gathered_cols[i]);
    }
  }

  // Conservative path: re-group by the full restored tuple, re-combining the (composable)
  // partial aggregates. Only reached when the distinct proof failed (duplicate full tuples may
  // exist) or the fast path is disabled.
  if (!tuples_proven_distinct && num_rows > 0) {
    SIRIUS_LOG_DEBUG(
      "finalize_surrogate_groupby: distinct proof failed or fast path disabled; re-grouping {} "
      "rows by the full restored tuple",
      num_rows);
    std::vector<cudf::column_view> key_views;
    key_views.reserve(group_idx.size());
    for (std::size_t i = 0; i < group_idx.size(); ++i) {
      key_views.push_back(cols[i]->view());
    }
    cudf::groupby::groupby grpby_obj(cudf::table_view(key_views), cudf::null_policy::INCLUDE);
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.reserve(cudf_aggregates.size());
    for (std::size_t i = 0; i < cudf_aggregates.size(); ++i) {
      cudf::groupby::aggregation_request request;
      request.values = cols.at(group_idx.size() + i)->view();
      switch (cudf_aggregates[i]) {
        case cudf::aggregation::Kind::MIN:
          request.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
          break;
        case cudf::aggregation::Kind::MAX:
          request.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
          break;
        case cudf::aggregation::Kind::SUM:
        case cudf::aggregation::Kind::COUNT_ALL:
        case cudf::aggregation::Kind::COUNT_VALID:
          request.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
          break;
        default:
          throw std::runtime_error(
            "finalize_surrogate_groupby: unsupported aggregate kind for re-group (planner "
            "invariant violated): " +
            std::to_string(static_cast<int>(cudf_aggregates[i])));
      }
      requests.push_back(std::move(request));
    }
    auto groupby_result = grpby_obj.aggregate(requests, stream, mr);
    auto new_cols       = groupby_result.first->release();
    for (auto& aggregation_result : groupby_result.second) {
      new_cols.push_back(std::move(aggregation_result.results[0]));
    }
    cols = std::move(new_cols);
  }

  auto output_table = std::make_unique<cudf::table>(std::move(cols));
  return sirius::make_data_batch(std::move(output_table), *space, stream, batch_telemetry());
}

}  // namespace op
}  // namespace sirius
