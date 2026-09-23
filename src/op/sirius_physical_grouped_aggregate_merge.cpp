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
#include "expression/ast/reference.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/group_by_bypass_policy.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_projection.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <string>
#include <vector>

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

namespace {

/// Fixed-width integral types the v1 bypass model covers. Floats and decimals are excluded
/// deliberately: their merge-time partial states are not modelled here (see
/// docs/super-sirius/group-by-bypass.md).
[[nodiscard]] bool bypass_supported_column_type(int type_id) noexcept
{
  switch (static_cast<cudf::type_id>(type_id)) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: return true;
    default: return false;
  }
}

/// Whether the path from @p merge to the first downstream sink is a bounded result-collection
/// path: unary, non-expanding, and terminating at a RESULT_COLLECTOR.
///
/// Anything that buffers or reshapes — TOP_N, ORDER_BY, a second GROUP BY, a join, a further
/// PARTITION — makes the merge's output feed work this model does not account for, so v1 declines
/// rather than guessing. Along an accepted path, `row_bytes`/`columns` total the columns every
/// step materializes while the merge output is still live.
struct downstream_shape {
  bool supported          = false;
  std::uint64_t row_bytes = 0;
  std::size_t columns     = 0;
};

/// Walk from @p merge to its collector. @p row holds the fixed width of each merge output column;
/// it is empty when the merge's own schema is unknown, in which case the candidate is rejected on
/// its state before these sizes matter.
[[nodiscard]] downstream_shape classify_bypass_downstream(const sirius_physical_operator& merge,
                                                          std::vector<std::uint64_t> row)
{
  using T = SiriusPhysicalOperatorType;
  downstream_shape shape;
  downstream_shape const rejected;
  if (merge.owning_delim_join() != nullptr) { return rejected; }
  // Charge the columns a FILTER or LIMIT actually copies into its output row.
  auto charge_row = [&] {
    for (auto const width : row) {
      shape.row_bytes = memory::saturating_add(shape.row_bytes, width);
    }
    shape.columns += row.size();
  };
  for (auto* cur = merge.get_parent_op(); cur != nullptr; cur = cur->get_parent_op()) {
    if (cur->type == T::RESULT_COLLECTOR) {
      shape.supported = true;
      return shape;
    }
    if (cur->children.size() != 1) { return rejected; }
    switch (cur->type) {
      case T::FILTER: {
        auto const* filter = dynamic_cast<const sirius_physical_filter*>(cur);
        if (filter == nullptr) { return rejected; }
        if (auto const* indices =
              std::get_if<std::vector<cudf::size_type>>(&filter->output_columns)) {
          // Filters can gather a subset or reorder columns. Later references index this output,
          // not the original merge row; preserve physical widths rather than logical types.
          if (indices->empty()) { return rejected; }
          std::vector<std::uint64_t> next;
          next.reserve(indices->size());
          for (auto const index : *indices) {
            if (index < 0 || static_cast<std::size_t>(index) >= row.size()) { return rejected; }
            next.push_back(row[static_cast<std::size_t>(index)]);
          }
          row = std::move(next);
        }
        charge_row();
        break;
      }
      case T::LIMIT:
      case T::STREAMING_LIMIT: charge_row(); break;
      case T::PROJECTION: {
        // Pure references are zero-copy views of the input. Every evaluated expression allocates
        // a new column of its result type, so only fixed-width results can be sized; any other
        // result (a string from CASE or a function, say) has no bound here and is rejected.
        // Temporaries inside a nested expression are not sized and fall under the headroom.
        auto const* projection = dynamic_cast<const sirius_physical_projection*>(cur);
        if (projection == nullptr || projection->select_list.size() != cur->types.size()) {
          return rejected;
        }
        std::vector<std::uint64_t> next;
        next.reserve(projection->select_list.size());
        for (std::size_t i = 0; i < projection->select_list.size(); ++i) {
          auto const& expr = *projection->select_list[i];
          if (expr.holds<sirius::ast::reference>()) {
            auto const index = expr.get<sirius::ast::reference>().column_index;
            if (index >= row.size()) { return rejected; }
            next.push_back(row[index]);
            continue;
          }
          auto const dtype = try_get_cudf_type(cur->types[i]);
          if (!dtype.has_value() || !cudf::is_fixed_width(*dtype)) { return rejected; }
          auto const width = static_cast<std::uint64_t>(cudf::size_of(*dtype));
          shape.row_bytes  = memory::saturating_add(shape.row_bytes, width);
          shape.columns += 1;
          next.push_back(width);
        }
        row = std::move(next);
        break;
      }
      default: return rejected;
    }
  }
  return rejected;
}

}  // namespace

bool sirius_physical_grouped_aggregate_merge::bypass_supported_aggregates() const
{
  // AVG and COUNT(DISTINCT) need post-merge projection (a cast/divide, or list element counting)
  // whose allocations are outside the model.
  if (has_avg || has_count_distinct) { return false; }
  if (cudf_aggregates.empty()) { return false; }
  for (auto kind : cudf_aggregates) {
    switch (kind) {
      case cudf::aggregation::Kind::SUM:
      case cudf::aggregation::Kind::MIN:
      case cudf::aggregation::Kind::MAX:
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: break;
      // COLLECT_SET arrives as a LIST and re-merges with MERGE_SETS; not modelled.
      default: return false;
    }
  }
  return true;
}

int sirius_physical_grouped_aggregate_merge::apply_memory_aware_bypass(
  const partition_sizing_input& in, int natural)
{
  // With the prototype off the PARTITION passes no metadata source, so there is nothing to do and
  // the automatic count is returned without any extra work on this path.
  if (!in.bypass_metadata_source) { return natural; }
  // Only an automatic count above 1 on a single admitted GPU can be overturned. Decide those
  // cases before walking the partition's batches for metadata; decide() would reject them anyway.
  if (natural <= 1 || _num_gpus > 1) { return natural; }
  auto const collected = in.bypass_metadata_source();
  if (!collected.has_value()) { return natural; }
  auto const& meta = *collected;

  group_by_bypass::candidate_input candidate;
  candidate.auto_num_partitions = natural;
  candidate.num_admitted_gpus   = _num_gpus;
  candidate.upstream_complete   = meta.upstream_complete;
  // A single admitted GPU must also mean the input really is on one device.
  candidate.single_gpu_resident = meta.single_gpu_resident;
  if (auto pipeline = get_pipeline()) {
    candidate.headroom_fraction = pipeline->get_operator_params().group_by_bypass_headroom_fraction;
  }
  candidate.total_rows                   = meta.total_rows;
  candidate.admissible_additional_budget = meta.admissible_additional_budget;

  // Split the observed physical columns at the grouping-key boundary — the same boundary
  // merge_grouped_aggregate itself uses — and check each side against the whitelist.
  auto const num_group_cols = group_idx.size();
  std::vector<std::uint64_t> output_widths;
  if (meta.columns.has_value() && meta.columns->size() == num_group_cols + cudf_aggregates.size()) {
    auto const& cols        = *meta.columns;
    bool types_ok           = bypass_supported_aggregates();
    std::uint64_t key_width = 0;
    std::uint64_t agg_width = 0;
    std::size_t null_keys   = 0;
    std::size_t null_aggs   = 0;
    for (std::size_t c = 0; c < cols.size(); ++c) {
      auto const& col = cols[c];
      if (col.fixed_width_bytes == 0 || !bypass_supported_column_type(col.type_id)) {
        types_ok = false;
        break;
      }
      if (c < num_group_cols) {
        key_width += col.fixed_width_bytes;
        null_keys += col.nullable ? 1 : 0;
      } else {
        agg_width += col.fixed_width_bytes;
        null_aggs += col.nullable ? 1 : 0;
      }
    }
    candidate.supported_state = types_ok;
    if (types_ok) {
      candidate.key_width_bytes      = key_width;
      candidate.agg_width_bytes      = agg_width;
      candidate.key_columns          = num_group_cols;
      candidate.agg_columns          = cols.size() - num_group_cols;
      candidate.nullable_key_columns = null_keys;
      candidate.nullable_agg_columns = null_aggs;
      // The whitelisted states merge into columns of the same physical type, so the merge output
      // row has the partial input's widths.
      output_widths.reserve(cols.size());
      for (auto const& col : cols) {
        output_widths.push_back(col.fixed_width_bytes);
      }
    }
  } else {
    // Either the schema could not be read, or it does not match this merge's own column layout.
    // Both are "unknown", not "zero".
    candidate.supported_state = false;
  }

  auto const shape               = classify_bypass_downstream(*this, std::move(output_widths));
  candidate.supported_downstream = shape.supported;
  candidate.downstream_row_bytes = shape.row_bytes;
  candidate.downstream_columns   = shape.columns;

  auto const decision = group_by_bypass::decide(candidate);
  bool const selected = decision.reason == group_by_bypass::decision_reason::bypass_selected;
  // Feed the same model into the existing cold-start hook. Rejected candidates and the
  // pre-existing automatic P=1 path keep the default estimate.
  _bypass_peak_memory_estimate.store(
    selected ? static_cast<std::size_t>(decision.model.additional_needed) : 0,
    std::memory_order_release);

  // Selections change the plan and are logged at INFO; rejections are routine and stay at DEBUG.
  auto const budget = candidate.admissible_additional_budget.has_value()
                        ? std::to_string(*candidate.admissible_additional_budget)
                        : std::string{"unknown"};
  if (selected) {
    SIRIUS_LOG_INFO(
      "group_by_bypass: operator_id={} device={} reason={} auto_p={} chosen_p={} "
      "additional_needed={} required={} budget={}",
      get_operator_id(),
      meta.target_device_id,
      group_by_bypass::reason_name(decision.reason),
      natural,
      decision.num_partitions,
      decision.model.additional_needed,
      decision.model.required_bytes,
      budget);
  } else {
    SIRIUS_LOG_DEBUG(
      "group_by_bypass: operator_id={} device={} reason={} auto_p={} chosen_p={} "
      "model_evaluated={} additional_needed={} required={} budget={}",
      get_operator_id(),
      meta.target_device_id,
      group_by_bypass::reason_name(decision.reason),
      natural,
      decision.num_partitions,
      decision.model_evaluated,
      decision.model.additional_needed,
      decision.model.required_bytes,
      budget);
  }

  return decision.num_partitions;
}

std::size_t sirius_physical_grouped_aggregate_merge::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  if (stats.bytes == 0) { return 0; }
  return std::max(sirius_physical_operator::no_history_peak_memory_estimate(stats),
                  _bypass_peak_memory_estimate.load(std::memory_order_acquire));
}

partition_strategy sirius_physical_grouped_aggregate_merge::get_partition_strategy(
  const partition_sizing_input& in)
{
  // Preserve automatic sizing unless the opt-in policy accepts the whole merge.
  int const natural = natural_num_partitions(in.total_bytes, _hash_partition_bytes, _num_gpus);
  // The prototype may turn AUTO > 1 into 1. It returns `natural` unchanged in every other
  // case, including when it is disabled.
  int const chosen = apply_memory_aware_bypass(in, natural);
  // Pre-size this merge's single input repository so every partition slot exists before batches
  // arrive (grouping is never broadcast / build-probe). Guarded on strictly-greater to respect the
  // repository's set_num_partitions contract.
  if (chosen > 1) {
    std::lock_guard<std::mutex> lg(lock);
    if (!ports.empty()) {
      auto& repo = ports.begin()->second->repo;
      if (repo != nullptr && static_cast<std::size_t>(chosen) > repo->num_partitions()) {
        repo->set_num_partitions(static_cast<std::size_t>(chosen));
      }
    }
  }
  return {chosen, /*broadcast=*/false, /*build_probe=*/false};
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
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_grouped_aggregate_merge::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (input_batches.size() == 0) {
    throw std::runtime_error(
      "We expect at least one input batch for grouped aggregate merge operator");
  }

  // Fast path: single batch with no post-processing needed
  if (input_batches.size() == 1 && !has_avg && !has_count_distinct) {
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
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
}  // namespace op
}  // namespace sirius
