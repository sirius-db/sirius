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
#include <cudf/lists/count_elements.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <mutex>
#include <optional>
#include <type_traits>
#include <variant>

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

namespace {

struct signed_key_range {
  int64_t low;
  int64_t high;
};

struct unsigned_key_range {
  uint64_t low;
  uint64_t high;
};

using host_key_range = std::variant<signed_key_range, unsigned_key_range>;

struct host_key_range_extractor {
  cudf::scalar const& minimum;
  cudf::scalar const& maximum;
  rmm::cuda_stream_view stream;

  template <typename T>
  std::optional<host_key_range> operator()() const
  {
    if constexpr (cudf::is_integral<T>()) {
      using scalar_type = cudf::scalar_type_t<T>;
      auto const low    = static_cast<scalar_type const&>(minimum).value(stream);
      auto const high   = static_cast<scalar_type const&>(maximum).value(stream);
      if constexpr (std::is_signed_v<T>) {
        return signed_key_range{static_cast<int64_t>(low), static_cast<int64_t>(high)};
      } else {
        return unsigned_key_range{static_cast<uint64_t>(low), static_cast<uint64_t>(high)};
      }
    } else if constexpr (cudf::is_timestamp<T>()) {
      using scalar_type = cudf::scalar_type_t<T>;
      auto const low    = static_cast<scalar_type const&>(minimum).value(stream);
      auto const high   = static_cast<scalar_type const&>(maximum).value(stream);
      return signed_key_range{static_cast<int64_t>(low.time_since_epoch().count()),
                              static_cast<int64_t>(high.time_since_epoch().count())};
    } else {
      return std::nullopt;
    }
  }
};

template <typename Range>
bool ranges_are_strictly_disjoint(const std::vector<host_key_range>& ranges)
{
  std::vector<Range> typed_ranges;
  typed_ranges.reserve(ranges.size());
  for (auto const& range : ranges) {
    auto const* typed_range = std::get_if<Range>(&range);
    if (typed_range == nullptr) { return false; }
    typed_ranges.push_back(*typed_range);
  }
  std::sort(typed_ranges.begin(), typed_ranges.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.low < rhs.low || (lhs.low == rhs.low && lhs.high < rhs.high);
  });
  for (std::size_t index = 1; index < typed_ranges.size(); ++index) {
    if (!(typed_ranges[index - 1].high < typed_ranges[index].low)) { return false; }
  }
  return true;
}

bool leading_key_ranges_are_strictly_disjoint(
  const std::vector<cucascade::read_only_data_batch>& batches, rmm::cuda_stream_view stream)
{
  std::optional<cudf::data_type> key_type;
  std::vector<host_key_range> ranges;
  ranges.reserve(batches.size());

  // The read-only accessors in batches own every table view used by this proof.
  for (auto const& batch : batches) {
    auto* const data  = batch.get_data();
    auto* const space = batch.get_memory_space();
    if (data == nullptr || space == nullptr || space->get_tier() != cucascade::memory::Tier::GPU) {
      return false;
    }
    auto const table = get_cudf_table_view(batch);
    if (table.num_columns() == 0 || table.num_rows() == 0) { return false; }
    auto const key = table.column(0);
    if (key.has_nulls() || (!cudf::is_integral(key.type()) && !cudf::is_timestamp(key.type()))) {
      return false;
    }
    if (key_type.has_value() && key.type() != *key_type) { return false; }
    key_type = key.type();

    auto [minimum, maximum] = cudf::minmax(key, stream, space->get_default_allocator());
    auto range =
      cudf::type_dispatcher(key.type(), host_key_range_extractor{*minimum, *maximum, stream});
    if (!range.has_value()) { return false; }
    ranges.push_back(*range);
  }

  if (ranges.empty()) { return false; }
  return std::holds_alternative<signed_key_range>(ranges.front())
           ? ranges_are_strictly_disjoint<signed_key_range>(ranges)
           : ranges_are_strictly_disjoint<unsigned_key_range>(ranges);
}

bool task_bytes_fit(const std::vector<cucascade::read_only_data_batch>& batches,
                    uint64_t byte_limit)
{
  uint64_t total_bytes = 0;
  for (auto const& batch : batches) {
    auto const* data = batch.get_data();
    if (data == nullptr) { return false; }
    auto const batch_bytes = static_cast<uint64_t>(data->get_size_in_bytes());
    if (batch_bytes > byte_limit - total_bytes) { return false; }
    total_bytes += batch_bytes;
  }
  return true;
}

}  // namespace

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
  grouping_sets         = grouped_aggregate->grouping_sets;
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
  auto& input        = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto input_batches = input.get_read_only_batches();
  if (input_batches.size() == 0) {
    throw std::runtime_error(
      "We expect at least one input batch for grouped aggregate merge operator");
  }

  auto const* input_port = ports.size() == 1 ? ports.begin()->second : nullptr;
  bool const one_upstream_partition =
    input_port != nullptr && input_port->repo != nullptr && input_port->repo->num_partitions() == 1;
  bool const passthrough_candidate =
    _enable_disjoint_groupby_passthrough && input_batches.size() > 1 && !has_avg &&
    !has_count_distinct && grouping_sets.size() <= 1 && !group_idx.empty() &&
    one_upstream_partition && task_bytes_fit(input_batches, _hash_partition_bytes);
  if (passthrough_candidate && leading_key_ranges_are_strictly_disjoint(input_batches, stream)) {
    return std::make_unique<pipelineable_operator_data>(std::move(input_batches));
  }

  // Fast path: single batch with no post-processing needed
  if (input_batches.size() == 1 && !has_avg && !has_count_distinct) {
    return std::make_unique<pipelineable_operator_data>(std::move(input_batches));
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
