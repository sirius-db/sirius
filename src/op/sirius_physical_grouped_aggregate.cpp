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

#include "op/sirius_physical_grouped_aggregate.hpp"

#include "config.hpp"
#include "data/data_batch_utils.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "telemetry/nvtx.hpp"

namespace sirius {
namespace op {

sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  std::size_t estimated_cardinality)
  : sirius_physical_grouped_aggregate(std::move(types),
                                      std::move(expressions),
                                      std::move(groups_p),
                                      {},
                                      {},
                                      estimated_cardinality,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)
{
}

// The parameters for `SELECT a, b, sum(v), GROUPING(a) ... GROUP BY ROLLUP(a, b)`:
// - expressions: the aggregates, each over bound_ref expressions to its input columns.
//   Here `sum(v)`
// - groups_p: the group by keys, each a bound_ref expression. Here `a` and `b`
// - grouping_sets_p: the grouping sets, each a set of positions in groups_p.
//   Here `{0, 1}`, `{0}` and `{}`. A plain GROUP BY has one set with all keys
// - grouping_functions_p: the GROUPING() functions, each the list of its arguments as
//   positions in groups_p. Here `[0]`. Empty when the query has no GROUPING()
sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<std::size_t>> grouping_functions_p,
  std::size_t estimated_cardinality,
  duckdb::TupleDataValidityType /*group_validity*/,
  duckdb::TupleDataValidityType /*distinct_validity*/)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  for (auto const& set : grouping_sets) {
    grouping_set_keys.emplace_back(set.begin(), set.end());
  }
  for (auto const& function : grouping_functions_p) {
    grouping_functions.emplace_back(function.begin(), function.end());
  }
  auto cudf_defs                    = convert_duckdb_aggregates_to_cudf(groups_p, expressions);
  group_idx                         = std::move(cudf_defs.group_idx);
  cudf_aggregates                   = std::move(cudf_defs.cudf_aggregates);
  cudf_aggregate_idx                = std::move(cudf_defs.cudf_aggregate_idx);
  cudf_aggregate_struct_col_indices = std::move(cudf_defs.cudf_aggregate_struct_col_indices);
  aggregate_slots                   = std::move(cudf_defs.aggregate_slots);
  has_avg                           = cudf_defs.has_avg;
  has_count_distinct                = cudf_defs.has_count_distinct;
}

duckdb::vector<sirius::logical_type>
sirius_physical_grouped_aggregate::get_count_distinct_local_output_types() const
{
  auto const aggregate_offset = num_output_group_columns();
  if (!has_count_distinct || has_avg || types.size() != aggregate_offset + aggregate_slots.size()) {
    throw std::runtime_error(
      "COUNT(DISTINCT) local schema requires a non-AVG one-slot-per-aggregate layout");
  }

  auto local_types = types;
  for (size_t slot_idx = 0; slot_idx < aggregate_slots.size(); ++slot_idx) {
    if (aggregate_slots[slot_idx].is_count_distinct) {
      local_types[aggregate_offset + slot_idx] = sirius::logical_type::make(sirius::type_id::LIST);
    }
  }
  return local_types;
}

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_grouped_aggregate::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  for (auto const& input_batch : input_batches) {
    auto* space = input_batch.get_memory_space();
    if (!space) { continue; }
    if (has_grouping_sets()) {
      results.push_back(
        gpu_aggregate_impl::local_grouping_sets_aggregate(input_batch,
                                                          group_idx,
                                                          cudf_aggregates,
                                                          cudf_aggregate_idx,
                                                          cudf_aggregate_struct_col_indices,
                                                          grouping_set_keys,
                                                          grouping_functions,
                                                          stream,
                                                          *space,
                                                          batch_telemetry()));
      continue;
    }
    auto result = gpu_aggregate_impl::local_grouped_aggregate(input_batch,
                                                              group_idx,
                                                              cudf_aggregates,
                                                              cudf_aggregate_idx,
                                                              cudf_aggregate_struct_col_indices,
                                                              stream,
                                                              *space,
                                                              batch_telemetry());
    results.push_back(std::move(result));
  }
  return std::make_unique<pipelineable_operator_data>(results);
}
}  // namespace op
}  // namespace sirius
