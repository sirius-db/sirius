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
#include "duckdb/common/exception.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "telemetry/nvtx.hpp"

#include <string>
#include <variant>

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

// expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression
// to a column groups_p is the list of group by columns. Each group by column is a bound_ref
// expression to a column grouping_sets_p is the list of grouping set. Each grouping set is a set of
// indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets
// and for every grouping set there is one radix_table grouping_functions_p is a list of indexes to
// the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the
// grouping set and the second level is the indexes to the groupby expression for that set.
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
  auto cudf_defs                    = convert_duckdb_aggregates_to_cudf(groups_p, expressions);
  group_idx                         = std::move(cudf_defs.group_idx);
  cudf_aggregates                   = std::move(cudf_defs.cudf_aggregates);
  cudf_aggregate_idx                = std::move(cudf_defs.cudf_aggregate_idx);
  cudf_aggregate_struct_col_indices = std::move(cudf_defs.cudf_aggregate_struct_col_indices);
  aggregate_slots                   = std::move(cudf_defs.aggregate_slots);
  has_avg                           = cudf_defs.has_avg;
  has_count_distinct                = cudf_defs.has_count_distinct;
  has_first                         = cudf_defs.has_first;
  // Grouping functions add output columns no slot covers, which fails the cover. Several grouping
  // sets add none, so they are refused by name: cudf::distinct would dedup on every key at once and
  // drop the other sets' rows.
  if (has_first && (grouping_sets.size() > 1 || !is_one_row_per_key())) {
    throw duckdb::NotImplementedException(
      "grouped FIRST is only supported as one row per key over one grouping set: the group "
      "keys and FIRST inputs must cover the operator's " +
      std::to_string(types.size()) + " output columns exactly once (falling back to CPU)");
  }
}

bool sirius_physical_grouped_aggregate::is_one_row_per_key() const
{
  return one_row_per_key_select(group_idx, aggregate_slots, types.size()).has_value();
}

duckdb::vector<sirius::logical_type>
sirius_physical_grouped_aggregate::get_count_distinct_local_output_types() const
{
  auto const aggregate_offset = group_idx.size();
  if (!has_count_distinct || has_avg || types.size() != aggregate_offset + aggregate_slots.size()) {
    throw std::runtime_error(
      "COUNT(DISTINCT) local schema requires a non-AVG one-slot-per-aggregate layout");
  }

  auto local_types = types;
  for (size_t slot_idx = 0; slot_idx < aggregate_slots.size(); ++slot_idx) {
    if (std::holds_alternative<count_distinct_slot>(aggregate_slots[slot_idx])) {
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
  auto const one_row_select = one_row_per_key_select(group_idx, aggregate_slots, types.size());
  for (auto const& input_batch : input_batches) {
    auto* space = input_batch.get_memory_space();
    if (!space) { continue; }
    auto result = one_row_select
                    ? gpu_aggregate_impl::local_one_row_per_key(
                        input_batch, group_idx, *one_row_select, stream, *space, batch_telemetry())
                    : gpu_aggregate_impl::local_grouped_aggregate(input_batch,
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
