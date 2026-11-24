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

#include "data/data_batch_utils.hpp"
#include "duckdb/common/assert.hpp"
#include "log/logging.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "op/aggregate/wide_integer_sum.hpp"
#include "sirius_config.hpp"
#include "telemetry/nvtx.hpp"

#include <algorithm>
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
  duckdb::vector<duckdb::unsafe_vector<duckdb::ProjectionIndex>> grouping_functions_p,
  std::size_t estimated_cardinality,
  duckdb::TupleDataValidityType /*group_validity*/,
  duckdb::TupleDataValidityType /*distinct_validity*/,
  std::vector<std::optional<std::uint64_t>> aggregate_input_max_abs)
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
  local_types                       = std::move(cudf_defs.local_types);
  has_avg                           = cudf_defs.has_avg;
  has_count_distinct                = cudf_defs.has_count_distinct;
  // `create_plan(LogicalAggregate&)` refuses FIRST over several grouping sets or beside GROUPING().
  D_ASSERT(!(has_first() && has_grouping_sets()));
  // Grouping-set expansion inserts the set id and GROUPING() values before the partial
  // aggregate carriers. Keep their types from the planner's final schema; AVG may expand
  // the aggregate suffix, and STDDEV_SAMP / COUNT(DISTINCT) replace its scalar carriers.
  if (has_grouping_sets()) {
    auto const keys_end   = static_cast<std::ptrdiff_t>(group_idx.size());
    auto const groups_end = static_cast<std::ptrdiff_t>(num_output_group_columns());
    local_types.insert(local_types.begin() + keys_end,
                       this->types.begin() + keys_end,
                       this->types.begin() + groups_end);
  }
  // A bound belongs to the aggregate's SUM slot; AVG's COUNT_VALID slot (sum_idx + 1) needs none.
  cudf_aggregate_input_max_abs.assign(cudf_aggregates.size(), std::nullopt);
  auto const bounded_idx = [](aggregate_slot const& slot) -> std::optional<std::size_t> {
    if (auto const* plain = std::get_if<plain_slot>(&slot)) { return plain->partial_idx; }
    if (auto const* avg = std::get_if<avg_slot>(&slot)) { return avg->sum_idx; }
    if (auto const* stddev = std::get_if<stddev_samp_slot>(&slot)) { return stddev->partial_idx; }
    return std::nullopt;
  };
  for (size_t i = 0; i < aggregate_slots.size() && i < aggregate_input_max_abs.size(); ++i) {
    auto const idx = bounded_idx(aggregate_slots[i]);
    if (aggregate_input_max_abs[i] && idx) {
      cudf_aggregate_input_max_abs[*idx] = aggregate_input_max_abs[i];
    }
  }
  cudf_aggregate_is_wide_integer_sum =
    wide_integer_sum_mask(aggregate_slots, cudf_aggregates.size());
}

duckdb::vector<sirius::logical_type>
sirius_physical_grouped_aggregate::get_count_distinct_local_output_types() const
{
  auto const aggregate_offset = num_output_group_columns();
  if (!has_count_distinct || has_avg || has_first() ||
      types.size() != aggregate_offset + aggregate_slots.size()) {
    throw std::runtime_error(
      "COUNT(DISTINCT) local schema requires a non-AVG, non-FIRST one-slot-per-aggregate layout");
  }

  auto local_types = types;
  for (size_t slot_idx = 0; slot_idx < aggregate_slots.size(); ++slot_idx) {
    if (std::holds_alternative<count_distinct_slot>(aggregate_slots[slot_idx])) {
      local_types[aggregate_offset + slot_idx] = sirius::logical_type::make(sirius::type_id::LIST);
    }
  }
  return local_types;
}

bool sirius_physical_grouped_aggregate::has_stddev_samp() const
{
  return has_slot<stddev_samp_slot>(aggregate_slots);
}

std::vector<std::optional<cudf::data_type>>
sirius_physical_grouped_aggregate::wide_integer_sum_partial_types() const
{
  auto const layout = partial_layout();
  if (types.size() != layout.num_columns()) { return {}; }
  auto const alternate_partial = [](aggregate_slot const& slot) -> std::optional<std::size_t> {
    if (auto const* plain = std::get_if<plain_slot>(&slot);
        plain && (plain->is_wide_integer_sum || plain->returns_hugeint)) {
      return plain->partial_idx;
    }
    if (auto const* avg = std::get_if<avg_slot>(&slot); avg && avg->is_wide_integer_sum) {
      return avg->sum_idx;
    }
    return std::nullopt;
  };
  std::vector<std::optional<cudf::data_type>> alternates(types.size());
  bool has_alternate = false;
  for (auto const& slot : aggregate_slots) {
    auto const partial = alternate_partial(slot);
    if (!partial) { continue; }
    auto const column  = layout.partial_column(*partial);
    alternates[column] = sirius::require_cudf_type(types[column]) == wide_integer_sum_type()
                           ? cudf::data_type{cudf::type_id::INT64}
                           : wide_integer_sum_type();
    has_alternate      = true;
  }
  if (!has_alternate) { return {}; }
  return alternates;
}

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_grouped_aggregate::execute"};
  auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);
  if (_table_aggregate) { return execute_with_table(input, stream); }
  const auto& input_batches = input.get_read_only_batches();
  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  auto const carried = carried_inputs(aggregate_slots);
  for (auto const& input_batch : input_batches) {
    auto* space = input_batch.get_memory_space();
    if (!space) { continue; }
    if (has_grouping_sets()) {
      results.push_back(gpu_aggregate_impl::local_grouping_sets_aggregate(
        input_batch,
        group_idx,
        cudf_aggregates,
        cudf_aggregate_idx,
        cudf_aggregate_struct_col_indices,
        grouping_set_keys,
        grouping_functions,
        stream,
        *space,
        batch_telemetry(),
        cudf_aggregate_input_max_abs,
        cudf_aggregate_is_wide_integer_sum,
        operator_options().enable_low_cardinality_groupby));
      continue;
    }
    auto const last_groups     = _last_batch_groups.load(std::memory_order_relaxed);
    bool const split_wide_sums = last_groups <= split_wide_sum_max_groups;
    cudf::size_type num_groups = 0;
    auto result                = gpu_aggregate_impl::local_grouped_aggregate(
      input_batch,
      group_idx,
      cudf_aggregates,
      cudf_aggregate_idx,
      cudf_aggregate_struct_col_indices,
      carried,
      stream,
      *space,
      batch_telemetry(),
      cudf_aggregate_input_max_abs,
      cudf_aggregate_is_wide_integer_sum,
      split_wide_sums,
      &num_groups,
      operator_options().enable_low_cardinality_groupby);
    _last_batch_groups.store(num_groups, std::memory_order_relaxed);
    results.push_back(std::move(result));
  }
  return std::make_unique<pipelineable_operator_data>(results);
}

bool sirius_physical_grouped_aggregate::enable_table_aggregate(
  table_aggregate_options const& options, std::string* why_not)
{
  auto const refuse = [&](std::string reason) {
    if (why_not != nullptr) { *why_not = std::move(reason); }
    return false;
  };
  if (has_grouping_sets()) { return refuse("grouping sets"); }
  if (group_idx.size() != 1) { return refuse("not a single key column"); }
  if (has_count_distinct || has_stddev_samp() || has_first()) {
    return refuse("COUNT(DISTINCT), STDDEV_SAMP or FIRST");
  }
  // A narrowed key or aggregate carrier has another runtime type than the declared one.
  if (has_physical_overrides()) { return refuse("narrow carriers on the aggregate"); }
  if (children.size() != 1) { return refuse("not exactly one input"); }
  for (auto const& slot : aggregate_slots) {
    if (auto const* plain = std::get_if<plain_slot>(&slot)) {
      // A HUGEINT input is summed in DECIMAL128 with a range check at the merge.
      if (plain->sums_hugeint_input) { return refuse("SUM over HUGEINT"); }
    } else if (auto const* avg = std::get_if<avg_slot>(&slot)) {
      if (avg->sums_hugeint_input) { return refuse("AVG over HUGEINT"); }
    } else {
      return refuse("an aggregate that is not COUNT, SUM, MIN, MAX or AVG");
    }
  }
  for (auto const& struct_columns : cudf_aggregate_struct_col_indices) {
    if (!struct_columns.empty()) { return refuse("an aggregate over several columns"); }
  }

  auto const& child      = *children[0];
  auto const& child_cols = child.get_types();
  auto const input_type  = [&](int column) -> std::optional<cudf::data_type> {
    if (column < 0 || static_cast<std::size_t>(column) >= child_cols.size()) {
      return std::nullopt;
    }
    auto const idx = static_cast<std::size_t>(column);
    if (child.has_physical_overrides()) {
      auto const& physical = child.get_physical_types();
      return idx < physical.size() ? std::optional{physical[idx]} : std::nullopt;
    }
    try {
      return sirius::require_cudf_type(child_cols[idx]);
    } catch (std::exception const&) {
      return std::nullopt;
    }
  };

  table_aggregate_plan plan;
  plan.key_column     = group_idx[0];
  auto const key_type = input_type(group_idx[0]);
  if (!key_type) { return refuse("key type has no cuDF carrier"); }
  plan.key_type = *key_type;
  for (std::size_t i = 0; i < cudf_aggregates.size(); ++i) {
    auto const type = input_type(cudf_aggregate_idx[i]);
    if (!type) { return refuse("aggregate input type has no cuDF carrier"); }
    plan.aggregates.push_back(cudf_aggregates[i]);
    plan.aggregate_columns.push_back(cudf_aggregate_idx[i]);
    plan.aggregate_input_types.push_back(*type);
  }
  plan.input_max_abs       = cudf_aggregate_input_max_abs;
  plan.is_wide_integer_sum = cudf_aggregate_is_wide_integer_sum;
  if (child.estimated_cardinality > 0) { plan.rows_hint = child.estimated_cardinality; }

  _table_aggregate = table_aggregate_state::try_create(std::move(plan), options, why_not);
  return _table_aggregate != nullptr;
}

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate::execute_with_table(
  const pipelineable_operator_data& input, ::cuda::stream_ref stream)
{
  const auto& input_batches = input.get_read_only_batches();
  // A replayed task passes the same input object, so these ids name the same rows on every
  // attempt (see pipelineable_operator_data::original_batch_ids). The table keys its
  // exactly-once records on them.
  auto const batch_ids = input.original_batch_ids();
  // Without one id per batch there is nothing to key a replay on. Where late materialization
  // restores a deferral at this operator's port, the rows a table would flush later would
  // bypass that restore. Both keep the cuDF path.
  bool const use_table = batch_ids.size() == input_batches.size() && port_directive().empty();
  if (!use_table && _table_aggregate->enabled()) {
    _table_aggregate->disable("input batches carry no stable ids, or a deferral ends here");
  }
  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  auto const carried = carried_inputs(aggregate_slots);
  for (std::size_t i = 0; i < input_batches.size(); ++i) {
    auto const& input_batch = input_batches[i];
    auto* space             = input_batch.get_memory_space();
    if (!space) { continue; }
    // The cuDF local aggregation, exactly as execute() runs it without a table.
    table_aggregate_state::legacy_aggregate const legacy = [&](cudf::table_view const& rows) {
      auto const last_groups     = _last_batch_groups.load(std::memory_order_relaxed);
      bool const split_wide_sums = last_groups <= split_wide_sum_max_groups;
      cudf::size_type num_groups = 0;
      auto result                = gpu_aggregate_impl::local_grouped_aggregate(
        rows,
        group_idx,
        cudf_aggregates,
        cudf_aggregate_idx,
        cudf_aggregate_struct_col_indices,
        carried,
        stream,
        *space,
        batch_telemetry(),
        cudf_aggregate_input_max_abs,
        cudf_aggregate_is_wide_integer_sum,
        split_wide_sums,
        &num_groups,
        operator_options().enable_low_cardinality_groupby);
      // A probe of the schema on no rows says nothing about the input's groups.
      if (rows.num_rows() > 0) { _last_batch_groups.store(num_groups, std::memory_order_relaxed); }
      return result;
    };
    auto const rows = sirius::get_cudf_table_view(input_batch);
    if (use_table) {
      _table_aggregate->aggregate_batch(
        rows, batch_ids[i], stream, *space, batch_telemetry(), legacy, results);
    } else {
      results.push_back(legacy(rows));
    }
  }
  if (use_table) { _table_aggregate->complete_batches(batch_ids); }
  return std::make_unique<pipelineable_operator_data>(results);
}

void sirius_physical_grouped_aggregate::on_finalize_operator()
{
  // This pipeline has finished: no task folds any more. Tables that hold rows wait for the
  // consumer's end-of-input flush tasks; the others are freed here, each on its own GPU.
  if (_table_aggregate) { _table_aggregate->release_idle_tables(); }
}
}  // namespace op
}  // namespace sirius
