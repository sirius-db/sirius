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

#include <expression/ast/node.hpp>
#include <op/sirius_physical_column_data_scan.hpp>
#include <op/sirius_physical_cte.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <op/sirius_physical_projection.hpp>
#include <planner/dynamic_filter/dynamic_filter_target_discovery.hpp>

#include <algorithm>
#include <cassert>
#include <string_view>
#include <variant>

namespace sirius::planner {

namespace {

// ANTI preserves LHS values; RIGHT/OUTER synthesize only NULLs rejected by producer equality.
bool probe_block_descent_admissible(duckdb::JoinType join_type) noexcept
{
  switch (join_type) {
    case duckdb::JoinType::INNER:
    case duckdb::JoinType::LEFT:
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::RIGHT:
    case duckdb::JoinType::OUTER:
    case duckdb::JoinType::MARK: return true;
    case duckdb::JoinType::SINGLE:
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::RIGHT_ANTI:
    case duckdb::JoinType::INVALID: return false;
  }
  return false;  // unreachable
}

// LEFT/OUTER synthesize only NULLs rejected by producer equality.
bool build_block_descent_admissible(duckdb::JoinType join_type) noexcept
{
  switch (join_type) {
    case duckdb::JoinType::INNER:
    case duckdb::JoinType::RIGHT:
    case duckdb::JoinType::LEFT:
    case duckdb::JoinType::OUTER: return true;
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::MARK:
    case duckdb::JoinType::SINGLE:
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::RIGHT_ANTI:
    case duckdb::JoinType::INVALID: return false;
  }
  return false;  // unreachable
}

std::vector<descent_step> as_steps(std::optional<descent_step> step)
{
  if (!step.has_value()) { return {}; }
  return {*step};
}

}  // namespace

std::optional<std::size_t> projection_reference_input(sirius::ast::node const& expression)
{
  if (!expression.is_reference()) { return std::nullopt; }
  return static_cast<std::size_t>(expression.as_reference().column_index);
}

std::optional<std::size_t> group_by_key_input(std::vector<int> const& group_idx,
                                              std::size_t grouping_set_count,
                                              std::size_t output_ordinal)
{
  if (grouping_set_count > 1) { return std::nullopt; }
  if (output_ordinal >= group_idx.size()) { return std::nullopt; }
  return static_cast<std::size_t>(group_idx[output_ordinal]);
}

std::optional<descent_step> join_block_descent(
  duckdb::JoinType join_type,
  std::vector<cudf::size_type> const& probe_block_output_columns,
  std::vector<cudf::size_type> const& build_block_output_columns,
  std::size_t output_ordinal,
  descent_policy policy)
{
  auto const probe_block_size = probe_block_output_columns.size();
  if (output_ordinal < probe_block_size) {
    if (policy.preserve_nulls &&
        (join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::OUTER)) {
      return std::nullopt;
    }
    if (!probe_block_descent_admissible(join_type)) { return std::nullopt; }
    return descent_step{
      .child_index   = 0,
      .child_ordinal = static_cast<std::size_t>(probe_block_output_columns[output_ordinal])};
  }
  if (!policy.descend_build_blocks) { return std::nullopt; }
  if (policy.preserve_nulls &&
      (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::OUTER)) {
    return std::nullopt;
  }
  if (!build_block_descent_admissible(join_type)) { return std::nullopt; }
  auto const build_ordinal = output_ordinal - probe_block_size;
  if (build_ordinal >= build_block_output_columns.size()) { return std::nullopt; }
  return descent_step{
    .child_index   = 1,
    .child_ordinal = static_cast<std::size_t>(build_block_output_columns[build_ordinal])};
}

std::vector<descent_step> descent_steps(sirius::op::sirius_physical_operator const& node,
                                        std::size_t output_ordinal,
                                        descent_policy policy)
{
  using sirius::op::SiriusPhysicalOperatorType;
  switch (node.type) {
    case SiriusPhysicalOperatorType::PROJECTION: {
      auto const& projection = node.Cast<sirius::op::sirius_physical_projection>();
      if (output_ordinal >= projection.select_list.size()) { return {}; }
      auto const input = projection_reference_input(*projection.select_list[output_ordinal]);
      if (!input.has_value()) { return {}; }
      return {descent_step{.child_index = 0, .child_ordinal = *input}};
    }
    case SiriusPhysicalOperatorType::HASH_GROUP_BY: {
      auto const& aggregate = node.Cast<sirius::op::sirius_physical_grouped_aggregate>();
      auto const input =
        group_by_key_input(aggregate.group_idx, aggregate.grouping_sets.size(), output_ordinal);
      if (!input.has_value()) { return {}; }
      return {descent_step{.child_index = 0, .child_ordinal = *input}};
    }
    case SiriusPhysicalOperatorType::HASH_JOIN: {
      auto const& join = node.Cast<sirius::op::sirius_physical_hash_join>();
      return as_steps(join_block_descent(join.join_type,
                                         join.lhs_output_columns.col_idxs,
                                         join.rhs_output_columns.col_idxs,
                                         output_ordinal,
                                         policy));
    }
    // Independent row predicates commute without changing column values.
    case SiriusPhysicalOperatorType::FILTER: {
      auto const& filter = node.Cast<sirius::op::sirius_physical_filter>();
      if (std::holds_alternative<sirius::op::passthrough>(filter.output_columns)) {
        return {descent_step{.child_index = 0, .child_ordinal = output_ordinal}};
      }
      auto const* gather = std::get_if<std::vector<cudf::size_type>>(&filter.output_columns);
      if (gather == nullptr || output_ordinal >= gather->size()) { return {}; }
      return {descent_step{.child_index   = 0,
                           .child_ordinal = static_cast<std::size_t>((*gather)[output_ordinal])}};
    }
    case SiriusPhysicalOperatorType::UNION: {
      std::vector<descent_step> steps;
      steps.reserve(node.children.size());
      for (std::size_t child_index = 0; child_index < node.children.size(); ++child_index) {
        assert(node.children[child_index] != nullptr);
        steps.push_back(descent_step{.child_index = child_index, .child_ordinal = output_ordinal});
      }
      return steps;
    }
    case SiriusPhysicalOperatorType::DYNAMIC_FILTER:
      return {descent_step{.child_index = 0, .child_ordinal = output_ordinal}};
    // The node's output is its second child's; the first child is the shared body.
    case SiriusPhysicalOperatorType::CTE:
      if (node.children.size() != 2 || output_ordinal >= node.types.size()) { return {}; }
      return {descent_step{.child_index = 1, .child_ordinal = output_ordinal}};
    // Copies rows without changing values. Output k is input k; the trailing copy count is not an
    // output, so descent never reaches it.
    case SiriusPhysicalOperatorType::REPLICATE:
      if (output_ordinal >= node.types.size()) { return {}; }
      return {descent_step{.child_index = 0, .child_ordinal = output_ordinal}};
    case SiriusPhysicalOperatorType::INVALID:
    case SiriusPhysicalOperatorType::ORDER_BY:
    case SiriusPhysicalOperatorType::LIMIT:
    case SiriusPhysicalOperatorType::UNORDERED_STREAMING_LIMIT:
    case SiriusPhysicalOperatorType::STREAMING_LIMIT:
    case SiriusPhysicalOperatorType::LIMIT_PERCENT:
    case SiriusPhysicalOperatorType::TOP_N:
    case SiriusPhysicalOperatorType::WINDOW:
    case SiriusPhysicalOperatorType::UNNEST:
    case SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
    // Dense count join outputs have no child-ordinal mapping in target discovery.
    case SiriusPhysicalOperatorType::DENSE_COUNT_JOIN:
    case SiriusPhysicalOperatorType::PERFECT_HASH_GROUP_BY:
    case SiriusPhysicalOperatorType::PARTITIONED_AGGREGATE:
    case SiriusPhysicalOperatorType::COPY_TO_FILE:
    case SiriusPhysicalOperatorType::BATCH_COPY_TO_FILE:
    case SiriusPhysicalOperatorType::RESERVOIR_SAMPLE:
    case SiriusPhysicalOperatorType::STREAMING_SAMPLE:
    case SiriusPhysicalOperatorType::STREAMING_WINDOW:
    case SiriusPhysicalOperatorType::PIVOT:
    case SiriusPhysicalOperatorType::COPY_DATABASE:
    case SiriusPhysicalOperatorType::TABLE_SCAN:
    case SiriusPhysicalOperatorType::DUMMY_SCAN:
    case SiriusPhysicalOperatorType::COLUMN_DATA_SCAN:
    case SiriusPhysicalOperatorType::CHUNK_SCAN:
    case SiriusPhysicalOperatorType::RECURSIVE_CTE_SCAN:
    case SiriusPhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN:
    case SiriusPhysicalOperatorType::CTE_SCAN:
    case SiriusPhysicalOperatorType::DELIM_SCAN:
    case SiriusPhysicalOperatorType::EXPRESSION_SCAN:
    case SiriusPhysicalOperatorType::POSITIONAL_SCAN:
    case SiriusPhysicalOperatorType::BLOCKWISE_NL_JOIN:
    case SiriusPhysicalOperatorType::NESTED_LOOP_JOIN:
    case SiriusPhysicalOperatorType::CROSS_PRODUCT:
    case SiriusPhysicalOperatorType::PIECEWISE_MERGE_JOIN:
    case SiriusPhysicalOperatorType::IE_JOIN:
    case SiriusPhysicalOperatorType::LEFT_DELIM_JOIN:
    case SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN:
    case SiriusPhysicalOperatorType::POSITIONAL_JOIN:
    case SiriusPhysicalOperatorType::ASOF_JOIN:
    case SiriusPhysicalOperatorType::RECURSIVE_CTE:
    case SiriusPhysicalOperatorType::RECURSIVE_KEY_CTE:
    case SiriusPhysicalOperatorType::INSERT:
    case SiriusPhysicalOperatorType::BATCH_INSERT:
    case SiriusPhysicalOperatorType::DELETE_OPERATOR:
    case SiriusPhysicalOperatorType::UPDATE:
    case SiriusPhysicalOperatorType::MERGE_INTO:
    case SiriusPhysicalOperatorType::CREATE_TABLE:
    case SiriusPhysicalOperatorType::CREATE_TABLE_AS:
    case SiriusPhysicalOperatorType::BATCH_CREATE_TABLE_AS:
    case SiriusPhysicalOperatorType::CREATE_INDEX:
    case SiriusPhysicalOperatorType::ALTER:
    case SiriusPhysicalOperatorType::CREATE_SEQUENCE:
    case SiriusPhysicalOperatorType::CREATE_VIEW:
    case SiriusPhysicalOperatorType::CREATE_SCHEMA:
    case SiriusPhysicalOperatorType::CREATE_MACRO:
    case SiriusPhysicalOperatorType::DROP:
    case SiriusPhysicalOperatorType::PRAGMA:
    case SiriusPhysicalOperatorType::TRANSACTION:
    case SiriusPhysicalOperatorType::CREATE_TYPE:
    case SiriusPhysicalOperatorType::ATTACH:
    case SiriusPhysicalOperatorType::DETACH:
    case SiriusPhysicalOperatorType::EXPLAIN:
    case SiriusPhysicalOperatorType::EXPLAIN_ANALYZE:
    case SiriusPhysicalOperatorType::EMPTY_RESULT:
    case SiriusPhysicalOperatorType::EXECUTE:
    case SiriusPhysicalOperatorType::PREPARE:
    case SiriusPhysicalOperatorType::VACUUM:
    case SiriusPhysicalOperatorType::EXPORT:
    case SiriusPhysicalOperatorType::SET:
    case SiriusPhysicalOperatorType::SET_VARIABLE:
    case SiriusPhysicalOperatorType::LOAD:
    case SiriusPhysicalOperatorType::INOUT_FUNCTION:
    case SiriusPhysicalOperatorType::RESULT_COLLECTOR:
    case SiriusPhysicalOperatorType::RESET:
    case SiriusPhysicalOperatorType::EXTENSION:
    case SiriusPhysicalOperatorType::VERIFY_VECTOR:
    case SiriusPhysicalOperatorType::UPDATE_EXTENSIONS:
    case SiriusPhysicalOperatorType::CREATE_SECRET:
    // Feeders are added after discovery in `insert_gpu_pipeline_operators()`: the join child's
    // CONCAT -> PARTITION -> optional BUILD_FILTER -> existing child, and the UNION arm's
    // PASSTHROUGH_SINK -> existing child. Targets therefore end up below the wrappers either way,
    // even though each wrapper is terminal if encountered here.
    case SiriusPhysicalOperatorType::PARTITION:
    case SiriusPhysicalOperatorType::CONCAT:
    case SiriusPhysicalOperatorType::BUILD_FILTER:
    case SiriusPhysicalOperatorType::PASSTHROUGH_SINK:
    case SiriusPhysicalOperatorType::MERGE_SORT:
    case SiriusPhysicalOperatorType::MERGE_GROUP_BY:
    case SiriusPhysicalOperatorType::MERGE_TOP_N:
    case SiriusPhysicalOperatorType::MERGE_AGGREGATE:
    case SiriusPhysicalOperatorType::SORT_PARTITION:
    case SiriusPhysicalOperatorType::SORT_SAMPLE:
    case SiriusPhysicalOperatorType::GPU_VALUES:
    case SiriusPhysicalOperatorType::GPU_SCAN:
    case SiriusPhysicalOperatorType::STREAMING_SOURCE:
    case SiriusPhysicalOperatorType::STREAMING_SINK: return {};
  }
  return {};  // unreachable
}

namespace {

bool steps_are_followable(sirius::op::sirius_physical_operator const& node,
                          std::vector<descent_step> const& steps)
{
  if (steps.empty()) { return false; }
  for (auto const& step : steps) {
    if (step.child_index >= node.children.size() || node.children[step.child_index] == nullptr) {
      return false;
    }
  }
  return true;
}

bool is_aggregate(sirius::op::SiriusPhysicalOperatorType type) noexcept
{
  using sirius::op::SiriusPhysicalOperatorType;
  return type == SiriusPhysicalOperatorType::HASH_GROUP_BY ||
         type == SiriusPhysicalOperatorType::PERFECT_HASH_GROUP_BY ||
         type == SiriusPhysicalOperatorType::PARTITIONED_AGGREGATE ||
         type == SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE ||
         type == SiriusPhysicalOperatorType::MERGE_GROUP_BY ||
         type == SiriusPhysicalOperatorType::MERGE_AGGREGATE;
}

struct reader_site {
  sirius::op::sirius_physical_operator const* scan = nullptr;
  bool under_aggregate                             = false;
};

void collect_readers(sirius::op::sirius_physical_operator const& node,
                     std::size_t cte_table_index,
                     bool under_aggregate,
                     std::vector<reader_site>& readers)
{
  if (node.type == sirius::op::SiriusPhysicalOperatorType::CTE_SCAN) {
    if (node.Cast<sirius::op::sirius_physical_column_data_scan>().cte_index == cte_table_index) {
      readers.push_back({.scan = &node, .under_aggregate = under_aggregate});
    }
    return;
  }
  bool const below = under_aggregate || is_aggregate(node.type);
  for (auto const& child : node.children) {
    if (child) { collect_readers(*child, cte_table_index, below, readers); }
  }
}

// A delim CTE has two readers: the outer rows, and their distinct keys feeding the subquery.
// A filter on the outer-row reader only drops outer rows the join would drop anyway, and the
// key reader then loses only keys of dropped rows. A filter on the key reader would instead
// drop outer rows the rest of the plan still needs, so the reader must be provably the first.
bool is_delim_outer_row_reader(sirius::op::sirius_physical_cte const& cte,
                               sirius::op::sirius_physical_operator const& scan)
{
  constexpr std::string_view delim_prefix = "__duckdb_delim_";
  constexpr std::string_view dedup_prefix = "__duckdb_delim_dedup_";
  std::string_view const name             = cte.ctename;
  if (name.substr(0, delim_prefix.size()) != delim_prefix ||
      name.substr(0, dedup_prefix.size()) == dedup_prefix || cte.children.size() != 2 ||
      !cte.children[1]) {
    return false;
  }
  std::vector<reader_site> readers;
  collect_readers(*cte.children[1], cte.table_index, /*under_aggregate=*/false, readers);
  if (readers.size() != 2) { return false; }
  for (std::size_t i = 0; i < 2; ++i) {
    if (readers[i].scan == &scan) {
      return !readers[i].under_aggregate && readers[1 - i].under_aggregate;
    }
  }
  return false;
}

using cte_stack = std::vector<sirius::op::sirius_physical_cte const*>;

void trace_probe_key_into(sirius::op::sirius_physical_operator& node,
                          std::size_t ordinal,
                          descent_policy policy,
                          cte_stack& ctes,
                          std::vector<route_terminal>& terminals)
{
  auto const steps = descent_steps(node, ordinal, policy);
  if (!steps_are_followable(node, steps)) {
    terminals.push_back(route_terminal{.node = &node, .ordinal = ordinal});
    if (node.type != sirius::op::SiriusPhysicalOperatorType::CTE_SCAN) { return; }
    auto const table_index = node.Cast<sirius::op::sirius_physical_column_data_scan>().cte_index;
    auto const owner       = std::find_if(ctes.begin(), ctes.end(), [&](auto const* cte) {
      return cte->table_index == table_index;
    });
    if (owner == ctes.end() || !is_delim_outer_row_reader(**owner, node) ||
        ordinal >= (*owner)->children[0]->types.size()) {
      return;
    }
    // Only scans are registered: a filter on a body that ends in anything else gains nothing.
    cte_stack body_ctes;
    std::vector<route_terminal> body;
    trace_probe_key_into(*(*owner)->children[0], ordinal, policy, body_ctes, body);
    bool const all_scans = !body.empty() && std::all_of(body.begin(), body.end(), [](auto const& t) {
      return t.node->type == sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN;
    });
    if (!all_scans) { return; }
    for (auto& terminal : body) {
      terminal.in_cte_body    = true;
      terminal.cte_table_index = table_index;
      terminals.push_back(terminal);
    }
    return;
  }
  bool const is_cte = node.type == sirius::op::SiriusPhysicalOperatorType::CTE;
  if (is_cte) { ctes.push_back(&node.Cast<sirius::op::sirius_physical_cte>()); }
  for (auto const& step : steps) {
    trace_probe_key_into(
      *node.children[step.child_index], step.child_ordinal, policy, ctes, terminals);
  }
  if (is_cte) { ctes.pop_back(); }
}

}  // namespace

std::vector<route_terminal> trace_probe_key(sirius::op::sirius_physical_operator& root,
                                            std::size_t a0,
                                            descent_policy policy)
{
  std::vector<route_terminal> terminals;
  cte_stack ctes;
  trace_probe_key_into(root, a0, policy, ctes, terminals);
  return terminals;
}

endpoint_placement place_endpoint(duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree,
                                  std::size_t a0,
                                  descent_policy policy,
                                  endpoint_factory const& make_endpoint)
{
  assert(subtree != nullptr);
  auto const steps = descent_steps(*subtree, a0, policy);
  if (steps_are_followable(*subtree, steps)) {
    // Ascending child order keeps site ordinals aligned with trace_probe_key() terminals.
    endpoint_placement result;
    for (auto const& step : steps) {
      auto& child_slot = subtree->children[step.child_index];
      auto placed =
        place_endpoint(std::move(child_slot), step.child_ordinal, policy, make_endpoint);
      child_slot = std::move(placed.subtree);
      result.site_ordinals.insert(
        result.site_ordinals.end(), placed.site_ordinals.begin(), placed.site_ordinals.end());
    }
    result.subtree = std::move(subtree);
    return result;
  }
  auto endpoint = make_endpoint(*subtree);
  endpoint->children.push_back(std::move(subtree));
  return endpoint_placement{.subtree = std::move(endpoint), .site_ordinals = {a0}};
}

}  // namespace sirius::planner
