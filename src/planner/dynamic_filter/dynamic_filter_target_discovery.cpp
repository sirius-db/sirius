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

// sirius
#include <cudf/utilities/traits.hpp>

#include <expression/ast/node.hpp>
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <op/sirius_physical_projection.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <planner/dynamic_filter/dynamic_filter_target_discovery.hpp>

// stdlib
#include <cassert>
#include <iterator>
#include <span>
#include <variant>

namespace sirius::planner {

namespace {

// The probe/left block carries the traced value out un-null-padded for these join types. RIGHT and
// FULL_OUTER (OUTER) null-pad the left block; SINGLE is unsupported by the GPU join.
bool probe_block_is_value_preserving(duckdb::JoinType join_type) noexcept
{
  switch (join_type) {
    case duckdb::JoinType::INNER:
    case duckdb::JoinType::LEFT:
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::MARK: return true;
    case duckdb::JoinType::RIGHT:
    case duckdb::JoinType::OUTER:
    case duckdb::JoinType::SINGLE:
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::RIGHT_ANTI:
    case duckdb::JoinType::INVALID: return false;
  }
  return false;  // unreachable
}

// For INNER and LEFT joins, removing a build row only removes a result row or creates a NULL-padded
// row that the producing join later drops. Other join types are not safe build-block routes.
bool build_block_is_value_preserving(duckdb::JoinType join_type) noexcept
{
  switch (join_type) {
    case duckdb::JoinType::INNER:
    case duckdb::JoinType::LEFT: return true;
    case duckdb::JoinType::RIGHT:
    case duckdb::JoinType::OUTER:
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::MARK:
    case duckdb::JoinType::SINGLE:
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::RIGHT_ANTI:
    case duckdb::JoinType::INVALID: return false;
  }
  return false;  // unreachable;
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
  // A negative entry would wrap into an enormous ordinal that no caller could detect as invalid.
  if (group_idx[output_ordinal] < 0) { return std::nullopt; }
  return static_cast<std::size_t>(group_idx[output_ordinal]);
}

bool boundary_key_matches_site_type(cudf::data_type key_storage_type,
                                    cudf::data_type site_column_type) noexcept
{
  // Plain equality, for every key type. The traced hops are all value- and type-preserving
  // (reference-only projections, filter gathers, row masks, positional group-key maps), so an
  // admitted key and the column it binds always agree; requiring that rather than assuming it
  // costs nothing and removes the question of which types carry a scale.
  return key_storage_type == site_column_type;
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
    if (!probe_block_is_value_preserving(join_type)) { return std::nullopt; }
    return descent_step{
      .child_index   = 0,
      .child_ordinal = static_cast<std::size_t>(probe_block_output_columns[output_ordinal])};
  }
  if (!policy.descend_build_blocks) { return std::nullopt; }
  if (!build_block_is_value_preserving(join_type)) { return std::nullopt; }
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
      // The Top-N self-trace refuses aggregates: an aggregate is a FULL barrier that also
      // destroys key lineage, so no threshold can reach below it.
      if (policy.top_n_self_trace) { return {}; }
      auto const& aggregate = node.Cast<sirius::op::sirius_physical_grouped_aggregate>();
      auto const input =
        group_by_key_input(aggregate.group_idx, aggregate.grouping_sets.size(), output_ordinal);
      if (!input.has_value()) { return {}; }
      return {descent_step{.child_index = 0, .child_ordinal = *input}};
    }
    case SiriusPhysicalOperatorType::HASH_JOIN: {
      // Join hops are a future per-proven-hop widening, not the minimal self-trace set.
      if (policy.top_n_self_trace) { return {}; }
      auto const& join = node.Cast<sirius::op::sirius_physical_hash_join>();
      return as_steps(join_block_descent(join.join_type,
                                         join.lhs_output_columns.col_idxs,
                                         join.rhs_output_columns.col_idxs,
                                         output_ordinal,
                                         policy));
    }
    // A filter is a row predicate over unchanged columns: value-preserving through its output
    // gather and commuting with any other independent row predicate applied below it.
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
    // A physical union's output is positionally aligned with every child by construction, so the
    // trace fans out into each child at the same ordinal.
    case SiriusPhysicalOperatorType::UNION: {
      // The Top-N self-trace stops here. Set operations are rejected during planning today, so a
      // physical UNION is never constructed and this hop could not be exercised; carrying an
      // untestable multi-branch path (and the guards it would need) is worse than terminating,
      // which is always sound. The hop returns with set-operation support, together with a test
      // that can run. The join policy keeps the fan-out unchanged.
      if (policy.top_n_self_trace) { return {}; }
      std::vector<descent_step> steps;
      steps.reserve(node.children.size());
      for (std::size_t child_index = 0; child_index < node.children.size(); ++child_index) {
        assert(node.children[child_index] != nullptr);
        steps.push_back(descent_step{.child_index = child_index, .child_ordinal = output_ordinal});
      }
      return steps;
    }
    // A row mask passes every column through unchanged and commutes with any other row filter, so a
    // later key's endpoint may descend past an earlier key's rather than stopping on it.
    case SiriusPhysicalOperatorType::DYNAMIC_FILTER:
      return {descent_step{.child_index = 0, .child_ordinal = output_ordinal}};
    case SiriusPhysicalOperatorType::INVALID:
    case SiriusPhysicalOperatorType::ORDER_BY:
    case SiriusPhysicalOperatorType::LIMIT:
    case SiriusPhysicalOperatorType::STREAMING_LIMIT:
    case SiriusPhysicalOperatorType::LIMIT_PERCENT:
    case SiriusPhysicalOperatorType::TOP_N:
    case SiriusPhysicalOperatorType::WINDOW:
    case SiriusPhysicalOperatorType::UNNEST:
    case SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
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
    case SiriusPhysicalOperatorType::CTE:
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
    case SiriusPhysicalOperatorType::PARTITION:
    case SiriusPhysicalOperatorType::CONCAT:
    case SiriusPhysicalOperatorType::MERGE_SORT:
    case SiriusPhysicalOperatorType::MERGE_GROUP_BY:
    case SiriusPhysicalOperatorType::MERGE_TOP_N:
    case SiriusPhysicalOperatorType::MERGE_AGGREGATE:
    case SiriusPhysicalOperatorType::SORT_PARTITION:
    case SiriusPhysicalOperatorType::SORT_SAMPLE:
    case SiriusPhysicalOperatorType::GPU_VALUES:
    case SiriusPhysicalOperatorType::GPU_SCAN:
    case SiriusPhysicalOperatorType::STREAMING_SOURCE: return {};
  }
  return {};  // unreachable
}

namespace {

// A missing child turns the current node into a terminal.
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

void trace_probe_key_into(sirius::op::sirius_physical_operator& node,
                          std::size_t ordinal,
                          descent_policy policy,
                          std::vector<route_terminal>& terminals)
{
  auto const steps = descent_steps(node, ordinal, policy);
  if (!steps_are_followable(node, steps)) {
    terminals.push_back(route_terminal{.node = &node, .ordinal = ordinal});
    return;
  }
  for (auto const& step : steps) {
    trace_probe_key_into(*node.children[step.child_index], step.child_ordinal, policy, terminals);
  }
}

}  // namespace

std::vector<route_terminal> trace_probe_key(sirius::op::sirius_physical_operator& root,
                                            std::size_t a0,
                                            descent_policy policy)
{
  std::vector<route_terminal> terminals;
  trace_probe_key_into(root, a0, policy, terminals);
  return terminals;
}

bool hop_is_material(sirius::op::sirius_physical_operator const& node) noexcept
{
  // The two-operator rule: a FILTER's predicate always runs per row, and a PROJECTION is material
  // exactly when any select-list entry is a non-reference expression. Cost-blindness, the single
  // definition of "reference", and why existing DYNAMIC_FILTER endpoints stay immaterial are the
  // header contract's.
  if (node.type == sirius::op::SiriusPhysicalOperatorType::FILTER) { return true; }
  if (node.type == sirius::op::SiriusPhysicalOperatorType::PROJECTION) {
    auto const& projection = node.Cast<sirius::op::sirius_physical_projection>();
    for (auto const& entry : projection.select_list) {
      if (!projection_reference_input(*entry).has_value()) { return true; }
    }
  }
  return false;
}

namespace {

/// One accepted all-keys hop: the shared child shape plus each key's remapped ordinal.
struct all_keys_step {
  std::size_t child_index = 0;
  std::vector<std::size_t> child_ordinals;
};

/**
 * @brief The hops every traced ordinal survives together, or empty when any ordinal stops here
 *
 * Each ordinal is remapped independently by @ref descent_steps, but the hop is accepted only when
 * every ordinal produces the same child shape -- otherwise the keys would part company and no
 * single site could carry the full tuple.
 */
std::vector<all_keys_step> all_keys_steps(sirius::op::sirius_physical_operator const& node,
                                          std::span<std::size_t const> ordinals,
                                          descent_policy policy)
{
  std::vector<std::vector<descent_step>> per_ordinal;
  per_ordinal.reserve(ordinals.size());
  for (auto const ordinal : ordinals) {
    auto steps = descent_steps(node, ordinal, policy);
    if (!steps_are_followable(node, steps)) { return {}; }
    per_ordinal.push_back(std::move(steps));
  }
  if (per_ordinal.empty()) { return {}; }

  auto const& shape = per_ordinal.front();
  for (auto const& steps : per_ordinal) {
    if (steps.size() != shape.size()) { return {}; }
    for (std::size_t i = 0; i < shape.size(); ++i) {
      if (steps[i].child_index != shape[i].child_index) { return {}; }
    }
  }

  std::vector<all_keys_step> accepted;
  accepted.reserve(shape.size());
  for (std::size_t i = 0; i < shape.size(); ++i) {
    all_keys_step step{.child_index = shape[i].child_index, .child_ordinals = {}};
    step.child_ordinals.reserve(ordinals.size());
    for (auto const& steps : per_ordinal) {
      step.child_ordinals.push_back(steps[i].child_ordinal);
    }
    accepted.push_back(std::move(step));
  }
  return accepted;
}

void trace_top_n_all_keys_into(sirius::op::sirius_physical_operator& node,
                               std::vector<std::size_t> ordinals,
                               descent_policy policy,
                               std::size_t material_hops,
                               std::vector<multi_key_route_terminal>& terminals)
{
  auto const steps = all_keys_steps(node, ordinals, policy);
  if (steps.empty()) {
    terminals.push_back(multi_key_route_terminal{
      .node = &node, .ordinals = std::move(ordinals), .material_hops = material_hops});
    return;
  }
  auto const hops = material_hops + (hop_is_material(node) ? 1 : 0);
  for (auto const& step : steps) {
    trace_top_n_all_keys_into(
      *node.children[step.child_index], step.child_ordinals, policy, hops, terminals);
  }
}

}  // namespace

std::vector<multi_key_route_terminal> trace_top_n_all_keys(
  sirius::op::sirius_physical_operator& root,
  std::span<std::size_t const> key_ordinals,
  descent_policy policy)
{
  std::vector<multi_key_route_terminal> terminals;
  if (key_ordinals.empty()) { return terminals; }
  trace_top_n_all_keys_into(
    root, std::vector<std::size_t>{key_ordinals.begin(), key_ordinals.end()}, policy, 0, terminals);
  return terminals;
}

bool is_parquet_reader_function(std::string const& function_name) noexcept
{
  return function_name == "parquet_scan" || function_name == "read_parquet" ||
         function_name == "sirius_read_parquet";
}

bool target_skips_reads(sirius::op::sirius_physical_operator const& node) noexcept
{
  if (node.type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) { return false; }
  return is_parquet_reader_function(
    node.Cast<sirius::op::sirius_physical_table_scan>().function.name);
}

top_n_target_kind classify_top_n_terminal(route_terminal const& terminal,
                                          sirius::op::top_n_filter_layer layer,
                                          std::size_t material_hops_above,
                                          bool consumer_skips_reads,
                                          bool coincides_with_lex_site)
{
  // The LEX predicate implies the inclusive first-key bound, so a site already receiving LEX must
  // not also receive FIRST_KEY.
  if (layer == sirius::op::top_n_filter_layer::FIRST_KEY && coincides_with_lex_site) {
    return top_n_target_kind::SUBSUMED_BY_LEX;
  }
  // The siting rule, applied to every terminal alike: the target either reads less because of the
  // predicate, or shields per-row work the sink would otherwise repeat over the same rows. Meeting
  // neither, its compaction pass buys back exactly the pass sink self-consumption already makes.
  if (!consumer_skips_reads && material_hops_above == 0) {
    return top_n_target_kind::SKIPPED_NO_WORK_SAVED;
  }
  assert(terminal.node != nullptr);
  if (terminal.node->type == sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    return top_n_target_kind::SCAN_BIND;
  }
  return top_n_target_kind::ENDPOINT_SITE;
}

endpoint_placement place_endpoint(duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree,
                                  std::size_t a0,
                                  descent_policy policy,
                                  endpoint_factory const& make_endpoint)
{
  // A singleton ordinal set reduces all_keys_steps to descent_steps plus followability, so the
  // delegation keeps this signature's behavior exactly, UNION fan-out included.
  auto placed = place_endpoint_all_keys(
    std::move(subtree), std::span<std::size_t const>{&a0, 1}, policy, make_endpoint);
  endpoint_placement result;
  result.subtree = std::move(placed.subtree);
  result.site_ordinals.reserve(placed.site_ordinals.size());
  for (auto const& site : placed.site_ordinals) {
    assert(site.size() == 1);
    result.site_ordinals.push_back(site.front());
  }
  return result;
}

multi_key_endpoint_placement place_endpoint_all_keys(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> subtree,
  std::span<std::size_t const> ordinals,
  descent_policy policy,
  endpoint_factory const& make_endpoint)
{
  assert(subtree != nullptr);
  assert(!ordinals.empty());
  auto const steps = all_keys_steps(*subtree, ordinals, policy);
  if (!steps.empty()) {
    // Ascending child order keeps site ordinals aligned with the order both traces report
    // terminals in.
    multi_key_endpoint_placement result;
    for (auto const& step : steps) {
      auto& child_slot = subtree->children[step.child_index];
      auto placed =
        place_endpoint_all_keys(std::move(child_slot), step.child_ordinals, policy, make_endpoint);
      child_slot = std::move(placed.subtree);
      result.site_ordinals.insert(result.site_ordinals.end(),
                                  std::make_move_iterator(placed.site_ordinals.begin()),
                                  std::make_move_iterator(placed.site_ordinals.end()));
    }
    result.subtree = std::move(subtree);
    return result;
  }
  // Deepest site every ordinal reaches together: the endpoint becomes this operator's new parent.
  auto endpoint = make_endpoint(*subtree);
  endpoint->children.push_back(std::move(subtree));
  return multi_key_endpoint_placement{
    .subtree       = std::move(endpoint),
    .site_ordinals = {std::vector<std::size_t>{ordinals.begin(), ordinals.end()}}};
}

}  // namespace sirius::planner
