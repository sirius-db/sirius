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
#include <expression/ast/node.hpp>
#include <op/sirius_physical_grouped_aggregate.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <op/sirius_physical_projection.hpp>
#include <planner/dynamic_filter_endpoint_placement.hpp>

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

// The build/right block is a direct projection of the build input's columns, and removing a build
// row only removes or NULL-pads rows P later drops, for INNER and LEFT, respectively. Every other
// type refuses: SEMI/ANTI emit no build block, MARK's build-block ordinal is the synthetic mark,
// RIGHT/FULL/RIGHT_SEMI/RIGHT_ANTI are deferred, SINGLE is unsupported.
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
  std::size_t output_ordinal)
{
  auto const probe_block_size = probe_block_output_columns.size();
  if (output_ordinal < probe_block_size) {
    if (!probe_block_is_value_preserving(join_type)) { return std::nullopt; }
    return descent_step{
      .child_index   = 0,
      .child_ordinal = static_cast<std::size_t>(probe_block_output_columns[output_ordinal])};
  }
  if (!build_block_is_value_preserving(join_type)) { return std::nullopt; }
  auto const build_ordinal = output_ordinal - probe_block_size;
  if (build_ordinal >= build_block_output_columns.size()) { return std::nullopt; }
  return descent_step{
    .child_index   = 1,
    .child_ordinal = static_cast<std::size_t>(build_block_output_columns[build_ordinal])};
}

std::optional<descent_step> pass_through_step(sirius::op::sirius_physical_operator const& node,
                                              std::size_t output_ordinal)
{
  using sirius::op::SiriusPhysicalOperatorType;
  switch (node.type) {
    case SiriusPhysicalOperatorType::PROJECTION: {
      auto const& projection = node.Cast<sirius::op::sirius_physical_projection>();
      if (output_ordinal >= projection.select_list.size()) { return std::nullopt; }
      auto const input = projection_reference_input(*projection.select_list[output_ordinal]);
      if (!input.has_value()) { return std::nullopt; }
      return descent_step{.child_index = 0, .child_ordinal = *input};
    }
    case SiriusPhysicalOperatorType::HASH_GROUP_BY: {
      auto const& aggregate = node.Cast<sirius::op::sirius_physical_grouped_aggregate>();
      auto const input =
        group_by_key_input(aggregate.group_idx, aggregate.grouping_sets.size(), output_ordinal);
      if (!input.has_value()) { return std::nullopt; }
      return descent_step{.child_index = 0, .child_ordinal = *input};
    }
    case SiriusPhysicalOperatorType::HASH_JOIN: {
      auto const& join = node.Cast<sirius::op::sirius_physical_hash_join>();
      return join_block_descent(join.join_type,
                                join.lhs_output_columns.col_idxs,
                                join.rhs_output_columns.col_idxs,
                                output_ordinal);
    }
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
    case SiriusPhysicalOperatorType::FILTER:
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
    case SiriusPhysicalOperatorType::UNION:
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
    case SiriusPhysicalOperatorType::DYNAMIC_FILTER:
    case SiriusPhysicalOperatorType::STREAMING_SOURCE: return std::nullopt;
  }
  return std::nullopt;  // unreachable
}

endpoint_site resolve_endpoint_site(sirius::op::sirius_physical_operator* probe_subtree_root,
                                    std::size_t a0)
{
  endpoint_site deepest{.node = probe_subtree_root, .ordinal = a0};
  auto* node       = probe_subtree_root;
  auto current_ord = a0;
  while (node != nullptr) {
    auto const step = pass_through_step(*node, current_ord);
    if (!step.has_value()) { break; }
    if (step->child_index >= node->children.size()) { break; }
    auto* child = node->children[step->child_index].get();
    if (child == nullptr) { break; }
    node        = child;
    current_ord = step->child_ordinal;
    deepest     = endpoint_site{.node = node, .ordinal = current_ord};
  }
  return deepest;
}

}  // namespace sirius::planner