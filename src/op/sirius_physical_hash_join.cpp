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

#include "op/sirius_physical_hash_join.hpp"

#include "config.hpp"
#include "cudf/aggregation.hpp"
#include "cudf/copying.hpp"
#include "cudf/join/distinct_hash_join.hpp"
#include "cudf/join/filtered_join.hpp"
#include "cudf/join/join.hpp"
#include "cudf/join/mark_join.hpp"
#include "cudf/join/mixed_join.hpp"
#include "cudf/null_mask.hpp"
#include "cudf/reduction.hpp"
#include "cudf/reduction/distinct_count.hpp"
#include "cudf/table/table_view.hpp"
#include "cudf/transform.hpp"
#include "cudf/types.hpp"
#include "cudf/utilities/memory_resource.hpp"
#include "cudf/version_config.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression_evaluator/ast_supported_types.hpp"
#include "expression_evaluator/gpu_expression_translator_internal.hpp"
#include "helper/numeric_narrowing.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace sirius {
namespace op {

/// Recursively collect all BoundReferenceExpression indices from an expression tree.
static void collect_bound_ref_indices(const duckdb::Expression& expr,
                                      std::unordered_set<std::size_t>& indices)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    indices.insert(expr.Cast<duckdb::BoundReferenceExpression>().index);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expr, [&](const duckdb::Expression& child) { collect_bound_ref_indices(child, indices); });
}

// Mixed plain/null-safe keys require different null policies, so route the null-safe keys
// to the conditional predicate. MARK joins cannot use MIXED_JOIN.
static bool wants_null_safe_routing(duckdb::vector<sirius::join_condition> const& conditions,
                                    duckdb::JoinType join_type)
{
  if (join_type == duckdb::JoinType::MARK) { return false; }
  bool has_plain_equal = false;
  bool has_null_safe   = false;
  for (auto const& c : conditions) {
    if (c.comparison == sirius::comparison_type::equal) {
      has_plain_equal = true;
    } else if (c.comparison == sirius::comparison_type::not_distinct_from) {
      has_null_safe = true;
    }
  }
  return has_plain_equal && has_null_safe;
}

// Every condition is null-safe (IS NOT DISTINCT FROM). For a MARK join this is the encodable
// case: null_equality::EQUAL covers every key, and a null-safe comparison is never UNKNOWN, so
// every mark is definite.
static bool all_keys_null_safe(duckdb::vector<sirius::join_condition> const& conditions)
{
  return !conditions.empty() &&
         std::all_of(conditions.begin(), conditions.end(), [](auto const& c) {
           return c.comparison == sirius::comparison_type::not_distinct_from;
         });
}

// A MARK join mixing null-safe keys with anything else (a plain `=`, or an inequality). cuDF takes
// one null_equality for every key and MARK does not route null-safe keys into the mixed-join
// predicate, so the null-safe key would silently inherit UNEQUAL. Callers must reject this shape.
static bool mark_join_mixes_null_safe_keys(duckdb::vector<sirius::join_condition> const& conditions)
{
  bool const has_null_safe = std::any_of(conditions.begin(), conditions.end(), [](auto const& c) {
    return c.comparison == sirius::comparison_type::not_distinct_from;
  });
  return has_null_safe && !all_keys_null_safe(conditions);
}

// Whether a condition belongs in the hash key: a plain `=`, or a null-safe key when it
// is not being routed to the conditional predicate (route_null_safe_to_conditional).
static bool is_hash_equality_key(sirius::comparison_type c, bool route_null_safe_to_conditional)
{
  if (c == sirius::comparison_type::equal) { return true; }
  if (c == sirius::comparison_type::not_distinct_from) { return !route_null_safe_to_conditional; }
  return false;
}

// Routed keys must be references or use one of cuDF AST's three supported casts.
// The planner materializes other expressions into referenced columns before this check.
static bool is_ast_translatable_key_side(sirius::ast::node const& side)
{
  if (side.holds<sirius::ast::reference>()) { return true; }
  if (side.holds<sirius::ast::cast>()) {
    auto const& c         = side.get<sirius::ast::cast>();
    auto const& supported = sirius::supported_ast_cast_types_native;
    return std::find(supported.begin(), supported.end(), c.target_type.id()) != supported.end() &&
           is_ast_translatable_key_side(*c.child);
  }
  return false;
}

// Reject routing unless every null-safe key can be represented by the cuDF AST.
static bool null_safe_keys_are_ast_routable(
  duckdb::vector<sirius::join_condition> const& conditions)
{
  for (auto const& c : conditions) {
    if (c.comparison != sirius::comparison_type::not_distinct_from) { continue; }
    if (!c.left || !c.right) { return false; }
    if (!is_ast_translatable_key_side(*c.left) || !is_ast_translatable_key_side(*c.right)) {
      return false;
    }
  }
  return true;
}

static cudf::filtered_join make_right_filtered_join(cudf::table_view const& right_keys,
                                                    cudf::null_equality compare_nulls,
                                                    rmm::cuda_stream_view stream)
{
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 6)
  return cudf::filtered_join(right_keys, compare_nulls, stream);
#else
  return cudf::filtered_join(right_keys, compare_nulls, cudf::set_as_build_table::RIGHT, stream);
#endif
}

// Heap-allocated variant for BUILD_PROBE mode, where one filtered_join is built once on the right
// (filter) keys and reused across many streamed left probe batches via semi_join.
static std::unique_ptr<cudf::filtered_join> make_right_filtered_join_ptr(
  cudf::table_view const& right_keys,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream)
{
#if CUDF_VERSION_MAJOR > 26 || (CUDF_VERSION_MAJOR == 26 && CUDF_VERSION_MINOR >= 6)
  return std::make_unique<cudf::filtered_join>(right_keys, compare_nulls, stream);
#else
  return std::make_unique<cudf::filtered_join>(
    right_keys, compare_nulls, cudf::set_as_build_table::RIGHT, stream);
#endif
}

// Build the semi-join hash table on the left/output side and probe with the (larger) right side.
// Wins over make_right_filtered_join only when the left side is substantially smaller than the
// right; gated by mark_join_build_switch_ratio at the call site.
static cudf::mark_join make_left_mark_join(cudf::table_view const& left_keys,
                                           cudf::null_equality compare_nulls,
                                           rmm::cuda_stream_view stream)
{
  return cudf::mark_join(left_keys, compare_nulls, cudf::join_prefilter::NO, stream);
}

bool sirius_physical_hash_join::is_join_type_supported(duckdb::JoinType join_type)
{
  // Keep in lockstep with the join-type dispatch in execute() and its BUILD_PROBE counterpart.
  switch (join_type) {
    case duckdb::JoinType::INNER:
    case duckdb::JoinType::LEFT:
    case duckdb::JoinType::RIGHT:
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::RIGHT_ANTI:
    case duckdb::JoinType::MARK:
    case duckdb::JoinType::OUTER: return true;
    // SINGLE (at most one build row per probe row, NULL-padded otherwise) has no arm in any
    // dispatch.
    default: return false;
  }
}

bool sirius_physical_hash_join::are_conditions_supported(
  duckdb::vector<sirius::join_condition>& conditions, duckdb::JoinType join_type)
{
  if (!is_join_type_supported(join_type)) { return false; }

  // All-null-safe MARK is fine (EQUAL keys, definite marks); only the mixture is unencodable.
  if (join_type == duckdb::JoinType::MARK && mark_join_mixes_null_safe_keys(conditions)) {
    return false;
  }

  // Keep support validation and constructor key classification identical.
  bool const route_null_safe = wants_null_safe_routing(conditions, join_type);

  // Unsupported casts should normally have been materialized by the planner; reject any
  // remaining untranslatable key rather than failing during execution.
  if (route_null_safe && !null_safe_keys_are_ast_routable(conditions)) { return false; }

  // Must have at least one hash-equality condition for a hash-based join.
  bool has_equality = false;
  for (auto const& cond : conditions) {
    if (is_hash_equality_key(cond.comparison, route_null_safe)) {
      has_equality = true;
      break;
    }
  }
  if (!has_equality) { return false; }

  // Pure equality join: always supported.
  bool has_inequality = false;
  for (auto const& cond : conditions) {
    if (!is_hash_equality_key(cond.comparison, route_null_safe)) {
      has_inequality = true;
      break;
    }
  }
  if (!has_inequality) { return true; }

  // Mixed join: collect the column indices used on each side of the hash-equality conditions.
  std::unordered_set<std::size_t> equality_left_cols, equality_right_cols;
  for (auto const& cond : conditions) {
    if (!is_hash_equality_key(cond.comparison, route_null_safe)) { continue; }
    auto left_owned  = sirius::ast::to_duckdb(*cond.left);
    auto right_owned = sirius::ast::to_duckdb(*cond.right);
    collect_bound_ref_indices(*left_owned, equality_left_cols);
    collect_bound_ref_indices(*right_owned, equality_right_cols);
  }

  // For each conditional condition (inequality or a routed null-safe key), verify its
  // left/right column references don't overlap with the hash-key columns on the same side.
  // cuDF's mixed_join API requires the equality and conditional table columns to be disjoint.
  for (auto const& cond : conditions) {
    if (is_hash_equality_key(cond.comparison, route_null_safe)) { continue; }
    std::unordered_set<std::size_t> ineq_left_cols, ineq_right_cols;
    auto left_owned  = sirius::ast::to_duckdb(*cond.left);
    auto right_owned = sirius::ast::to_duckdb(*cond.right);
    collect_bound_ref_indices(*left_owned, ineq_left_cols);
    collect_bound_ref_indices(*right_owned, ineq_right_cols);
    for (auto const idx : ineq_left_cols) {
      if (equality_left_cols.count(idx) > 0) { return false; }
    }
    for (auto const idx : ineq_right_cols) {
      if (equality_right_cols.count(idx) > 0) { return false; }
    }
  }

  return true;
}

namespace {

void reorder_join_conditions(duckdb::vector<sirius::join_condition>& conditions,
                             bool route_null_safe_to_conditional)
{
  bool is_ordered     = true;
  bool seen_non_equal = false;
  for (auto& cond : conditions) {
    if (is_hash_equality_key(cond.comparison, route_null_safe_to_conditional)) {
      if (seen_non_equal) {
        is_ordered = false;
        break;
      }
    } else {
      seen_non_equal = true;
    }
  }
  if (is_ordered) { return; }
  duckdb::vector<sirius::join_condition> equal_conditions;
  duckdb::vector<sirius::join_condition> other_conditions;
  for (auto& cond : conditions) {
    if (is_hash_equality_key(cond.comparison, route_null_safe_to_conditional)) {
      equal_conditions.push_back(std::move(cond));
    } else {
      other_conditions.push_back(std::move(cond));
    }
  }
  conditions.clear();
  for (auto& cond : equal_conditions) {
    conditions.push_back(std::move(cond));
  }
  for (auto& cond : other_conditions) {
    conditions.push_back(std::move(cond));
  }
}

}  // namespace

sirius_physical_hash_join::sirius_physical_hash_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right,
  duckdb::vector<sirius::join_condition> cond,
  duckdb::JoinType join_type,
  const duckdb::vector<std::size_t>& left_projection_map,
  const duckdb::vector<std::size_t>& right_projection_map,
  duckdb::vector<sirius::logical_type> delim_types,
  std::size_t estimated_cardinality,
  uint64_t max_build_hash_table_bytes,
  dynamic_filter_publish_plan dynamic_filter_plan,
  uint64_t hash_partition_bytes,
  uint64_t max_broadcast_join_size,
  dynamic_filter_stats* dynamic_filter_stats_sink)
  : sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType::HASH_JOIN,
                                                sirius::from_duckdb_vec(op.types),
                                                estimated_cardinality),
    conditions(std::move(cond)),
    join_type(join_type),
    delim_types(std::move(delim_types)),
    _dynamic_filter_plan(std::move(dynamic_filter_plan))
{
  // Backstop for the planner's screen: throwing here still lands in plan generation, which falls
  // back to CPU, rather than aborting the query from execute().
  if (!is_join_type_supported(join_type)) {
    throw duckdb::NotImplementedException("sirius_physical_hash_join: unsupported join type: " +
                                          duckdb::JoinTypeToString(join_type));
  }

  _max_build_hash_table_bytes = max_build_hash_table_bytes;
  _hash_partition_bytes       = hash_partition_bytes;
  _max_broadcast_join_size    = max_broadcast_join_size;

  // Route mixed null-safe keys to a NULL_EQUAL predicate; plain `=` remains a hash key.
  bool const wants_routing   = wants_null_safe_routing(conditions, join_type);
  bool const route_null_safe = wants_routing && null_safe_keys_are_ast_routable(conditions);

  // Defensive backstop: never silently apply UNEQUAL semantics to a null-safe key.
  if (wants_routing && !route_null_safe) {
    throw std::runtime_error(
      "sirius_physical_hash_join: a null-safe (IS NOT DISTINCT FROM) key mixed with a plain `=` "
      "key carries an expression the cuDF AST cannot express (it only casts to INT64/UINT64/"
      "FLOAT64), so it cannot be routed into the mixed-join predicate");
  }
  reorder_join_conditions(conditions, route_null_safe);

  // Backstop for the planner's screen; reaching here would silently give the null-safe key
  // NULL != NULL semantics.
  if (join_type == duckdb::JoinType::MARK && mark_join_mixes_null_safe_keys(conditions)) {
    throw duckdb::NotImplementedException(
      "sirius_physical_hash_join: MARK join mixing a null-safe (IS NOT DISTINCT FROM) key with a "
      "plain key is not supported");
  }

  // Pure null-safe hash keys use EQUAL; any plain hash key requires UNEQUAL. Routed
  // null-safe keys get their semantics from NULL_EQUAL instead.
  compare_nulls_ = cudf::null_equality::UNEQUAL;
  if (join_type == duckdb::JoinType::MARK) {
    // All-null-safe MARK takes EQUAL and emits definite marks; any other MARK keeps UNEQUAL and
    // the three-valued IN/EXISTS reconstruction, which assumes a NULL key never matches.
    mark_is_null_safe_ = all_keys_null_safe(conditions);
    if (mark_is_null_safe_) { compare_nulls_ = cudf::null_equality::EQUAL; }
  } else {
    bool saw_null_safe   = false;
    bool saw_plain_equal = false;
    for (auto const& cond : conditions) {
      if (cond.comparison == sirius::comparison_type::equal) {
        saw_plain_equal = true;
      } else if (cond.comparison == sirius::comparison_type::not_distinct_from) {
        saw_null_safe = true;
      }
    }
    if (saw_null_safe && !saw_plain_equal) { compare_nulls_ = cudf::null_equality::EQUAL; }
  }

  _dynamic_filter_stats = dynamic_filter_stats_sink;
  if (_dynamic_filter_stats != nullptr && _dynamic_filter_plan.enabled()) {
    _dynamic_filter_stats->producers_enabled.fetch_add(1, std::memory_order_relaxed);
  }

  children.push_back(std::move(left));
  children.push_back(std::move(right));

  auto& lhs_input_types = children[0]->get_types();

  if (left_projection_map.empty()) {
    lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
    for (std::size_t i = 0; i < lhs_input_types.size(); i++) {
      lhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(i));
    }
  } else {
    lhs_output_columns.col_idxs.reserve(left_projection_map.size());
    for (auto& col_idx : left_projection_map) {
      if (col_idx < lhs_input_types.size()) {
        lhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(col_idx));
      } else {
        printf("WARNING:In sirius_physical_hash_join: left_projection_map index out of range");
      }
    }
  }

  for (auto& lhs_col : lhs_output_columns.col_idxs) {
    auto& lhs_col_type = lhs_input_types[lhs_col];
    lhs_output_columns.col_types.push_back(lhs_col_type);
  }

  auto& rhs_input_types = children[1]->get_types();

  if (right_projection_map.empty()) {
    rhs_output_columns.col_idxs.reserve(rhs_input_types.size());
    for (std::size_t i = 0; i < rhs_input_types.size(); i++) {
      rhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(i));
    }
  } else {
    rhs_output_columns.col_idxs.reserve(right_projection_map.size());
    for (auto& col_idx : right_projection_map) {
      if (col_idx < rhs_input_types.size()) {
        rhs_output_columns.col_idxs.emplace_back(static_cast<cudf::size_type>(col_idx));
      } else {
        printf("WARNING:In sirius_physical_hash_join: right_projection_map index out of range");
      }
    }
  }

  for (auto& rhs_col : rhs_output_columns.col_idxs) {
    auto& rhs_col_type = rhs_input_types[rhs_col];
    rhs_output_columns.col_types.push_back(rhs_col_type);
  }

  for (auto& condition : conditions) {
    auto left_owned        = sirius::ast::to_duckdb(*condition.left);
    auto right_owned       = sirius::ast::to_duckdb(*condition.right);
    auto const* left_expr  = left_owned.get();
    auto const* right_expr = right_owned.get();

    if (!is_hash_equality_key(condition.comparison, route_null_safe)) {
      // Inequality (and routed null-safe) conditions are handled at execute time via the
      // cuDF mixed_join binary predicate. No key index extraction is needed here.
      continue;
    }

    is_all_inequality_join = false;
    num_equality_conditions++;

    // Extract left key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    key_cast_info cast_info;
    auto left_class  = left_expr->GetExpressionClass();
    auto right_class = right_expr->GetExpressionClass();

    if (left_class == duckdb::ExpressionClass::BOUND_REF) {
      left_key_col_indices.push_back(left_expr->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (left_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = left_expr->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error(
          "Unsupported join condition: BOUND_CAST child is not BOUND_REF (left)");
      }
      left_key_col_indices.push_back(
        bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_left        = true;
      cast_info.left_target_type = duckdb::GetCudfType(left_expr->return_type);
      cast_necessary             = true;
    } else {
      throw std::runtime_error("Unsupported join condition left expression");
    }

    // Extract right key index (may be BOUND_REF or BOUND_CAST wrapping a BOUND_REF)
    if (right_class == duckdb::ExpressionClass::BOUND_REF) {
      right_key_col_indices.push_back(right_expr->Cast<duckdb::BoundReferenceExpression>().index);
    } else if (right_class == duckdb::ExpressionClass::BOUND_CAST) {
      auto& bound_cast = right_expr->Cast<duckdb::BoundCastExpression>();
      if (bound_cast.child->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        throw std::runtime_error(
          "Unsupported join condition: BOUND_CAST child is not BOUND_REF (right)");
      }
      right_key_col_indices.push_back(
        bound_cast.child->Cast<duckdb::BoundReferenceExpression>().index);
      cast_info.cast_right        = true;
      cast_info.right_target_type = duckdb::GetCudfType(right_expr->return_type);
      cast_necessary              = true;
    } else {
      throw std::runtime_error("Unsupported join condition right expression");
    }

    key_casts.push_back(cast_info);
  }

  // Mixed join: has at least one equality condition (for hashing) and at least one inequality
  // condition (for the binary predicate).
  if (!is_all_inequality_join && (num_equality_conditions < conditions.size())) {
    // A MARK join must run in BUILD_PROBE mode (it needs the full build resident for build_has_null
    // and per-row marks)
    if (join_type == duckdb::JoinType::MARK) {
      throw std::runtime_error(
        "sirius_physical_hash_join: MARK join with mixed (equality + inequality) conditions is not "
        "supported");
    }
    _join_mode = HASH_JOIN_MODE::MIXED_JOIN;
  }
};

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_hash_join::build_join_pipelines(pipeline::sirius_pipeline& current,
                                                     pipeline::sirius_meta_pipeline& meta_pipeline,
                                                     sirius_physical_operator& op)
{
  auto& state = meta_pipeline.get_state();
  state.add_pipeline_operator(current, op);

  // Plan-time wrap_join inserted CONCAT_build → PARTITION_build → original_build as
  // op.children[1]. Use CONCAT_build as the build_meta sink (not `op`) to avoid a
  // redundant [op] build pipeline, and recurse past it so CONCAT_build's own
  // build_pipelines doesn't create a duplicate sink meta.
  auto& build_child = *op.children[1];
  D_ASSERT(build_child.is_sink());
  D_ASSERT(!build_child.children.empty());
  auto& build_meta = meta_pipeline.create_child_meta_pipeline(current, build_child);
  build_meta.build(*build_child.children[0]);

  op.children[0]->build_pipelines(current, meta_pipeline);

  switch (op.type) {
    case SiriusPhysicalOperatorType::POSITIONAL_JOIN:
      throw not_implemented_exception("POSITIONAL_JOIN is not implemented yet");
    case SiriusPhysicalOperatorType::CROSS_PRODUCT:
      throw not_implemented_exception("CROSS_PRODUCT is not implemented yet");
    default: break;
  }
}

void sirius_physical_hash_join::build_pipelines(pipeline::sirius_pipeline& current,
                                                pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // is_sink() is true iff the tree parent is a PARTITION (nested-join case); otherwise HJ
  // contributes to the downstream chain's pipeline as its source.
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

  // Both sides feed HJ through plan-gen CONCAT wraps. Create a child meta per side (each
  // a CONCAT-sink single-op pipeline) and recurse past the CONCAT so it doesn't
  // redundantly create its own meta.
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

build_probe_decision select_build_probe_action(std::vector<build_probe_slot_view> const& slots)
{
  if (slots.empty()) { return {build_probe_action::none, std::nullopt}; }

  // 1. Prefer starting a build: the first NOT_BUILT partition that has both its (concat-folded)
  //    build batch and a probe batch. The scheduling task builds that partition's hash table and
  //    probes its first batch together.
  for (std::size_t p = 0; p < slots.size(); ++p) {
    auto const& s = slots[p];
    if (s.state == BUILD_HASH_TABLE_STATE::NOT_BUILT && s.has_build_batch && s.has_probe_batch) {
      return {build_probe_action::schedule_build, p};
    }
  }
  // 2. Otherwise probe an already-built partition that has probe data waiting.
  for (std::size_t p = 0; p < slots.size(); ++p) {
    auto const& s = slots[p];
    if (s.state == BUILD_HASH_TABLE_STATE::BUILT && s.has_probe_batch) {
      return {build_probe_action::schedule_probe, p};
    }
  }
  // 3. No schedulable work. If any partition still lacks its build batch, wait on the build
  // producer
  //    so builds can start; otherwise every partition is building or draining probe input. Only
  //    when all partitions are torn down is the operator truly finished. These actions name no
  //    partition (the caller waits on the port's single upstream producer, shared by all
  //    partitions).
  bool all_destroyed = true;
  for (auto const& s : slots) {
    if (s.state != BUILD_HASH_TABLE_STATE::DESTROYED) { all_destroyed = false; }
    if (s.state == BUILD_HASH_TABLE_STATE::NOT_BUILT && !s.has_build_batch) {
      return {build_probe_action::wait_for_build, std::nullopt};
    }
  }
  if (all_destroyed) { return {build_probe_action::none, std::nullopt}; }
  return {build_probe_action::wait_for_probe, std::nullopt};
}

//===----------------------------------------------------------------------===//
// STANDARD / MIXED_JOIN partial-barrier scheduling (pure helpers, unit-tested in
// test/cpp/operator/test_cross_schedule.cpp).
//===----------------------------------------------------------------------===//

std::vector<cross_schedule_discard> collect_cross_schedule_discards(
  std::vector<partition_cross_schedule>& cross, bool probe_finished, bool build_finished)
{
  std::vector<cross_schedule_discard> discards;
  for (std::size_t p = 0; p < cross.size(); ++p) {
    auto& c = cross[p];
    // A probe batch is done once no more build batches can arrive (build finished) AND it has been
    // paired with every known build batch. Requires >= 1 build batch: the empty-opposite case is
    // skipped, matching the legacy grid (which likewise never popped a batch it could not pair).
    if (build_finished && !c.build_ids.empty()) {
      for (std::size_t i = 0; i < c.probe_ids.size(); ++i) {
        uint64_t const id = c.probe_ids[i];
        if (c.probe_popped.count(id)) { continue; }
        if (c.probe_paired_count[i] == c.build_ids.size()) {
          c.probe_popped.insert(id);
          discards.push_back({p, /*is_build=*/false, id});
        }
      }
    }
    if (probe_finished && !c.probe_ids.empty()) {
      for (std::size_t j = 0; j < c.build_ids.size(); ++j) {
        uint64_t const id = c.build_ids[j];
        if (c.build_popped.count(id)) { continue; }
        if (c.build_paired_count[j] == c.probe_ids.size()) {
          c.build_popped.insert(id);
          discards.push_back({p, /*is_build=*/true, id});
        }
      }
    }
  }
  return discards;
}

cross_schedule_pair next_cross_schedule_pair(std::vector<partition_cross_schedule>& cross,
                                             bool probe_finished,
                                             bool build_finished)
{
  for (std::size_t p = 0; p < cross.size(); ++p) {
    auto& c                 = cross[p];
    std::size_t const total = c.probe_ids.size() * c.build_ids.size();
    if (c.scheduled_pairs.size() >= total) { continue; }  // fully scheduled (or a side is empty)
    for (std::size_t i = 0; i < c.probe_ids.size(); ++i) {
      for (std::size_t j = 0; j < c.build_ids.size(); ++j) {
        if (c.scheduled_pairs.insert(encode_cross_pair(i, j)).second) {
          ++c.probe_paired_count[i];
          ++c.build_paired_count[j];
          return {cross_schedule_kind::emit_pair, p, i, j};
        }
      }
    }
  }
  // No schedulable pair now. If a producer can still deliver batches that would create new pairs,
  // wait on it; otherwise the operator is done.
  if (!build_finished) { return {cross_schedule_kind::wait_build, 0, 0, 0}; }
  if (!probe_finished) { return {cross_schedule_kind::wait_probe, 0, 0, 0}; }
  return {cross_schedule_kind::done, 0, 0, 0};
}

bool has_pending_cross_orphan(std::vector<partition_cross_schedule> const& cross,
                              bool probe_finished,
                              bool build_finished)
{
  // Only a terminal state: an empty opposite side is only known-final once both producers finished.
  if (!(probe_finished && build_finished)) { return false; }
  for (auto const& c : cross) {
    // Surviving probe batches whose build side is empty.
    if (c.build_ids.empty()) {
      for (uint64_t id : c.probe_ids) {
        if (!c.probe_popped.count(id)) { return true; }
      }
    }
    // Surviving build batches whose probe side is empty.
    if (c.probe_ids.empty()) {
      for (uint64_t id : c.build_ids) {
        if (!c.build_popped.count(id)) { return true; }
      }
    }
  }
  return false;
}

cross_schedule_orphan next_cross_schedule_orphan(std::vector<partition_cross_schedule>& cross,
                                                 bool probe_finished,
                                                 bool build_finished)
{
  if (!(probe_finished && build_finished)) { return {}; }
  for (std::size_t p = 0; p < cross.size(); ++p) {
    auto& c = cross[p];
    if (c.build_ids.empty()) {  // empty build → each surviving probe batch is an orphan
      for (uint64_t id : c.probe_ids) {
        if (c.probe_popped.insert(id).second) {
          return {/*found=*/true, p, /*present_is_build=*/false, id};
        }
      }
    }
    if (c.probe_ids.empty()) {  // empty probe → each surviving build batch is an orphan
      for (uint64_t id : c.build_ids) {
        if (c.build_popped.insert(id).second) {
          return {/*found=*/true, p, /*present_is_build=*/true, id};
        }
      }
    }
  }
  return {};
}

cross_schedule_kind peek_cross_schedule_kind(std::vector<partition_cross_schedule> const& cross,
                                             bool probe_finished,
                                             bool build_finished)
{
  for (auto const& c : cross) {
    if (c.scheduled_pairs.size() < c.probe_ids.size() * c.build_ids.size()) {
      return cross_schedule_kind::emit_pair;
    }
  }
  // Terminal empty-opposite batches still need a single-side task before the operator is done.
  if (has_pending_cross_orphan(cross, probe_finished, build_finished)) {
    return cross_schedule_kind::emit_pair;
  }
  if (!build_finished) { return cross_schedule_kind::wait_build; }
  if (!probe_finished) { return cross_schedule_kind::wait_probe; }
  return cross_schedule_kind::done;
}

partition_strategy compute_hash_join_partition_strategy(uint64_t total_bytes,
                                                        bool is_build_side,
                                                        bool build_foldable,
                                                        int num_gpus,
                                                        uint64_t hash_partition_bytes,
                                                        uint64_t max_build_hash_table_bytes,
                                                        uint64_t max_broadcast_join_size,
                                                        duckdb::JoinType join_type,
                                                        HASH_JOIN_MODE join_mode,
                                                        double estimated_probe_to_build_ratio)
{
  // Invariant: num_gpus defaults to 1 and is only ever set to a hardware GPU count >= 1. A value
  // < 1 is a programming error (it makes the per-partition division below ill-defined).
  if (num_gpus < 1) {
    throw std::invalid_argument("compute_hash_join_partition_strategy: num_gpus (" +
                                std::to_string(num_gpus) + ") must be >= 1");
  }

  int const natural = natural_num_partitions(total_bytes, hash_partition_bytes, num_gpus);

  // Only the build side can drive broadcast / BUILD_PROBE. Right-family joins are probe-driven
  // (probe partition sizes the join), so they always take the plain STANDARD natural count.
  if (!is_build_side) { return {natural, /*broadcast=*/false, /*build_probe=*/false}; }

  bool const is_mark         = join_type == duckdb::JoinType::MARK;
  bool const is_right_family = join_type == duckdb::JoinType::RIGHT ||
                               join_type == duckdb::JoinType::RIGHT_SEMI ||
                               join_type == duckdb::JoinType::RIGHT_ANTI;
  bool const is_mixed      = join_mode == HASH_JOIN_MODE::MIXED_JOIN;
  bool const is_full_outer = join_type == duckdb::JoinType::OUTER;
  uint64_t const small     = partition_small_table_bytes(num_gpus);

  // Broadcast candidacy. MARK multi-GPU is forced broadcast (build_has_null must be globally
  // consistent); otherwise a build is a candidate when it is below the small-table threshold, OR
  // below max_broadcast_join_size while the probe side is large relative to the build (replicating
  // the build avoids shuffling a much larger probe across GPUs).
  bool const broadcast_candidate =
    is_mark ? num_gpus > 1
            : (total_bytes < small ||
               (total_bytes < max_broadcast_join_size &&
                estimated_probe_to_build_ratio >= static_cast<double>(num_gpus) * 1.25));

  // BUILD_PROBE eligibility. MARK/SEMI/ANTI are eligible (persistent filtered_join built on the
  // right, reused across streamed left probe batches); RIGHT_SEMI/RIGHT_ANTI/RIGHT and full OUTER
  // emit build-side output and stay on the STANDARD path.
  //
  // Evaluated at the count BUILD_PROBE would run with — one build table per GPU, capped at
  // `natural` — not at the natural count: `hash_partition_bytes` targets a streaming batch size
  // and must not veto `max_build_hash_table_bytes`, which is what sizes the folded hash table.
  // A broadcast join charges the FULL build to every GPU; a hash-partitioned build charges the
  // per-GPU average.
  int const build_probe_partitions = std::max(1, std::min(natural, num_gpus));
  uint64_t const per_gpu_build_bytes =
    broadcast_candidate ? total_bytes : total_bytes / static_cast<uint64_t>(build_probe_partitions);
  bool build_probe = per_gpu_build_bytes < max_build_hash_table_bytes && build_foldable &&
                     !is_right_family && !is_mixed && !is_full_outer;

  // A MARK join must always run in BUILD_PROBE mode. It needs the entire build side resident to
  // compute the global build_has_null sentinel and the per-probe-row marks, and on multi-GPU it is
  // always broadcast (build replicated to every GPU). Force BUILD_PROBE on even when the build
  // exceeds the hash-table budget. MARK is never right-family/full-outer, and MARK + MIXED is
  // rejected at construction.
  if (is_mark) { build_probe = true; }

  bool const broadcast = broadcast_candidate && build_probe;

  // BUILD_PROBE runs at the count its eligibility was measured at; MARK single-GPU is clamped to
  // one partition; broadcast takes num_gpus; everything else the natural count.
  int const num_partitions = broadcast                    ? num_gpus
                             : build_probe                ? build_probe_partitions
                             : (is_mark && num_gpus <= 1) ? 1
                                                          : natural;
  return {num_partitions, broadcast, build_probe};
}

partition_strategy sirius_physical_hash_join::get_partition_strategy(
  const partition_sizing_input& in)
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  const std::size_t probe_card_est = children.size() > 0 ? children[0]->estimated_cardinality : 0;
  std::size_t build_card_est       = children.size() > 1 ? children[1]->estimated_cardinality : 0;
  build_card_est                   = std::max(build_card_est, 1UL);
  const double estimated_probe_to_build_ratio =
    static_cast<double>(probe_card_est) / build_card_est;
  auto const strategy = compute_hash_join_partition_strategy(in.total_bytes,
                                                             in.is_build_side,
                                                             in.build_foldable,
                                                             _num_gpus,
                                                             _hash_partition_bytes,
                                                             _max_build_hash_table_bytes,
                                                             _max_broadcast_join_size,
                                                             join_type,
                                                             _join_mode,
                                                             estimated_probe_to_build_ratio);

  if (join_type == duckdb::JoinType::MARK && _num_gpus > 1 && in.is_build_side &&
      in.total_bytes >= partition_small_table_bytes(_num_gpus)) {
    SIRIUS_LOG_WARN(
      "sirius_physical_hash_join id {}: forcing broadcast for MARK join with build side {} bytes "
      "(exceeds standard broadcast limit of {} bytes)",
      this->get_operator_id(),
      in.total_bytes,
      partition_small_table_bytes(_num_gpus));
  }

  if (strategy.build_probe) {
    _join_mode = HASH_JOIN_MODE::BUILD_PROBE;
    // One hash-table slot per partition. Elements are non-movable (atomic build_state), so build a
    // fresh right-sized vector and move-assign it (steals the buffer, no element moves).
    _partition_build_states =
      std::vector<per_partition_build_state>(static_cast<std::size_t>(strategy.num_partitions));
  }
  if (strategy.broadcast) { _broadcast = true; }

  // Pre-size the build and probe input repositories so every partition slot exists before batches
  // arrive. Gated on > 1 (a single partition needs no growth).
  if (strategy.num_partitions > 1) {
    auto const parts = static_cast<std::size_t>(strategy.num_partitions);
    for (std::string_view const port_id :
         {std::string_view{"build"}, std::string_view{"default"}}) {
      auto* p = get_port(port_id);
      if (p != nullptr && p->repo != nullptr && parts > p->repo->num_partitions()) {
        p->repo->set_num_partitions(parts);
      }
    }
  }

  const char* join_mode_str = _join_mode == HASH_JOIN_MODE::BUILD_PROBE  ? "BUILD_PROBE"
                              : _join_mode == HASH_JOIN_MODE::MIXED_JOIN ? "MIXED_JOIN"
                                                                         : "STANDARD";

  SIRIUS_LOG_DEBUG(
    "sirius_physical_hash_join id {} partition strategy: {} partitions ({} GPUs), build side {} "
    "bytes. Join Type: {}. Join Mode: {} {}. build_card_est {} probe_card_est {}",
    this->get_operator_id(),
    strategy.num_partitions,
    _num_gpus,
    in.total_bytes,
    duckdb::JoinTypeToString(join_type),
    join_mode_str,
    (strategy.broadcast ? " [broadcast]" : ""),
    build_card_est,
    probe_card_est);
  return strategy;
}

bool sirius_physical_hash_join::is_build_probe_mode()
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  return _join_mode == HASH_JOIN_MODE::BUILD_PROBE;
}

bool sirius_physical_hash_join::publishes_dynamic_filters() const
{
  // Replica restriction completes before execution; the plan is immutable afterward.
  return _dynamic_filter_plan.enabled();
}

void sirius_physical_hash_join::set_build_arrives_whole(bool arrives_whole)
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  _build_arrives_whole = arrives_whole;
}

std::vector<build_probe_slot_view> sirius_physical_hash_join::snapshot_build_probe_slots()
{
  auto* build_port = get_port("build");
  auto* probe_port = get_port("default");
  if (!build_port || !probe_port) {
    throw std::runtime_error(
      "In sirius_physical_hash_join:snapshot_build_probe_slots: missing expected ports in "
      "operator " +
      std::to_string(this->get_operator_id()));
  }
  std::vector<build_probe_slot_view> slots(_partition_build_states.size());
  for (std::size_t p = 0; p < _partition_build_states.size(); ++p) {
    slots[p].state = _partition_build_states[p].build_state.load(std::memory_order_acquire);
    slots[p].has_build_batch = build_port->repo->size(p) > 0;
    slots[p].has_probe_batch = probe_port->repo->size(p) > 0;
  }
  return slots;
}

std::vector<std::size_t> broadcast_slots_to_discard(std::vector<build_probe_slot_view> const& slots,
                                                    bool probe_finished)
{
  std::vector<std::size_t> to_discard;
  if (!probe_finished) { return to_discard; }
  for (std::size_t p = 0; p < slots.size(); ++p) {
    auto const& s = slots[p];
    // NOT_BUILT means the slot was never scheduled (so its replicated build batch is still in the
    // repo); no probe batch with the probe side finished means none is coming. A BUILT slot already
    // consumed its build batch, and a slot with probe data will still be built — leave those.
    if (s.state == BUILD_HASH_TABLE_STATE::NOT_BUILT && s.has_build_batch && !s.has_probe_batch) {
      to_discard.push_back(p);
    }
  }
  return to_discard;
}

void sirius_physical_hash_join::discard_build_only_slots_if_probe_complete()
{
  if (!_broadcast) { return; }
  auto* build_port = get_port("build");
  auto* probe_port = get_port("default");
  if (!build_port || !probe_port) { return; }
  // Only once the probe upstream is finished do we know no further probe data can arrive for any
  // slot. Broadcast replicates the build table to every slot, but the probe side is unpartitioned,
  // so slots on GPUs that saw no probe rows get build data that will never be probed.
  bool const probe_finished =
    probe_port->src_pipeline && probe_port->src_pipeline->is_pipeline_finished();

  for (std::size_t p : broadcast_slots_to_discard(snapshot_build_probe_slots(), probe_finished)) {
    // Build-only slot with no probe and none coming: drop its replicated build batch(es), freeing
    // each on the GPU it was folded onto (rmm requires the owning device to be current).
    while (auto batch = build_port->repo->pop_next_data_batch(p)) {
      int device_id = -1;
      {
        auto ro = batch->to_read_only();
        if (auto* ms = ro.get_memory_space(); ms != nullptr) { device_id = ms->get_device_id(); }
      }
      std::optional<rmm::cuda_set_device_raii> device_guard;
      if (device_id >= 0) { device_guard.emplace(rmm::cuda_device_id{device_id}); }
      batch.reset();
    }
    _partition_build_states[p].build_state.store(BUILD_HASH_TABLE_STATE::DESTROYED,
                                                 std::memory_order_release);
    SIRIUS_LOG_DEBUG(
      "sirius_physical_hash_join id {}: broadcast discard of build-only slot {} (probe complete)",
      this->get_operator_id(),
      p);
  }
}

std::optional<task_creation_hint> sirius_physical_hash_join::get_next_task_hint()
{
  std::lock_guard<std::mutex> lg(op_state_mutex);
  auto* build_port = get_port("build");
  auto* probe_port = get_port("default");
  if (!build_port || !probe_port) {
    throw std::runtime_error(
      "In sirius_physical_hash_join:get_next_task_hint: missing expected ports in operator " +
      std::to_string(this->get_operator_id()));
  }
  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    // Each partition owns one hash table and runs its own build-then-probe sequence; those
    // sequences interleave (a built partition probes on its GPU while another still builds on a
    // different GPU). Pick the next action from a per-partition snapshot.

    // Broadcast mode: reclaim slots that will never be probed before deciding the next action, so
    // the operator can reach completion instead of waiting forever on their absent probe data.
    discard_build_only_slots_if_probe_complete();
    auto const decision = select_build_probe_action(snapshot_build_probe_slots());
    switch (decision.action) {
      case build_probe_action::schedule_build:
        // Claim this partition's slot so exactly one build task is issued for it. The paired
        // get_next_task_input_data_for_build_probe scans for the SCHEDULING slot and advances it.
        _partition_build_states[decision.partition.value()].build_state.store(
          BUILD_HASH_TABLE_STATE::SCHEDULING, std::memory_order_release);
        return task_creation_hint{TaskCreationHint::READY, this};
      case build_probe_action::schedule_probe:
        return task_creation_hint{TaskCreationHint::READY, this};
      case build_probe_action::wait_for_build: {
        auto* producer = &build_port->src_pipeline->get_operators()[0].get();
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
      }
      case build_probe_action::wait_for_probe: {
        auto* producer = &probe_port->src_pipeline->get_operators()[0].get();
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
      }
      case build_probe_action::none:
      default:
        // All partitions are torn down: the operator is complete.
        return std::nullopt;
    }
  } else {
    // STANDARD / MIXED_JOIN partial barrier: schedule per-partition build x probe pairs as batches
    // arrive on either side, rather than waiting (via the base FULL-barrier hint) for both upstream
    // pipelines to finish. refresh_cross_schedule also frees fully-consumed batches, so completion
    // (all_ports_empty) converges once both producers finish and every pair has been scheduled.
    auto const finished = refresh_cross_schedule();
    switch (peek_cross_schedule_kind(_cross, finished.first, finished.second)) {
      case cross_schedule_kind::emit_pair: return task_creation_hint{TaskCreationHint::READY, this};
      case cross_schedule_kind::wait_build: {
        auto* producer = &build_port->src_pipeline->get_operators()[0].get();
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
      }
      case cross_schedule_kind::wait_probe: {
        auto* producer = &probe_port->src_pipeline->get_operators()[0].get();
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
      }
      case cross_schedule_kind::done:
      default: return std::nullopt;
    }
  }
}

std::unique_ptr<operator_data> sirius_physical_hash_join::get_next_task_input_data_for_build_probe()
{
  auto* build_port = get_port("build");
  auto* probe_port = get_port("default");
  if (!build_port || !probe_port) {
    throw std::runtime_error(
      "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: missing expected "
      "ports in operator " +
      std::to_string(this->get_operator_id()));
  }

  // How a partition's tasks are tagged for GPU routing (task_creator uses tag % num_gpus):
  //  - Multiple partitions: tag with the real partition index p, so partitions spread one-per-GPU
  //    and execute() can select the matching hash-table slot from the tag.
  //  - Single partition: tag with operator_id, preserving the historical routing where several
  //    small single-partition BUILD_PROBE joins in one query spread across GPUs instead of all
  //    pinning to GPU 0. execute() maps the lone partition back to slot 0.
  auto const partition_tag = [this](std::size_t p) -> std::size_t {
    return _partition_build_states.size() == 1 ? this->get_operator_id() : p;
  };

  // Prefer a partition awaiting its build (SCHEDULING, claimed by get_next_task_hint): issue a task
  // carrying that partition's single folded build batch plus its first probe batch. Concurrent
  // input-data calls each claim a distinct SCHEDULING slot because we advance it to SCHEDULED here
  // under op_state_mutex.
  for (std::size_t p = 0; p < _partition_build_states.size(); ++p) {
    if (_partition_build_states[p].build_state.load(std::memory_order_acquire) !=
        BUILD_HASH_TABLE_STATE::SCHEDULING) {
      continue;
    }
    if (build_port->repo->size(p) != 1) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: expected exactly 1 "
        "(concat-folded) build batch for partition " +
        std::to_string(p) + " in operator " + std::to_string(this->get_operator_id()));
    }
    std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
    input_batch.push_back(probe_port->repo->pop_next_data_batch(p));
    input_batch.push_back(build_port->repo->pop_next_data_batch(p));
    _partition_build_states[p].build_state.store(BUILD_HASH_TABLE_STATE::SCHEDULED,
                                                 std::memory_order_release);
    // Every task of partition p (this build+first-probe and all later probe-only tasks) shares the
    // same tag, so they land on the same GPU as p's hash table.
    return std::make_unique<partitioned_operator_data>(std::move(input_batch), partition_tag(p));
  }

  // Otherwise issue a probe-only task for a built partition that still has probe data.
  for (std::size_t p = 0; p < _partition_build_states.size(); ++p) {
    if (_partition_build_states[p].build_state.load(std::memory_order_acquire) !=
          BUILD_HASH_TABLE_STATE::BUILT ||
        probe_port->repo->size(p) == 0) {
      continue;
    }
    std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
    auto batch = probe_port->repo->pop_next_data_batch(p);
    if (batch) {
      input_batch.push_back(std::move(batch));
    } else {
      SIRIUS_LOG_WARN(
        "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: expected to pop a "
        "probe batch for partition {} but got none in operator {}",
        p,
        this->get_operator_id());
    }
    return std::make_unique<partitioned_operator_data>(std::move(input_batch), partition_tag(p));
  }

  // No SCHEDULING slot and no BUILT slot with probe data. This happens when a hint's READY raced
  // ahead of another task draining the same probe data; there is simply nothing to issue now.
  SIRIUS_LOG_WARN(
    "In sirius_physical_hash_join:get_next_task_input_data_for_build_probe: no schedulable "
    "partition (build/probe already drained) in operator {}",
    this->get_operator_id());
  return nullptr;
}

std::pair<bool, bool> sirius_physical_hash_join::refresh_cross_schedule()
{
  auto* probe_port = get_port("default");
  auto* build_port = get_port("build");
  if (probe_port->repo->num_partitions() != build_port->repo->num_partitions()) {
    throw std::runtime_error(
      "In sirius_physical_hash_join:refresh_cross_schedule: number of partitions for probe and "
      "build ports must match in operator " +
      std::to_string(this->get_operator_id()));
  }
  std::size_t const num_partitions = probe_port->repo->num_partitions();
  if (_cross.size() < num_partitions) { _cross.resize(num_partitions); }

  bool const probe_finished =
    probe_port->src_pipeline && probe_port->src_pipeline->is_pipeline_finished();
  bool const build_finished =
    build_port->src_pipeline && build_port->src_pipeline->is_pipeline_finished();

  // Re-poll both ports and merge any newly-arrived batch IDs (first-seen order preserved; the
  // *_seen sets both dedup and keep already-popped IDs from being re-added).
  for (std::size_t p = 0; p < num_partitions; ++p) {
    auto& c = _cross[p];
    for (uint64_t id : probe_port->repo->get_batch_ids(p)) {
      if (c.probe_id_seen.insert(id).second) {
        c.probe_ids.push_back(id);
        c.probe_paired_count.push_back(0);
      }
    }
    for (uint64_t id : build_port->repo->get_batch_ids(p)) {
      if (c.build_id_seen.insert(id).second) {
        c.build_ids.push_back(id);
        c.build_paired_count.push_back(0);
      }
    }
  }

  // Non-INNER whole-side invariant: whichever side must be seen whole for this join type must stay
  // a single concat-folded batch once its producer is finished. A regression in
  // sirius_physical_concat that stopped folding would otherwise silently corrupt results, so fail
  // loudly instead. MARK must never reach this path (it is forced to BUILD_PROBE mode).
  if (join_type == duckdb::JoinType::MARK) {
    throw std::runtime_error(
      "In sirius_physical_hash_join:refresh_cross_schedule: MARK join must run in BUILD_PROBE "
      "mode, "
      "not STANDARD/MIXED, in operator " +
      std::to_string(this->get_operator_id()));
  }
  bool const check_build_whole =
    build_finished && (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::SEMI ||
                       join_type == duckdb::JoinType::ANTI || join_type == duckdb::JoinType::OUTER);
  bool const check_probe_whole =
    probe_finished && is_right_family();  // RIGHT/RIGHT_SEMI/RIGHT_ANTI
  bool const check_probe_whole_outer = probe_finished && join_type == duckdb::JoinType::OUTER;
  for (std::size_t p = 0; p < num_partitions; ++p) {
    if (check_build_whole && _cross[p].build_ids.size() > 1) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:refresh_cross_schedule: join type " +
        duckdb::JoinTypeToString(join_type) +
        " requires a single concat-folded build batch, but "
        "partition " +
        std::to_string(p) + " has " + std::to_string(_cross[p].build_ids.size()) +
        " build batches in operator " + std::to_string(this->get_operator_id()));
    }
    if ((check_probe_whole || check_probe_whole_outer) && _cross[p].probe_ids.size() > 1) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:refresh_cross_schedule: join type " +
        duckdb::JoinTypeToString(join_type) +
        " requires a single concat-folded probe batch, but "
        "partition " +
        std::to_string(p) + " has " + std::to_string(_cross[p].probe_ids.size()) +
        " probe batches in operator " + std::to_string(this->get_operator_id()));
    }
  }

  // Free every fully-consumed batch from its repository. Borrowers of a batch hold their own live
  // shared_ptr across the pop, so this only drops the repo's owning reference.
  if (probe_finished || build_finished) {
    for (auto const& d : collect_cross_schedule_discards(_cross, probe_finished, build_finished)) {
      auto* port = d.is_build ? build_port : probe_port;
      port->repo->pop_data_batch_by_id(d.batch_id, d.partition);
    }
  }
  return {probe_finished, build_finished};
}

std::unique_ptr<operator_data> sirius_physical_hash_join::get_next_task_input_data()
{
  // Hold the mutex for the entire operation to prevent concurrent pop/get races. A pop on one
  // thread must not remove a batch that another thread's get expects to find.
  std::lock_guard<std::mutex> lg(op_state_mutex);

  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    return get_next_task_input_data_for_build_probe();
  }

  // STANDARD / MIXED_JOIN partial barrier: batches on each side arrive progressively. Re-poll both
  // ports, free fully-consumed batches, then schedule the next per-partition (probe, build) pair.
  auto* probe_port = get_port("default");
  auto* build_port = get_port("build");
  if (!probe_port || !build_port) {
    throw std::runtime_error(
      "In sirius_physical_hash_join:get_next_task_input_data: missing expected ports in operator " +
      std::to_string(this->get_operator_id()));
  }

  auto const finished = refresh_cross_schedule();
  auto const step     = next_cross_schedule_pair(_cross, finished.first, finished.second);
  if (step.kind == cross_schedule_kind::emit_pair) {
    auto& c                 = _cross[step.partition];
    uint64_t const probe_id = c.probe_ids[step.probe_idx];
    uint64_t const build_id = c.build_ids[step.build_idx];

    // Always borrow (get, not pop): freeing is handled centrally by refresh_cross_schedule's
    // discard sweep once a batch has been paired with every batch of a finished opposite side.
    auto probe_batch = probe_port->repo->get_data_batch_by_id(probe_id, step.partition);
    auto build_batch = build_port->repo->get_data_batch_by_id(build_id, step.partition);
    if (!probe_batch || !build_batch) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:get_next_task_input_data: expected resident probe and build "
        "batches for a newly-scheduled pair in partition " +
        std::to_string(step.partition) + " of operator " + std::to_string(this->get_operator_id()));
    }

    std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
    input_batch.reserve(2);
    input_batch.push_back(std::move(probe_batch));  // [0] = probe / "default" / left
    input_batch.push_back(std::move(build_batch));  // [1] = build / "build" / right

    return std::make_unique<partitioned_operator_data>(std::move(input_batch), step.partition);
  }

  // No normal pair. If both producers finished, a partition may have batches on one side and an
  // empty opposite side (e.g. a join against an empty table): those survivors are never paired.
  // Emit each survivor as an ordinary two-batch task by SYNTHESIZING the empty opposite batch here,
  // so execute() runs the normal join dispatch unchanged — cuDF NULL-pads the survivor for LEFT /
  // RIGHT / OUTER / ANTI and yields zero rows for INNER / SEMI. Popping the survivor also drains
  // the repo so the pipeline can complete.
  auto const orphan = next_cross_schedule_orphan(_cross, finished.first, finished.second);
  if (orphan.found) {
    auto* present_port = orphan.present_is_build ? build_port : probe_port;
    auto present_batch =
      present_port->repo->pop_data_batch_by_id(orphan.batch_id, orphan.partition);
    if (!present_batch) { return nullptr; }  // already drained by a concurrent caller

    // Synthesize the empty opposite side from the absent child's output schema (children[0] =
    // probe/left, children[1] = build/right), on the surviving batch's device. The absent child's
    // physical sidecar, when it has one, is the carrier schema its batches would have arrived with,
    // so synthesizing from it keeps every output batch of this join agreeing with the carriers the
    // join's own sidecar advertises for that side. The batch's memory space is only reachable
    // through a read-only accessor; take one transiently (shared lock) and release it before
    // handing the idle batch to the task.
    cucascade::memory::memory_space* ms = nullptr;
    {
      auto present_ro = present_batch->to_read_only();
      ms              = present_ro.get_memory_space();
    }
    if (!ms) {
      throw std::runtime_error(
        "In sirius_physical_hash_join:get_next_task_input_data: surviving orphan batch has no "
        "memory space in operator " +
        std::to_string(this->get_operator_id()));
    }
    auto const& absent = orphan.present_is_build ? *children[0]   // absent probe
                                                 : *children[1];  // absent build
    rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{ms->get_device_id()}};
    auto empty_table = absent.has_physical_overrides()
                         ? sirius::make_empty_table(absent.get_physical_types())
                         : sirius::make_empty_table(absent.get_types());
    auto empty_batch =
      make_data_batch(std::move(empty_table), *ms, cudf::get_default_stream(), batch_telemetry());

    std::vector<std::shared_ptr<cucascade::data_batch>> input_batch;
    input_batch.reserve(2);
    if (orphan.present_is_build) {
      input_batch.push_back(std::move(empty_batch));    // [0] = empty probe / "default" / left
      input_batch.push_back(std::move(present_batch));  // [1] = build / "build" / right
    } else {
      input_batch.push_back(std::move(present_batch));  // [0] = probe / "default" / left
      input_batch.push_back(std::move(empty_batch));    // [1] = empty build / "build" / right
    }
    return std::make_unique<partitioned_operator_data>(std::move(input_batch), orphan.partition);
  }

  return nullptr;
}

/// Result of prepare_join_keys for a single join side: the key table view and any cast columns
/// that must remain alive.
struct join_side_keys_result {
  // Owned cast columns - kept alive so the table view referencing them remains valid
  std::vector<std::unique_ptr<cudf::column>> owned_cast_columns;
  cudf::table_view keys;
  // Storage for column views used to build the table_view (must outlive the table_view)
  std::vector<cudf::column_view> key_views;
};

/// Build the key table view for one side of the join.
/// If cast_necessary is false, this simply selects the key columns from the input batch.
/// If cast_necessary is true, each key column that requires a cast is cast to its target type
/// via cudf::cast before being included in the key table.
/// @param is_left_side  If true, uses cast_left/left_target_type from key_casts; otherwise uses
///                      cast_right/right_target_type.
static join_side_keys_result prepare_join_keys(
  const ::cucascade::read_only_data_batch& input_batch,
  const std::vector<cudf::size_type>& key_col_indices,
  bool cast_necessary,
  const std::vector<sirius_physical_hash_join::key_cast_info>& key_casts,
  bool is_left_side,
  rmm::cuda_stream_view stream)
{
  join_side_keys_result result;

  cudf::table_view table = get_cudf_table_view(input_batch);

  if (!cast_necessary) {
    // INVARIANT: every entry in key_col_indices must address a column in
    // `table`. PR #732 closed the only known violator — DuckDB's
    // LATE_MATERIALIZATION optimizer was rewriting `ORDER BY ... LIMIT N`
    // into a self-RIGHT_SEMI_JOIN keyed on parquet virtual columns
    // (file_index / file_row_number) that Sirius's scan path silently
    // drops, leaving the join with key_col_indices entries pointing past
    // the physical batch. Disabling that pass in
    // src/transparent/sirius_optimizer_extension.cpp removed the bad
    // emitter. If you hit this throw, a new emitter has been introduced —
    // fix it at the source rather than reintroducing the historical
    // silent filter (see PR #732 comment 3242605041 for the prior shape).
    auto const num_cols = table.num_columns();
    for (auto idx : key_col_indices) {
      if (idx >= num_cols) {
        throw std::out_of_range("prepare_join_keys: key_col_indices entry " + std::to_string(idx) +
                                " is >= input table column count " + std::to_string(num_cols) +
                                " (is_left_side=" + (is_left_side ? "true" : "false") +
                                "). The upstream emitter wired a join key that does not exist in "
                                "the physical batch — fix the emitter, do not paper over it here.");
      }
    }
    result.keys = table.select(key_col_indices);
    return result;
  }

  // Slow path: iterate over key columns and cast where needed
  for (size_t i = 0; i < key_col_indices.size(); i++) {
    const auto& cast_info        = key_casts[i];
    const cudf::column_view& col = table.column(key_col_indices[i]);
    bool needs_cast              = is_left_side ? cast_info.cast_left : cast_info.cast_right;
    cudf::data_type target_type =
      is_left_side ? cast_info.left_target_type : cast_info.right_target_type;

    if (needs_cast) {
      auto cast_col = sirius::cast_through_rep(col, target_type, stream);
      result.key_views.push_back(cast_col->view());
      result.owned_cast_columns.push_back(std::move(cast_col));
    } else {
      result.key_views.push_back(col);
    }
  }

  result.keys = cudf::table_view(result.key_views);
  return result;
}

/// Gather output columns from both sides of a join using row index vectors, then assemble the
/// result into an operator_data. Handles collect/oob policy selection based on join type.
/// @param left_indices   Row indices into left_full; may be null if the left side is not collected.
/// @param right_indices  Row indices into right_full; may be null if the right side is not
///                       collected.
/// @param memory_space   Memory space of the input batch used to tag the output data batch.
static std::unique_ptr<operator_data> gather_join_output(
  duckdb::JoinType join_type,
  const cudf::table_view& left_full,
  const cudf::table_view& right_full,
  std::vector<cudf::size_type> const& lhs_col_idxs,
  std::vector<cudf::size_type> const& rhs_col_idxs,
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices,
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_indices,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info = {})
{
  bool collect_left =
    (join_type != duckdb::JoinType::RIGHT_SEMI && join_type != duckdb::JoinType::RIGHT_ANTI);
  bool collect_right = (join_type != duckdb::JoinType::SEMI && join_type != duckdb::JoinType::ANTI);

  cudf::out_of_bounds_policy left_oob  = cudf::out_of_bounds_policy::DONT_CHECK;
  cudf::out_of_bounds_policy right_oob = cudf::out_of_bounds_policy::DONT_CHECK;
  if (join_type == duckdb::JoinType::LEFT || join_type == duckdb::JoinType::OUTER ||
      join_type == duckdb::JoinType::SEMI) {
    right_oob = cudf::out_of_bounds_policy::NULLIFY;
  }
  if (join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::OUTER ||
      join_type == duckdb::JoinType::RIGHT_SEMI) {
    left_oob = cudf::out_of_bounds_policy::NULLIFY;
  }

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  if (collect_left) {
    cudf::table_view left_cols_to_gather = left_full.select(lhs_col_idxs);
    cudf::column_view left_map_view(cudf::data_type(cudf::type_id::INT32),
                                    left_indices->size(),
                                    left_indices->data(),
                                    nullptr,
                                    0,
                                    0,
                                    {});
    auto left_result = cudf::gather(left_cols_to_gather, left_map_view, left_oob, stream);
    out_cols         = left_result->release();
  }
  if (collect_right) {
    cudf::table_view right_cols_to_gather = right_full.select(rhs_col_idxs);
    cudf::column_view right_map_view(cudf::data_type(cudf::type_id::INT32),
                                     right_indices->size(),
                                     right_indices->data(),
                                     nullptr,
                                     0,
                                     0,
                                     {});
    auto right_result   = cudf::gather(right_cols_to_gather, right_map_view, right_oob, stream);
    auto right_out_cols = right_result->release();
    for (auto& col : right_out_cols) {
      out_cols.push_back(std::move(col));
    }
  }

  auto output_cudf_table = std::make_unique<cudf::table>(std::move(out_cols));
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{
      make_data_batch(std::move(output_cudf_table), memory_space, stream, telemetry_info)});
}

/// Assemble output for a distinct_hash_join left_join.
/// distinct_hash_join::left_join returns only build indices (one per probe row, in probe order).
/// Left (probe) columns are copied directly; right (build) columns are gathered with NULLIFY.
static std::unique_ptr<operator_data> gather_distinct_left_join_output(
  const cudf::table_view& left_full,
  const cudf::table_view& right_full,
  std::vector<cudf::size_type> const& lhs_col_idxs,
  std::vector<cudf::size_type> const& rhs_col_idxs,
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_indices,
  cucascade::memory::memory_space& memory_space,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info = {})
{
  std::vector<std::unique_ptr<cudf::column>> out_cols;

  // Left (probe): all rows appear in order — copy selected columns directly.
  cudf::table_view left_cols = left_full.select(lhs_col_idxs);
  for (cudf::size_type i = 0; i < left_cols.num_columns(); i++) {
    out_cols.push_back(std::make_unique<cudf::column>(left_cols.column(i), stream));
  }

  // Right (build): gather using build_indices; unmatched entries are JoinNoneValue → NULLIFY.
  cudf::table_view right_cols = right_full.select(rhs_col_idxs);
  cudf::column_view right_map(cudf::data_type(cudf::type_id::INT32),
                              static_cast<cudf::size_type>(build_indices->size()),
                              build_indices->data(),
                              nullptr,
                              0,
                              0,
                              {});
  auto right_result =
    cudf::gather(right_cols, right_map, cudf::out_of_bounds_policy::NULLIFY, stream);
  for (auto& col : right_result->release()) {
    out_cols.push_back(std::move(col));
  }

  auto output_cudf_table = std::make_unique<cudf::table>(std::move(out_cols));
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{
      make_data_batch(std::move(output_cudf_table), memory_space, stream, telemetry_info)});
}

/// @brief Whether any column of @p keys carries at least one NULL value.
static bool table_has_any_null(cudf::table_view const& keys)
{
  return std::ranges::any_of(keys, [](auto const& col) { return col.null_count() > 0; });
}

/// @brief Largest build cudf::distinct_count can answer for. Hard limitation.
static constexpr cudf::size_type k_max_distinct_count_rows =
  std::numeric_limits<cudf::size_type>::max() / 2;

/// @brief Largest build the runtime distinctness test is allowed to probe at all. Heuristic
/// limitation.
static constexpr cudf::size_type k_max_distinct_probe_rows = 128 * 1024 * 1024;
static_assert(k_max_distinct_probe_rows <= k_max_distinct_count_rows);

/// @brief Number of rows with which to sample the keys cheaply for a quick refutation of
/// non-distinctness. The static set for the sample will fit in the L2 cache for recent
/// architectures at 16MB.
static constexpr cudf::size_type k_distinct_refute_sample_rows = 1024 * 1024;

/// @brief Exact runtime test that the build keys hold no duplicate rows.
/// @note Reads a count back to the host, so it synchronizes the stream.
static bool build_keys_are_distinct(cudf::table_view const& build_keys,
                                    cudf::null_equality nulls_equal,
                                    rmm::cuda_stream_view stream)
{
  auto const num_rows = build_keys.num_rows();
  if (num_rows == 1) { return true; }
  if (build_keys.num_columns() == 0 || num_rows <= 0 || num_rows > k_max_distinct_probe_rows) {
    return false;
  }
  // Guard the full uniqueness test on a sample.
  if (num_rows > 4 * k_distinct_refute_sample_rows) {
    auto const prefix = cudf::slice(build_keys, {0, k_distinct_refute_sample_rows}, stream).front();
    if (cudf::distinct_count(prefix, nulls_equal, stream) != k_distinct_refute_sample_rows) {
      return false;
    }
  }
  return cudf::distinct_count(build_keys, nulls_equal, stream) == num_rows;
}

/// @brief Thread-safe check-and-set for the join-wide _build_has_null sentinel.
///
/// Sentinel encoding: -1 = unset, 0 = false, 1 = true.
/// On first call: CAS from -1 to the new value.
/// On subsequent calls: verify the existing value matches; throw if it does not (indicates
/// inconsistent build batches, which should never happen for MARK joins that are forced to a
/// single partition or broadcast).
static void set_build_has_null(std::atomic<int>& atomic_flag, bool has_null)
{
  int const new_val = has_null ? 1 : 0;
  int expected      = -1;
  if (!atomic_flag.compare_exchange_strong(expected, new_val, std::memory_order_acq_rel)) {
    // Already set — expected now holds the actual current value.
    if (expected != new_val) {
      throw std::runtime_error(
        "sirius_physical_hash_join: build_has_null inconsistency across build batches for MARK "
        "join — this should not happen when the join is forced to a single partition or broadcast");
    }
  }
}

/// @brief the MARK join output from the semi_join matching row indices.
///
/// Copies all left output columns (all rows pass through, no gather), then creates a BOOL8 mark
/// column initialized to false and scatters true at every position in semi_indices. Finally applies
/// SQL three-valued logic: an unmatched left row is NULL (not false) when its probe key is NULL, or
/// when the build/right side contains a NULL join key.
///
/// The scattered values are already correct (true at matched rows, false elsewhere), so only a null
/// mask is added. When @p marks_are_definite (an all-null-safe MARK, matched under EQUAL) no mask
/// is added at all: a null-safe comparison is never UNKNOWN, so an unmatched row is a definite
/// FALSE. Otherwise this is the IN/EXISTS three-valued rule under UNEQUAL matching. Because a
/// NULL key never matches under UNEQUAL, a matched row always has a valid probe key, so the
/// desired validity reduces to two cases:
///   - build_has_null == true : every unmatched row is NULL, so valid == matched. The mask is the
///                              mark values themselves (cudf::bools_to_mask).
///   - build_has_null == false: valid == probe row validity (all probe key columns valid). The mask
///                              is cudf::bitmask_and over the probe keys (empty when none
///                              nullable).
///
/// @param semi_indices  Device vector of left-side row indices that matched the join condition,
///                      as returned by cuDF's semi-join. Used as the scatter map for the mark
///                      column.
/// @param left_full     Full left-side table view (all columns, all rows) before output projection.
/// @param lhs_output_col_idxs  Column indices within @p left_full to include in the output.
///                             Drives the projection of the left side.
/// @param probe_keys    Probe/left join key columns (post-cast); their per-row validity drives the
///                      NULL mark when the build side has no NULL key.
/// @param build_has_null  Whether the build/right side contains a NULL in any join key column.
///                        Ignored when @p marks_are_definite.
/// @param marks_are_definite  Every key is null-safe, so the output column gets no null mask
///                            (sirius_physical_hash_join::mark_is_null_safe()).
/// @param left_batch    The original left-side data batch; used to propagate memory space metadata
///                      to the returned operator_data.
/// @param stream        CUDA stream on which all device operations are launched.
static std::unique_ptr<operator_data> resolve_mark_join_result(
  rmm::device_uvector<cudf::size_type> const& semi_indices,
  cudf::table_view const& left_full,
  std::vector<cudf::size_type> const& lhs_output_col_idxs,
  cudf::table_view const& probe_keys,
  bool build_has_null,
  bool marks_are_definite,
  ::cucascade::read_only_data_batch const& left_batch,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info = {})
{
  cudf::table_view left_cols_to_output = left_full.select(lhs_output_col_idxs);
  auto num_left_rows                   = left_cols_to_output.num_rows();

  std::vector<std::unique_ptr<cudf::column>> mark_out_cols;
  for (cudf::size_type i = 0; i < left_cols_to_output.num_columns(); i++) {
    mark_out_cols.push_back(std::make_unique<cudf::column>(left_cols_to_output.column(i), stream));
  }

  // Create BOOL8 mark column: start all-false, scatter true at matching positions
  cudf::numeric_scalar<bool> false_scalar(false, true, stream);
  auto mark_column = cudf::make_column_from_scalar(false_scalar, num_left_rows, stream);

  if (semi_indices.size() > 0) {
    cudf::numeric_scalar<bool> true_scalar(true, true, stream);
    cudf::column_view scatter_map(cudf::data_type(cudf::type_id::INT32),
                                  static_cast<cudf::size_type>(semi_indices.size()),
                                  semi_indices.data(),
                                  nullptr,
                                  0,
                                  0,
                                  {});
    // The scatter API is a bit confusing when it says: the number of elements in first arg i.e.
    // the vector should have same number of columns in the target table. It is essentially a
    // row-scatter operation. For our use case, we have only column i.e. target mark column;
    // therefore we are good. The scalar is broadcasted to respective positions provided by the
    // scatter map.
    auto scattered = cudf::scatter({std::ref(static_cast<cudf::scalar const&>(true_scalar))},
                                   scatter_map,
                                   cudf::table_view({mark_column->view()}),
                                   stream);
    mark_column    = std::move(scattered->release()[0]);
  }

  // Under EQUAL matching every comparison is definite, so the scattered values are the whole
  // answer -- no null mask, no three-valued logic.
  if (marks_are_definite) {
    mark_out_cols.push_back(std::move(mark_column));
    auto definite_table = std::make_unique<cudf::table>(std::move(mark_out_cols));
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<::cucascade::data_batch>>{make_data_batch(
        std::move(definite_table), *left_batch.get_memory_space(), stream, telemetry_info)});
  }

  // Apply SQL three-valued logic by attaching a null mask: unmatched rows become NULL when the
  // build side has a NULL key (valid == matched) or when the probe key is NULL (valid == probe key
  // validity). See the function doc comment for the derivation.
  rmm::device_buffer null_mask;
  cudf::size_type null_count = 0;
  if (build_has_null) {
    auto [mask, count] = cudf::bools_to_mask(mark_column->view(), stream);
    null_mask          = std::move(*mask);
    null_count         = count;
  } else {
    auto [mask, count] = cudf::bitmask_and(probe_keys, stream);
    null_mask          = std::move(mask);
    null_count         = count;
  }
  if (null_count > 0) { mark_column->set_null_mask(std::move(null_mask), null_count); }

  mark_out_cols.push_back(std::move(mark_column));
  auto output_cudf_table = std::make_unique<cudf::table>(std::move(mark_out_cols));
  return std::make_unique<pipelineable_operator_data>(
    std::vector<std::shared_ptr<::cucascade::data_batch>>{make_data_batch(
      std::move(output_cudf_table), *left_batch.get_memory_space(), stream, telemetry_info)});
}

std::unique_ptr<operator_data> sirius_physical_hash_join::execute(const operator_data& input_data,
                                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_hash_join::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();

  if (is_all_inequality_join) {
    throw std::runtime_error(
      "Error sirius_physical_hash_join being asked to do all inequality join of type: " +
      duckdb::JoinTypeToString(join_type));
  }

  cudf::table_view left_full, right_full;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_indices, right_indices;

  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    // Each partition owns one hash table. The incoming batch is tagged with its partition index
    // (get_next_task_input_data_for_build_probe), which selects the per-partition slot; the
    // scheduler has already pinned this task to that partition's GPU.
    auto const* partitioned = dynamic_cast<const partitioned_operator_data*>(&input_data);
    if (!partitioned) {
      throw std::runtime_error(
        "In sirius_physical_hash_join::execute: BUILD_PROBE expects partitioned_operator_data in "
        "operator " +
        std::to_string(this->get_operator_id()));
    }
    // With a single partition the task is tagged with operator_id (for cross-join GPU spread), so
    // map any tag back to the lone slot 0; with multiple partitions the tag is the real partition
    // index and selects its slot directly.
    std::size_t const partition =
      _partition_build_states.size() == 1 ? std::size_t{0} : partitioned->get_partition_idx();
    if (partition >= _partition_build_states.size()) {
      throw std::runtime_error(
        "In sirius_physical_hash_join::execute: BUILD_PROBE partition index " +
        std::to_string(partition) + " out of range (" +
        std::to_string(_partition_build_states.size()) + ") in operator " +
        std::to_string(this->get_operator_id()));
    }
    auto& slot = _partition_build_states[partition];

    if (slot.build_state.load(std::memory_order_acquire) == BUILD_HASH_TABLE_STATE::SCHEDULED) {
      if (input_batches.size() != 2) {
        throw std::runtime_error(
          "In sirius_physical_hash_join::execute: BUILD_PROBE SCHEDULED expects probe + build "
          "batch, got " +
          std::to_string(input_batches.size()) + " batches in operator " +
          std::to_string(this->get_operator_id()));
      }
      auto const& build_batch_ro  = input_batches[1];
      auto build_keys_result      = prepare_join_keys(build_batch_ro,
                                                 right_key_col_indices,
                                                 cast_necessary,
                                                 key_casts,
                                                 /*is_left_side=*/false,
                                                 stream);
      cudf::table_view build_keys = build_keys_result.keys;
      {
        // This partition's slot has a single writer — the one SCHEDULED build task — and no probe
        // task for it runs until build_state becomes BUILT below, so the slot needs no lock. The
        // release-store of build_state publishes these writes to acquiring probe tasks. Distinct
        // partitions build concurrently on their own GPUs.
        if (auto const* ms = build_batch_ro.get_memory_space(); ms) {
          slot.device_id = ms->get_device_id();
        }
        slot.built_table_cast_columns = std::move(build_keys_result.owned_cast_columns);
        slot.build_table              = build_batch_ro;
        if (join_type == duckdb::JoinType::MARK || join_type == duckdb::JoinType::SEMI ||
            join_type == duckdb::JoinType::ANTI) {
          // MARK/SEMI/ANTI: build a reusable filtered_join on the right (filter) keys; each probe
          // batch's semi_join/anti_join returns left-row match indices (scattered into a BOOL8 mark
          // for MARK, gathered as the output rows for SEMI/ANTI).
          slot.filtered_table = make_right_filtered_join_ptr(build_keys, compare_nulls(), stream);
          // Record whether the build keys contain a NULL; MARK three-valued logic needs it at probe
          // time, but the build keys are not retained beyond this scope.
          if (join_type == duckdb::JoinType::MARK) {
            set_build_has_null(_build_has_null, table_has_any_null(build_keys));
          }
          SIRIUS_LOG_DEBUG("sirius_physical_hash_join id {}: using filtered_join (BUILD_PROBE {})",
                           this->get_operator_id(),
                           duckdb::JoinTypeToString(join_type));
        } else if ((join_type == duckdb::JoinType::INNER || join_type == duckdb::JoinType::LEFT) &&
                   (unique_build_keys ||
                    (runtime_distinct_build_probe &&
                     compare_nulls() == cudf::null_equality::UNEQUAL &&
                     build_keys_are_distinct(build_keys, compare_nulls(), stream)))) {
          slot.distinct_hash_table =
            std::make_unique<cudf::distinct_hash_join>(build_keys, compare_nulls(), 0.5, stream);
          SIRIUS_LOG_DEBUG(
            "sirius_physical_hash_join id {}: using distinct_hash_join (BUILD_PROBE, {})",
            this->get_operator_id(),
            unique_build_keys ? "proven unique" : "runtime-distinct");
        } else {
          slot.hash_table = std::make_unique<cudf::hash_join>(build_keys, compare_nulls(), stream);
        }
        stream.synchronize();  // Ensure the hash table is fully built before we allow any probe
                               // batches to proceed.
        slot.build_state.store(BUILD_HASH_TABLE_STATE::BUILT, std::memory_order_release);
      }
    }
    if (slot.build_state.load(std::memory_order_acquire) == BUILD_HASH_TABLE_STATE::BUILT) {
      // Hash table is built, we can process probe batches. The probe-side keys will be processed in
      // the same way as the mixed join path, but with an equality-only predicate.
      auto probe_keys_result      = prepare_join_keys(input_batches[0],
                                                 left_key_col_indices,
                                                 cast_necessary,
                                                 key_casts,
                                                 /*is_left_side=*/true,
                                                 stream);
      cudf::table_view probe_keys = probe_keys_result.keys;

      left_full = get_cudf_table_view(input_batches[0]);

      if (join_type == duckdb::JoinType::MARK) {
        // Reuse the persistent filtered_join (built on the right/filter side): probe with this left
        // batch to get its matched left-row indices, then materialize all left rows + BOOL8 mark.
        auto semi_indices = slot.filtered_table->semi_join(probe_keys, stream);
        return resolve_mark_join_result(*semi_indices,
                                        left_full,
                                        lhs_output_columns.col_idxs,
                                        probe_keys,
                                        _build_has_null.load(std::memory_order_acquire) > 0,
                                        mark_is_null_safe(),
                                        input_batches[0],
                                        stream,
                                        batch_telemetry());
      }

      if (join_type == duckdb::JoinType::SEMI || join_type == duckdb::JoinType::ANTI) {
        // Reuse the persistent filtered_join (built on the right/filter side): probe with this left
        // batch to get the matched (SEMI) / unmatched (ANTI) left-row indices, then fall through to
        // gather_join_output, which collects the left side only (collect_right is false).
        // right_full stays default-constructed since it is never dereferenced for these join types.
        left_indices = (join_type == duckdb::JoinType::SEMI)
                         ? slot.filtered_table->semi_join(probe_keys, stream)
                         : slot.filtered_table->anti_join(probe_keys, stream);
      } else {
        right_full = slot.build_table.value()
                       .get_data()
                       ->cast<cucascade::gpu_table_representation>()
                       .get_table_view();

        if (slot.distinct_hash_table) {
          // Distinct hash join path (unique build keys, INNER or LEFT only).
          if (join_type == duckdb::JoinType::INNER) {
            auto result   = slot.distinct_hash_table->inner_join(probe_keys, stream);
            left_indices  = std::move(result.first);
            right_indices = std::move(result.second);
          } else {
            // LEFT: returns only build indices; probe indices are implicit [0..N-1].
            auto build_indices = slot.distinct_hash_table->left_join(probe_keys, stream);
            return gather_distinct_left_join_output(left_full,
                                                    right_full,
                                                    lhs_output_columns.col_idxs,
                                                    rhs_output_columns.col_idxs,
                                                    std::move(build_indices),
                                                    *input_batches[0].get_memory_space(),
                                                    stream,
                                                    batch_telemetry());
          }
        } else {
          if (join_type == duckdb::JoinType::INNER) {
            auto result   = slot.hash_table->inner_join(probe_keys, {}, stream);
            left_indices  = std::move(result.first);
            right_indices = std::move(result.second);
          } else if (join_type == duckdb::JoinType::LEFT) {
            auto result   = slot.hash_table->left_join(probe_keys, {}, stream);
            left_indices  = std::move(result.first);
            right_indices = std::move(result.second);
          } else if (join_type == duckdb::JoinType::OUTER) {
            auto result   = slot.hash_table->full_join(probe_keys, {}, stream);
            left_indices  = std::move(result.first);
            right_indices = std::move(result.second);
          } else {
            throw std::runtime_error("Unsupported join type in BUILD_PROBE mode: " +
                                     duckdb::JoinTypeToString(join_type));
          }
        }
      }

    } else {
      throw std::runtime_error(std::format(
        "In sirius_physical_hash_join::execute: invalid hash table build state {} for partition {} "
        "in BUILD_PROBE mode for operator id {}",
        static_cast<int>(slot.build_state.load(std::memory_order_acquire)),
        partition,
        this->get_operator_id()));
    }

  } else if (_join_mode == HASH_JOIN_MODE::MIXED_JOIN) {
    if (input_batches.size() != 2) {
      throw std::runtime_error("Expected 2 input batches for hash join, got " +
                               std::to_string(input_batches.size()) + " input batches");
    }
    left_full  = get_cudf_table_view(input_batches[0]);
    right_full = get_cudf_table_view(input_batches[1]);
    // Mixed join: equality conditions drive the hash table; inequality conditions are evaluated
    // via a cuDF AST binary predicate on the full input tables.
    auto left_keys_result     = prepare_join_keys(input_batches[0],
                                              left_key_col_indices,
                                              cast_necessary,
                                              key_casts,
                                              /*is_left_side=*/true,
                                              stream);
    auto right_keys_result    = prepare_join_keys(input_batches[1],
                                               right_key_col_indices,
                                               cast_necessary,
                                               key_casts,
                                               /*is_left_side=*/false,
                                               stream);
    cudf::table_view left_eq  = left_keys_result.keys;
    cudf::table_view right_eq = right_keys_result.keys;

    sirius::gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto pred =
      translator.translate_join_conditions(conditions, num_equality_conditions, conditions.size());
    if (!pred) {
      throw std::runtime_error(
        "In sirius_physical_hash_join: failed to translate mixed join inequality conditions to "
        "cuDF AST predicate");
    }

    if (join_type == duckdb::JoinType::MARK) {
      auto semi_indices = cudf::mixed_left_semi_join(
        left_eq, right_eq, left_full, right_full, pred->back(), compare_nulls(), stream);
      set_build_has_null(_build_has_null, table_has_any_null(right_eq));
      return resolve_mark_join_result(*semi_indices,
                                      left_full,
                                      lhs_output_columns.col_idxs,
                                      left_eq,
                                      _build_has_null.load(std::memory_order_acquire) > 0,
                                      mark_is_null_safe(),
                                      input_batches[0],
                                      stream,
                                      batch_telemetry());
    } else if (join_type == duckdb::JoinType::INNER) {
      auto result = cudf::mixed_inner_join(
        left_eq, right_eq, left_full, right_full, pred->back(), compare_nulls(), {}, stream);
      left_indices  = std::move(result.first);
      right_indices = std::move(result.second);
    } else if (join_type == duckdb::JoinType::LEFT) {
      auto result = cudf::mixed_left_join(
        left_eq, right_eq, left_full, right_full, pred->back(), compare_nulls(), {}, stream);
      left_indices  = std::move(result.first);
      right_indices = std::move(result.second);
    } else if (join_type == duckdb::JoinType::RIGHT) {
      // Implement as a swapped left join: right becomes the probe side, left becomes the build
      // side. The predicate is rebuilt with LEFT/RIGHT table references flipped to match.
      auto swapped_pred = translator.translate_join_conditions(
        conditions, num_equality_conditions, conditions.size(), /*swap_sides=*/true);
      if (!swapped_pred) {
        throw std::runtime_error(
          "In sirius_physical_hash_join: failed to translate swapped predicate for RIGHT mixed "
          "join");
      }
      auto result   = cudf::mixed_left_join(right_eq,
                                          left_eq,
                                          right_full,
                                          left_full,
                                          swapped_pred->back(),
                                          compare_nulls(),
                                            {},
                                          stream);
      right_indices = std::move(result.first);
      left_indices  = std::move(result.second);
    } else if (join_type == duckdb::JoinType::OUTER) {
      auto result = cudf::mixed_full_join(
        left_eq, right_eq, left_full, right_full, pred->back(), compare_nulls(), {}, stream);
      left_indices  = std::move(result.first);
      right_indices = std::move(result.second);
    } else if (join_type == duckdb::JoinType::SEMI) {
      left_indices = cudf::mixed_left_semi_join(
        left_eq, right_eq, left_full, right_full, pred->back(), compare_nulls(), stream);
    } else if (join_type == duckdb::JoinType::ANTI) {
      left_indices = cudf::mixed_left_anti_join(
        left_eq, right_eq, left_full, right_full, pred->back(), compare_nulls(), stream);
    } else if (join_type == duckdb::JoinType::RIGHT_SEMI) {
      auto swapped_pred = translator.translate_join_conditions(
        conditions, num_equality_conditions, conditions.size(), /*swap_sides=*/true);
      if (!swapped_pred) {
        throw std::runtime_error(
          "In sirius_physical_hash_join: failed to translate swapped predicate for RIGHT_SEMI "
          "mixed join");
      }
      right_indices = cudf::mixed_left_semi_join(
        right_eq, left_eq, right_full, left_full, swapped_pred->back(), compare_nulls(), stream);
    } else if (join_type == duckdb::JoinType::RIGHT_ANTI) {
      auto swapped_pred = translator.translate_join_conditions(
        conditions, num_equality_conditions, conditions.size(), /*swap_sides=*/true);
      if (!swapped_pred) {
        throw std::runtime_error(
          "In sirius_physical_hash_join: failed to translate swapped predicate for RIGHT_ANTI "
          "mixed join");
      }
      right_indices = cudf::mixed_left_anti_join(
        right_eq, left_eq, right_full, left_full, swapped_pred->back(), compare_nulls(), stream);
    } else {
      throw std::runtime_error("Unsupported join type for mixed join: " +
                               duckdb::JoinTypeToString(join_type));
    }
  } else {  // STANDARD HASH JOIN
    if (input_batches.size() != 2) {
      throw std::runtime_error("Expected 2 input batches for hash join, got " +
                               std::to_string(input_batches.size()) + " input batches");
    }
    left_full                   = get_cudf_table_view(input_batches[0]);
    right_full                  = get_cudf_table_view(input_batches[1]);
    auto left_keys_result       = prepare_join_keys(input_batches[0],
                                              left_key_col_indices,
                                              cast_necessary,
                                              key_casts,
                                              /*is_left_side=*/true,
                                              stream);
    auto right_keys_result      = prepare_join_keys(input_batches[1],
                                               right_key_col_indices,
                                               cast_necessary,
                                               key_casts,
                                               /*is_left_side=*/false,
                                               stream);
    cudf::table_view left_keys  = left_keys_result.keys;
    cudf::table_view right_keys = right_keys_result.keys;

    if (unique_build_keys &&
        (join_type == duckdb::JoinType::INNER || join_type == duckdb::JoinType::LEFT)) {
      // Distinct hash join: build on right (build) keys, probe with left (probe) keys.
      // Only reached for pure-equal joins (build-uniqueness gate), so compare_nulls()
      // is UNEQUAL here.
      cudf::distinct_hash_join dht(right_keys, compare_nulls(), 0.5, stream);
      SIRIUS_LOG_DEBUG("sirius_physical_hash_join id {}: using distinct_hash_join (STANDARD)",
                       this->get_operator_id());
      if (join_type == duckdb::JoinType::INNER) {
        auto result   = dht.inner_join(left_keys, stream);
        left_indices  = std::move(result.first);
        right_indices = std::move(result.second);
      } else {
        // LEFT: returns only build indices; probe indices are implicit.
        auto build_indices = dht.left_join(left_keys, stream);
        return gather_distinct_left_join_output(left_full,
                                                right_full,
                                                lhs_output_columns.col_idxs,
                                                rhs_output_columns.col_idxs,
                                                std::move(build_indices),
                                                *input_batches[0].get_memory_space(),
                                                stream,
                                                batch_telemetry());
      }
    } else if (join_type == duckdb::JoinType::INNER) {
      auto join_result = cudf::inner_join(left_keys, right_keys, compare_nulls(), stream);
      left_indices     = std::move(join_result.first);
      right_indices    = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::LEFT) {
      auto join_result = cudf::left_join(left_keys, right_keys, compare_nulls(), stream);
      left_indices     = std::move(join_result.first);
      right_indices    = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::RIGHT) {
      auto join_result = cudf::left_join(right_keys, left_keys, compare_nulls(), stream);
      right_indices    = std::move(join_result.first);
      left_indices     = std::move(join_result.second);
    } else if (join_type == duckdb::JoinType::SEMI) {
      auto filtered_join_object = make_right_filtered_join(right_keys, compare_nulls(), stream);
      left_indices              = filtered_join_object.semi_join(left_keys, stream);
    } else if (join_type == duckdb::JoinType::RIGHT_SEMI) {
      auto filtered_join_object = make_right_filtered_join(left_keys, compare_nulls(), stream);
      right_indices             = filtered_join_object.semi_join(right_keys, stream);
    } else if (join_type == duckdb::JoinType::ANTI) {
      auto filtered_join_object = make_right_filtered_join(right_keys, compare_nulls(), stream);
      left_indices              = filtered_join_object.anti_join(left_keys, stream);
    } else if (join_type == duckdb::JoinType::RIGHT_ANTI) {
      auto filtered_join_object = make_right_filtered_join(left_keys, compare_nulls(), stream);
      right_indices             = filtered_join_object.anti_join(right_keys, stream);
    } else if (join_type == duckdb::JoinType::MARK) {
      // MARK join: output ALL left rows + a BOOL8 column indicating match presence.
      // Use a semi join to find which left rows have matches in the right table; both APIs
      // return left-side match indices that resolve_mark_join_result scatters into the BOOL8 mark.
      //
      // Adaptive build side: filtered_join builds on the right; when the right (probe) side is
      // much larger than the left (output) side, building on the smaller left via cudf::mark_join
      // is faster (see issue #510 microbenchmark). Crossover is hardware-dependent and configured
      // via operator_params::mark_join_build_switch_ratio (0 disables).
      if (mark_join_build_switch_ratio > 0.0 && left_full.num_rows() > 0 &&
          static_cast<double>(right_full.num_rows()) >=
            mark_join_build_switch_ratio * static_cast<double>(left_full.num_rows())) {
        SIRIUS_LOG_DEBUG(
          "sirius_physical_hash_join id {}: MARK using cudf::mark_join (build on left, "
          "left_rows={}, right_rows={})",
          this->get_operator_id(),
          left_full.num_rows(),
          right_full.num_rows());
        auto mark_join_object = make_left_mark_join(left_keys, compare_nulls(), stream);
        auto semi_indices     = mark_join_object.semi_join(right_keys, stream);
        set_build_has_null(_build_has_null, table_has_any_null(right_keys));
        return resolve_mark_join_result(*semi_indices,
                                        left_full,
                                        lhs_output_columns.col_idxs,
                                        left_keys,
                                        _build_has_null.load(std::memory_order_acquire) > 0,
                                        mark_is_null_safe(),
                                        input_batches[0],
                                        stream,
                                        batch_telemetry());
      }
      auto filtered_join_object = make_right_filtered_join(right_keys, compare_nulls(), stream);
      auto semi_indices         = filtered_join_object.semi_join(left_keys, stream);
      set_build_has_null(_build_has_null, table_has_any_null(right_keys));
      return resolve_mark_join_result(*semi_indices,
                                      left_full,
                                      lhs_output_columns.col_idxs,
                                      left_keys,
                                      _build_has_null.load(std::memory_order_acquire) > 0,
                                      mark_is_null_safe(),
                                      input_batches[0],
                                      stream,
                                      batch_telemetry());
    } else if (join_type == duckdb::JoinType::OUTER) {
      auto join_result = cudf::full_join(left_keys, right_keys, compare_nulls(), stream);
      left_indices     = std::move(join_result.first);
      right_indices    = std::move(join_result.second);
    } else {
      throw std::runtime_error("Unsupported join type: " + duckdb::JoinTypeToString(join_type));
    }
  }

  return gather_join_output(join_type,
                            left_full,
                            right_full,
                            lhs_output_columns.col_idxs,
                            rhs_output_columns.col_idxs,
                            std::move(left_indices),
                            std::move(right_indices),
                            *input_batches[0].get_memory_space(),
                            stream,
                            batch_telemetry());
}

void sirius_physical_hash_join::publish_dynamic_filters(cudf::table_view const& build_view,
                                                        rmm::cuda_stream_view stream)
{
  // The delivery hook owns the PUBLISHING claim until this function sets a terminal state.
  D_ASSERT(_dynamic_filter_publication_state.load(std::memory_order_acquire) ==
           dynamic_filter_publication_state::PUBLISHING);

  try {
    if (_dynamic_filter_plan.enabled()) {
      auto const outcome =
        sirius::op::publish_dynamic_filters(_dynamic_filter_plan, build_view, stream);
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] dynamic-filter publication: {} key(s) considered, {} skipped "
        "(domain gate), {} skipped (type mismatch), {} membership + {} zone-map built, {} "
        "filter(s) "
        "pushed across {} active target(s).",
        outcome.keys_considered,
        outcome.keys_skipped_domain_gate,
        outcome.keys_skipped_type_mismatch,
        outcome.membership_filters_built,
        outcome.zone_map_filters_built,
        outcome.filters_pushed,
        outcome.active_targets);
      if (_dynamic_filter_stats != nullptr) {
        auto& stats        = *_dynamic_filter_stats;
        auto const relaxed = std::memory_order_relaxed;
        stats.keys_considered.fetch_add(outcome.keys_considered, relaxed);
        stats.keys_with_known_domain.fetch_add(outcome.keys_with_known_domain, relaxed);
        stats.keys_skipped_domain_gate.fetch_add(outcome.keys_skipped_domain_gate, relaxed);
        stats.keys_skipped_type_mismatch.fetch_add(outcome.keys_skipped_type_mismatch, relaxed);
        stats.keys_build_exceeded_domain.fetch_add(outcome.keys_build_exceeded_domain, relaxed);
        stats.membership_filters_built.fetch_add(outcome.membership_filters_built, relaxed);
        stats.zone_map_filters_built.fetch_add(outcome.zone_map_filters_built, relaxed);
        stats.publications_skipped_targets_drained.fetch_add(outcome.skipped_targets_drained,
                                                             relaxed);
        stats.filters_pushed.fetch_add(outcome.filters_pushed, relaxed);
      }
    }
    _dynamic_filter_publication_state.store(dynamic_filter_publication_state::FINISHED,
                                            std::memory_order_release);
    if (_dynamic_filter_stats != nullptr) {
      _dynamic_filter_stats->publications_finished.fetch_add(1, std::memory_order_relaxed);
    }
  } catch (rmm::out_of_memory const& oom) {
    // Dynamic filters are optional; device OOM fails publication without failing the query.
    // FAILED, not reopen: retrying a sibling delivery under the same memory pressure is the
    // storm this catch exists to avoid (the no-usable-source skip path reopens instead).
    _dynamic_filter_publication_state.store(dynamic_filter_publication_state::FAILED,
                                            std::memory_order_release);
    if (_dynamic_filter_stats != nullptr) {
      _dynamic_filter_stats->publications_failed.fetch_add(1, std::memory_order_relaxed);
    }
    SIRIUS_LOG_WARN(
      "[sirius_physical_hash_join] dynamic-filter publication (id={}) hit device memory "
      "exhaustion; continuing without filters: {}",
      get_operator_id(),
      oom.what());
  } catch (...) {
    _dynamic_filter_publication_state.store(dynamic_filter_publication_state::FAILED,
                                            std::memory_order_release);
    if (_dynamic_filter_stats != nullptr) {
      _dynamic_filter_stats->publications_failed.fetch_add(1, std::memory_order_relaxed);
    }
    throw;
  }
}

void sirius_physical_hash_join::push_data_batch_partitioned(
  std::string_view port_id,
  std::shared_ptr<::cucascade::data_batch> batch,
  std::size_t partition_idx)
{
  // Publish only from a complete build; a partial filter could drop valid join rows.
  bool claimed = false;
  if (port_id == "build" && batch) {
    bool wired_but_unusable = false;
    HASH_JOIN_MODE mode     = HASH_JOIN_MODE::STANDARD;
    {
      std::scoped_lock lg(op_state_mutex);
      const bool open = _dynamic_filter_publication_state.load(std::memory_order_acquire) ==
                        dynamic_filter_publication_state::OPEN;
      const bool wired = _dynamic_filter_plan.enabled();
      claimed          = open && wired && _build_arrives_whole;
      // Claim under the mutex that closes OPEN, preventing finalization from racing publication.
      if (claimed) {
        _dynamic_filter_publication_state.store(dynamic_filter_publication_state::PUBLISHING,
                                                std::memory_order_release);
      }

      wired_but_unusable = open && wired && !claimed && !_build_not_whole_reported;
      if (wired_but_unusable) { _build_not_whole_reported = true; }
      mode = _join_mode;
    }
    if (claimed && _dynamic_filter_stats != nullptr) {
      _dynamic_filter_stats->publication_attempts.fetch_add(1, std::memory_order_relaxed);
    }
    if (wired_but_unusable) {
      SIRIUS_LOG_DEBUG(
        "[sirius_physical_hash_join] dynamic filter NOT published (id={}): mode={}; the upstream "
        "PARTITION did not report a build arriving as one batch covering the whole build side. It "
        "reports one for a BUILD_PROBE build that is single-partition or broadcast, and otherwise "
        "only for a build-side sizing decision that lands in a single partition and finds a "
        "build-side CONCAT to fold. Probe-driven sizing (right-family joins), a hash-partitioned "
        "multi-partition build, and a missing build-side CONCAT each fail that. See this join's "
        "partition strategy log line.",
        get_operator_id(),
        mode == HASH_JOIN_MODE::BUILD_PROBE  ? "BUILD_PROBE"
        : mode == HASH_JOIN_MODE::MIXED_JOIN ? "MIXED_JOIN"
                                             : "STANDARD");
      if (_dynamic_filter_stats != nullptr) {
        _dynamic_filter_stats->publications_skipped_build_not_whole.fetch_add(
          1, std::memory_order_relaxed);
      }
    }
  }

  if (!claimed) {
    sirius_physical_partition_consumer_operator::push_data_batch_partitioned(
      port_id, batch, partition_idx);
    return;
  }

  try {
    // Acquire the read lock before routing makes the batch eligible for downgrade.
    auto build_ro = batch->to_read_only();
    sirius_physical_partition_consumer_operator::push_data_batch_partitioned(
      port_id, batch, partition_idx);

    nvtx3::scoped_range nvtx_range{"dynfilter::publish_hook"};
    auto* ms = build_ro.get_data() ? build_ro.get_memory_space() : nullptr;
    bool const gpu_resident =
      ms != nullptr && build_ro.get_current_tier() == ::cucascade::memory::Tier::GPU;
    bool const source_usable =
      gpu_resident && _dynamic_filter_plan.has_replica_on_device(ms->get_device_id());
    if (!source_usable) {
      if (_dynamic_filter_stats != nullptr) {
        _dynamic_filter_stats->publications_skipped_source_not_resident.fetch_add(
          1, std::memory_order_relaxed);
      }
      if (gpu_resident) {
        SIRIUS_LOG_DEBUG(
          "[sirius_physical_hash_join] dynamic-filter publication (id={}) skipped: the "
          "whole-build batch is resident on GPU {}, a device this join's plan holds no replica "
          "space for.",
          get_operator_id(),
          ms->get_device_id());
      } else {
        SIRIUS_LOG_DEBUG(
          "[sirius_physical_hash_join] dynamic-filter publication (id={}) skipped: the "
          "whole-build batch is not GPU-resident.",
          get_operator_id());
      }
      // Reopen for another broadcast delivery; OPEN transitions share op_state_mutex.
      std::scoped_lock lg(op_state_mutex);
      _dynamic_filter_publication_state.store(dynamic_filter_publication_state::OPEN,
                                              std::memory_order_release);
      return;
    }

    // Wait for the build writer before reading on the publication stream.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{ms->get_device_id()}};
    auto publish_stream = ms->acquire_stream();
    if (auto const writer_event = build_ro.get_writer_event(); writer_event != nullptr) {
      auto const status = cudaStreamWaitEvent(publish_stream.value(), writer_event, 0);
      if (status != cudaSuccess) {
        throw std::runtime_error(
          std::string("[sirius_physical_hash_join::push_data_batch_partitioned] dynamic-filter "
                      "writer-event wait failed: ") +
          cudaGetErrorString(status));
      }
    } else {
      auto const status = cudaDeviceSynchronize();
      if (status != cudaSuccess) {
        throw std::runtime_error(
          std::string("[sirius_physical_hash_join::push_data_batch_partitioned] dynamic-filter "
                      "source synchronization failed: ") +
          cudaGetErrorString(status));
      }
    }
    publish_dynamic_filters(sirius::get_cudf_table_view(build_ro), publish_stream);
  } catch (...) {
    // Handle only failures that occurred before publish_dynamic_filters().
    auto expected = dynamic_filter_publication_state::PUBLISHING;
    if (_dynamic_filter_publication_state.compare_exchange_strong(
          expected,
          dynamic_filter_publication_state::FAILED,
          std::memory_order_acq_rel,
          std::memory_order_acquire) &&
        _dynamic_filter_stats != nullptr) {
      _dynamic_filter_stats->publications_failed.fetch_add(1, std::memory_order_relaxed);
    }
    throw;
  }
}

void sirius_physical_hash_join::on_finalize_operator()
{
  std::scoped_lock lg(op_state_mutex);

  // Finalization closes only an unclaimed publication window.
  auto expected = dynamic_filter_publication_state::OPEN;
  _dynamic_filter_publication_state.compare_exchange_strong(
    expected,
    dynamic_filter_publication_state::CLOSED,
    std::memory_order_acq_rel,
    std::memory_order_acquire);

  if (_join_mode == HASH_JOIN_MODE::BUILD_PROBE) {
    // Each partition's hash table lives on its own GPU (partition_idx % num_gpus). Free every slot
    // on the device it was built on so cuco/rmm releases memory in the right device context.
    for (auto& slot : _partition_build_states) {
      std::optional<rmm::cuda_set_device_raii> device_guard;
      if (slot.device_id >= 0) { device_guard.emplace(rmm::cuda_device_id{slot.device_id}); }
      slot.hash_table.reset();
      slot.distinct_hash_table.reset();
      slot.filtered_table.reset();
      slot.build_table = std::nullopt;
      slot.built_table_cast_columns.clear();
      slot.build_state.store(BUILD_HASH_TABLE_STATE::DESTROYED, std::memory_order_release);
    }
  }
}

}  // namespace op
}  // namespace sirius
