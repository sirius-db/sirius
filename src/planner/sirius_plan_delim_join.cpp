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

#include "duckdb/common/enum_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_delim_direct.hpp"
#include "sirius_context.hpp"

#include <memory>
#include <utility>

namespace sirius::planner {

namespace {

/// Read the delim-direct-lowering enable flag from the active SiriusContext config. Defaults to
/// enabled (the registered default) when the state is unavailable.
bool delim_direct_lowering_enabled(duckdb::ClientContext& context)
{
  auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!state) { return true; }
  return state->get_config().get_operator_params().enable_delim_direct_lowering;
}

// Translate a vector of DuckDB expressions into Sirius AST nodes at the planner
// boundary. The source vector is drained; size and order are preserved, with a
// null slot wherever from_duckdb declines an unsupported shape (a fallback
// signal) — matching the prior bulk-translation null-skip semantics.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    out.push_back(e ? sirius::ast::from_duckdb(*e) : nullptr);
  }
  return out;
}

}  // namespace

static void gather_delim_scans(
  sirius::op::sirius_physical_operator& op,
  duckdb::vector<duckdb::const_reference<sirius::op::sirius_physical_operator>>& delim_scans,
  std::size_t delim_index)
{
  if (op.type == sirius::op::SiriusPhysicalOperatorType::DELIM_SCAN) {
    SIRIUS_LOG_DEBUG("Found a delim scan");
    SIRIUS_LOG_DEBUG("op type: {}", op::SiriusPhysicalOperatorToString(op.type));
    auto& scan       = op.Cast<sirius::op::sirius_physical_column_data_scan>();
    scan.delim_index = duckdb::optional_idx(delim_index);
    if (scan.delim_index.IsValid()) {
      SIRIUS_LOG_DEBUG("Scan delim index: {}", scan.delim_index.GetIndex());
    } else {
      SIRIUS_LOG_DEBUG("Scan delim index invalid");
    }
    delim_scans.push_back(op);
  }
  for (auto& child : op.children) {
    gather_delim_scans(*child, delim_scans, delim_index);
  }
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_delim_join(duckdb::LogicalComparisonJoin& op)
{
  // Pure-equality EXISTS / NOT EXISTS delims collapse into a single direct semi/anti hash join
  // (sirius_plan_delim_direct.cpp). Ineligible shapes — with the typed reason logged — keep the
  // regular delim lowering below.
  if (delim_direct_lowering_enabled(context)) {
    auto analysis = classify_delim_direct_lowering(op);
    if (analysis.eligible()) {
      apply_delim_direct_lowering(op, std::move(analysis));
      SIRIUS_LOG_INFO(
        "[delim_direct] Lowered a DELIM join to a direct {} hash join with {} condition(s).",
        duckdb::EnumUtil::ToString(op.join_type),
        op.conditions.size());
      return plan_comparison_join(op);
    }
    SIRIUS_LOG_INFO("[delim_direct] Keeping the DELIM lowering for a {} delim join: {}.",
                    duckdb::EnumUtil::ToString(op.join_type),
                    to_string(analysis.refusal));
  }

  // first create the underlying join
  auto plan = plan_comparison_join(op);
  // this should create a join, not a cross product
  D_ASSERT(plan && plan->type != sirius::op::SiriusPhysicalOperatorType::CROSS_PRODUCT);
  // duplicate eliminated join
  // first gather the scans on the duplicate eliminated data set from the delim side
  const std::size_t delim_idx = op.delim_flipped ? 0 : 1;
  duckdb::vector<duckdb::const_reference<sirius::op::sirius_physical_operator>> delim_scans;
  gather_delim_scans(*plan->children[delim_idx], delim_scans, ++this->delim_index);
  if (delim_scans.empty()) {
    // no duplicate eliminated scans in the delim side!
    // in this case we don't need to create a delim join
    // just push the normal join
    return plan;
  }
  duckdb::vector<duckdb::LogicalType> delim_types;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> distinct_groups, distinct_expressions;
  for (auto& delim_expr : op.duplicate_eliminated_columns) {
    D_ASSERT(delim_expr->GetExpressionType() == duckdb::ExpressionType::BOUND_REF);
    auto& bound_ref = delim_expr->Cast<duckdb::BoundReferenceExpression>();
    delim_types.push_back(bound_ref.return_type);
    distinct_groups.push_back(
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(bound_ref.return_type, bound_ref.index));
  }
  // now create the duplicate eliminated join
  duckdb::unique_ptr<sirius::op::sirius_physical_delim_join> delim_join;
  if (op.delim_flipped) {
    delim_join = duckdb::make_uniq<sirius::op::sirius_physical_right_delim_join>(
      sirius::from_duckdb_vec(op.types),
      std::move(plan),
      delim_scans,
      op.estimated_cardinality,
      duckdb::optional_idx(this->delim_index));
  } else {
    delim_join = duckdb::make_uniq<sirius::op::sirius_physical_left_delim_join>(
      sirius::from_duckdb_vec(op.types),
      std::move(plan),
      delim_scans,
      op.estimated_cardinality,
      duckdb::optional_idx(this->delim_index));
  }
  // we still have to create the DISTINCT clause that is used to generate the duplicate eliminated
  // chunk. `distinct_root` owns it and `distinct` raw-points at it; wrap_delim_distinct may
  // later deepen the owning chain, but the object never relocates so `distinct` stays valid.
  auto distinct_uptr = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate>(
    sirius::from_duckdb_vec(delim_types),
    translate_expressions(std::move(distinct_expressions)),
    translate_expressions(std::move(distinct_groups)),
    op.estimated_cardinality);
  delim_join->distinct      = distinct_uptr.get();
  delim_join->distinct_root = std::move(distinct_uptr);

  return std::move(delim_join);
}

}  // namespace sirius::planner
