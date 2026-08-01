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

#include "transparent/sirius_optimizer_extension.hpp"

#include "planner/duckdb_join_filter_candidate_adapter.hpp"
#include "sirius_context.hpp"

#include <duckdb/common/enums/optimizer_type.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <log/logging.hpp>

#include <cstddef>
#include <exception>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::transparent {

namespace {

bool gpu_execution_enabled(const duckdb::ClientContext& context)
{
  duckdb::Value setting;
  auto lookup_result = context.TryGetCurrentSetting("gpu_execution", setting);
  return lookup_result && !setting.IsNull() && setting.GetValue<bool>();
}

//===--------------------------------------------------------------------===//
// Join-dependent filter derivation (see sirius_pre_optimizer_hook)
//===--------------------------------------------------------------------===//

/// Guards against quadratic blow-up: an N-branch disjunction derives one N-child OR per table.
constexpr std::size_t kMaxDerivedDisjuncts = 16;

void collect_table_indexes(duckdb::Expression const& expr, std::unordered_set<duckdb::idx_t>& out)
{
  duckdb::ExpressionIterator::VisitExpression<duckdb::BoundColumnRefExpression>(
    expr, [&out](duckdb::BoundColumnRefExpression const& colref) {
      out.insert(colref.binding.table_index);
    });
}

bool references_multiple_tables(duckdb::Expression const& expr)
{
  std::unordered_set<duckdb::idx_t> tables;
  collect_table_indexes(expr, tables);
  return tables.size() > 1;
}

/// The single-table conjuncts of one OR branch, grouped by table index. Ordered so the derived
/// predicates - and therefore the plan - are deterministic.
using per_table_conjuncts =
  std::map<duckdb::idx_t, duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>>;

/// AND-decompose @p expr; every leaf that restricts exactly one table is filed under that table in
/// @p conjuncts.
void extract_single_table_conjuncts(duckdb::Expression const& expr, per_table_conjuncts& conjuncts)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CONJUNCTION &&
      expr.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_AND) {
    for (auto const& child : expr.Cast<duckdb::BoundConjunctionExpression>().children) {
      extract_single_table_conjuncts(*child, conjuncts);
    }
    return;
  }
  // The derived copy is evaluated on rows the original disjunction rejects, and - once pushed into
  // a scan - ahead of the branch it came from. Anything whose evaluation is observable (volatile,
  // throwing, subquery, unbound parameter) would therefore change behaviour, so it is left alone.
  // This is stricter than DuckDB's own JoinDependentFilterRule, which only checks volatility.
  if (expr.IsVolatile() || expr.CanThrow() || expr.HasSubquery() || expr.HasParameter()) { return; }

  std::unordered_set<duckdb::idx_t> tables;
  collect_table_indexes(expr, tables);
  if (tables.size() != 1) { return; }

  conjuncts[*tables.begin()].push_back(expr.Copy());
}

/// AND @p conjuncts back into a single expression. Never called with an empty list.
duckdb::unique_ptr<duckdb::Expression> conjoin(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> const& conjuncts)
{
  if (conjuncts.size() == 1) { return conjuncts[0]->Copy(); }
  auto result =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto const& conjunct : conjuncts) {
    result->children.push_back(conjunct->Copy());
  }
  return result;
}

/// Append to @p filter, for every table restricted by *all* branches of an OR-ed filter expression,
/// the OR of those per-branch restrictions.
void derive_join_dependent_filters(duckdb::LogicalFilter& filter)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> derived;

  for (auto const& expression : filter.expressions) {
    if (expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONJUNCTION ||
        expression->GetExpressionType() != duckdb::ExpressionType::CONJUNCTION_OR) {
      continue;
    }
    auto const& disjunction   = expression->Cast<duckdb::BoundConjunctionExpression>();
    std::size_t const num_alt = disjunction.children.size();
    if (num_alt < 2 || num_alt > kMaxDerivedDisjuncts) { continue; }

    // A disjunction confined to one table is already pushable as it stands; there is nothing to
    // derive. Only a branch spanning several tables hides a single-table restriction.
    bool spans_multiple_tables = false;
    for (auto const& branch : disjunction.children) {
      if (references_multiple_tables(*branch)) {
        spans_multiple_tables = true;
        break;
      }
    }
    if (!spans_multiple_tables) { continue; }

    std::vector<per_table_conjuncts> per_branch(num_alt);
    for (std::size_t i = 0; i < num_alt; i++) {
      extract_single_table_conjuncts(*disjunction.children[i], per_branch[i]);
    }

    for (auto const& entry : per_branch[0]) {
      auto const table_index = entry.first;

      // A branch that restricts this table not at all leaves it unrestricted overall - the
      // disjunction then implies nothing about it.
      bool restricted_by_every_branch = true;
      for (std::size_t i = 1; i < num_alt; i++) {
        if (per_branch[i].find(table_index) == per_branch[i].end()) {
          restricted_by_every_branch = false;
          break;
        }
      }
      if (!restricted_by_every_branch) { continue; }

      // Drop the conjuncts that every branch carries verbatim: DuckDB's DistributivityRule hoists
      // those out of the OR by itself, so keeping them would only bolt a duplicate predicate onto
      // the scan. Dropping them weakens each branch, and a weaker branch still yields an implied
      // (so still sound) disjunction.
      auto is_branch_invariant = [&](duckdb::Expression const& conjunct) {
        for (std::size_t i = 0; i < num_alt; i++) {
          auto const& other = per_branch[i].find(table_index)->second;
          bool found        = false;
          for (auto const& candidate : other) {
            if (candidate->Equals(conjunct)) {
              found = true;
              break;
            }
          }
          if (!found) { return false; }
        }
        return true;
      };

      auto restriction = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
        duckdb::ExpressionType::CONJUNCTION_OR);
      bool every_branch_restricts_further = true;
      for (std::size_t i = 0; i < num_alt; i++) {
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> varying;
        for (auto const& conjunct : per_branch[i].find(table_index)->second) {
          if (!is_branch_invariant(*conjunct)) { varying.push_back(conjunct->Copy()); }
        }
        if (varying.empty()) {
          // This branch adds nothing beyond what is already hoisted, so the OR is trivially true.
          every_branch_restricts_further = false;
          break;
        }
        restriction->children.push_back(conjoin(varying));
      }
      if (!every_branch_restricts_further) { continue; }

      derived.push_back(std::move(restriction));
    }
  }

  for (auto& restriction : derived) {
    filter.expressions.push_back(std::move(restriction));
  }
}

void derive_join_dependent_filters_recursive(duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_FILTER) {
    derive_join_dependent_filters(op.Cast<duckdb::LogicalFilter>());
  }
  for (auto& child : op.children) {
    derive_join_dependent_filters_recursive(*child);
  }
}

}  // namespace

void sirius_pre_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!plan || !gpu_execution_enabled(input.context)) { return; }

  // Optimizer hooks must not throw: a failed derivation only costs the pushdown, never the query.
  try {
    derive_join_dependent_filters_recursive(*plan);
  } catch (std::exception& e) {
    // NOTE: upstream uses sirius::sanitized_message() (util/duckdb_error_message.hpp, added by
    // #1339) which prefers duckdb::ErrorData::RawMessage(). This branch predates #1339, and the
    // helper is only cosmetic on a debug log line, so use what() directly rather than pulling in
    // an unrelated PR.
    SIRIUS_LOG_DEBUG("Transparent execution: join-dependent filter derivation failed: {}",
                     e.what());
  }
}

namespace {

/// True when @p a and @p b have the same operator type and child count at every node.
bool same_shape(duckdb::LogicalOperator const& a, duckdb::LogicalOperator const& b)
{
  if (a.type != b.type || a.children.size() != b.children.size()) { return false; }
  for (std::size_t i = 0; i < a.children.size(); i++) {
    if (!same_shape(*a.children[i], *b.children[i])) { return false; }
  }
  return true;
}

void copy_cardinalities(duckdb::LogicalOperator const& original, duckdb::LogicalOperator& copy)
{
  // A node the optimizer never estimated is left alone, so LogicalOperator::EstimateCardinality
  // keeps its lazy max-of-children fallback rather than being pinned to a fabricated 0.
  if (original.has_estimated_cardinality) {
    copy.SetEstimatedCardinality(original.estimated_cardinality);
  }
  for (std::size_t i = 0; i < original.children.size(); i++) {
    copy_cardinalities(*original.children[i], *copy.children[i]);
  }
}

/// Carry the optimizer's cardinality estimates onto a copied plan.
///
/// LogicalOperator::Copy() round-trips through BinarySerializer, and neither
/// `estimated_cardinality` nor `has_estimated_cardinality` is serialized (they appear nowhere in
/// duckdb/src/storage/serialization/), so every node of a copy comes back with 0/false. Sirius
/// then reads the estimates via EstimateCardinality() in plan_comparison_join, which - with the
/// flag cleared - silently falls back to "max of the children" and bottoms out at raw base-table
/// row counts. Every filter and every join selectivity is discarded, so a build side over
/// filtered orders reports lineitem's 5,999,330,220 rows. That number reaches
/// sirius_physical_hash_join::get_partition_strategy as estimated_probe_to_build_ratio.
void preserve_estimated_cardinality(duckdb::LogicalOperator const& original,
                                    duckdb::LogicalOperator& copy)
{
  // All-or-nothing, matching preserve_dynamic_filter_metadata: a shape mismatch means the
  // deserialized tree is not node-for-node comparable and any pairing would be guesswork.
  if (!same_shape(original, copy)) { return; }
  copy_cardinalities(original, copy);
}

}  // namespace

duckdb::unique_ptr<duckdb::LogicalOperator> copy_logical_plan(duckdb::LogicalOperator const& plan,
                                                              duckdb::ClientContext& context)
{
  auto copy = plan.Copy(context);
  planner::duckdb_join_filter_candidate_adapter::preserve_dynamic_filter_metadata(plan, *copy);
  preserve_estimated_cardinality(plan, *copy);
  return copy;
}

void sirius_optimizer_hook(duckdb::OptimizerExtensionInput& input,
                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  if (!gpu_execution_enabled(input.context)) { return; }

  auto& context = input.context;

  auto ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!ctx || !ctx->is_initialized()) { return; }
  auto conn_state = duckdb::get_sirius_connection_state(context);
  if (!conn_state || conn_state->is_internal_query_active()) { return; }

  // Copy the optimized plan into THIS connection's per-connection state,
  // stamped with the current planning generation. OnFinalizePrepare will
  // attempt create_plan() on this copy — that's the single source of truth for
  // GPU support. If the plan contains unsupported operators, create_plan()
  // throws and we fall back to CPU. A capture whose planning attempt never
  // reaches finalize (e.g. Connection::ExtractPlan) is structurally rejected
  // at the next attempt by the generation check.
  try {
    conn_state->set_captured_plan(copy_logical_plan(*plan, context));
  } catch (duckdb::NotImplementedException&) {
    // Plan not serializable — skip GPU.
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Transparent execution: failed to copy logical plan: {}", e.what());
  }
}

}  // namespace sirius::transparent
