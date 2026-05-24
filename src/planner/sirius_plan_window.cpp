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

#include "duckdb/common/assert.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "op/sirius_physical_window.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

namespace {

bool is_bound_ref(const duckdb::Expression& expr)
{
  return expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF;
}

// Phase 1 supports only ROW_NUMBER / RANK / DENSE_RANK, sharing one PARTITION BY / ORDER BY, with
// column-reference keys, no FILTER / EXCLUDE, and no HUGEINT child columns. Anything else throws
// NotImplementedException so the whole query falls back to DuckDB CPU.
void guard_phase1(duckdb::LogicalWindow& op)
{
  // Child output columns are op.types[0 .. input_width); the rest are window results.
  const std::size_t input_width = op.types.size() - op.expressions.size();
  for (std::size_t i = 0; i < input_width; ++i) {
    auto id = op.types[i].id();
    if (id == duckdb::LogicalTypeId::HUGEINT || id == duckdb::LogicalTypeId::UHUGEINT) {
      throw duckdb::NotImplementedException(
        "Window not supported: HUGEINT/UHUGEINT child column (cuDF has no int128)");
    }
  }

  const duckdb::BoundWindowExpression* first = nullptr;
  for (auto& expr : op.expressions) {
    auto& window = expr->Cast<duckdb::BoundWindowExpression>();

    if (window.type != duckdb::ExpressionType::WINDOW_ROW_NUMBER &&
        window.type != duckdb::ExpressionType::WINDOW_RANK &&
        window.type != duckdb::ExpressionType::WINDOW_RANK_DENSE) {
      throw duckdb::NotImplementedException(
        "Window not supported: only ROW_NUMBER/RANK/DENSE_RANK in Phase 1");
    }
    if (window.exclude_clause != duckdb::WindowExcludeMode::NO_OTHER) {
      throw duckdb::NotImplementedException("Window not supported: EXCLUDE clause");
    }
    if (window.filter_expr) {
      throw duckdb::NotImplementedException("Window not supported: FILTER clause");
    }
    for (auto& partition : window.partitions) {
      if (!is_bound_ref(*partition)) {
        throw duckdb::NotImplementedException("Window not supported: non-column PARTITION BY key");
      }
    }
    for (auto& order : window.orders) {
      if (!is_bound_ref(*order.expression)) {
        throw duckdb::NotImplementedException("Window not supported: non-column ORDER BY key");
      }
    }
    // RANK/DENSE_RANK without ORDER BY degenerate to all-1; fall back rather than emit that.
    if ((window.type == duckdb::ExpressionType::WINDOW_RANK ||
         window.type == duckdb::ExpressionType::WINDOW_RANK_DENSE) &&
        window.orders.empty()) {
      throw duckdb::NotImplementedException(
        "Window not supported: RANK/DENSE_RANK without ORDER BY");
    }

    // All expressions in this LogicalWindow must share identical PARTITION BY / ORDER BY: DuckDB
    // only splits heterogeneous windows into separate PhysicalWindow at physical-plan time, but
    // Sirius intercepts the logical plan, so one LogicalWindow may carry mixed framings.
    if (first == nullptr) {
      first = &window;
    } else if (!first->PartitionsAreEquivalent(window) ||
               first->orders.size() != window.orders.size() ||
               first->GetSharedOrders(window) != window.orders.size()) {
      throw duckdb::NotImplementedException(
        "Window not supported: expressions with differing PARTITION BY / ORDER BY");
    }
  }
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalWindow& op)
{
  D_ASSERT(op.children.size() == 1);

  guard_phase1(
    op);  // throws NotImplementedException -> graceful CPU fallback for unsupported shapes

  auto plan   = create_plan(*op.children[0]);
  auto window = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                       sirius::op::sirius_physical_window>(
    op.types, std::move(op.expressions), op.estimated_cardinality);
  window->children.push_back(std::move(plan));
  return window;
}

}  // namespace sirius::planner
