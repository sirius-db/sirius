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

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <memory>

namespace sirius::planner {

namespace {

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

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalProjection& op)
{
  D_ASSERT(op.children.size() == 1);
  auto plan = create_plan(*op.children[0]);

#ifdef DEBUG
  for (auto& expr : op.expressions) {
    D_ASSERT(!expr->IsWindow());
    D_ASSERT(!expr->IsAggregate());
  }
#endif
  if (plan->types.size() == op.types.size()) {
    // check if this projection can be omitted entirely
    // this happens if a projection simply emits the columns in the same order
    // e.g. PROJECTION(#0, #1, #2, #3, ...)
    bool omit_projection = true;
    for (std::size_t i = 0; i < op.types.size(); i++) {
      if (op.expressions[i]->type == duckdb::ExpressionType::BOUND_REF) {
        auto& bound_ref = op.expressions[i]->Cast<duckdb::BoundReferenceExpression>();
        if (bound_ref.index == i) { continue; }
      }
      omit_projection = false;
      break;
    }
    if (omit_projection) {
      // the projection only directly projects the child' columns: omit it entirely
      return plan;
    }
  }

  return push_projection(std::move(plan),
                         sirius::from_duckdb_vec(op.types),
                         translate_expressions(std::move(op.expressions)),
                         op.estimated_cardinality);
}

}  // namespace sirius::planner
