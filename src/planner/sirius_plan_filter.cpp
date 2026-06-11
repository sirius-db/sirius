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

#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_filter.hpp"
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
sirius_physical_plan_generator::create_plan(duckdb::LogicalFilter& op)
{
  D_ASSERT(op.children.size() == 1);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan = create_plan(*op.children[0]);
  if (!op.expressions.empty()) {
    D_ASSERT(plan->types.size() > 0);
    // If the filter carries multiple predicates, AND them together into a single expression so
    // the operator only ever owns one Sirius AST node.
    duckdb::unique_ptr<duckdb::Expression> combined;
    if (op.expressions.size() > 1) {
      auto conjunction = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
        duckdb::ExpressionType::CONJUNCTION_AND);
      for (auto& expr : op.expressions) {
        conjunction->children.push_back(std::move(expr));
      }
      combined = std::move(conjunction);
    } else {
      combined = std::move(op.expressions[0]);
    }
    auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(
      plan->types, sirius::ast::from_duckdb(*combined), op.estimated_cardinality);
    filter->children.push_back(std::move(plan));
    plan = std::move(filter);
  }
  if (op.HasProjectionMap()) {
    // there is a projection map, generate a physical projection
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    for (std::size_t i = 0; i < op.projection_map.size(); i++) {
      select_list.push_back(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(op.types[i], op.projection_map[i]));
    }
    plan = push_projection(std::move(plan),
                           sirius::from_duckdb_vec(op.types),
                           translate_expressions(std::move(select_list)),
                           op.estimated_cardinality);
  }
  return plan;
}

}  // namespace sirius::planner
