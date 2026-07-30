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

#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_filter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <cudf/types.hpp>

#include <memory>
#include <vector>

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
  // Reject nested filter predicates before planning the child.
  for (auto const& predicate : op.expressions) {
    reject_nested_column_operation(*predicate, "a filter predicate");
  }
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan = create_plan(*op.children[0]);

  // A filter that carries a projection map drops/reorders columns on its way out. When there is a
  // predicate to evaluate, fold that gather into the filter's select(): the predicate runs over the
  // full input, but only the projected columns are materialized, so columns referenced only by the
  // predicate are never materialized and no trailing projection is needed. A projection map with no
  // predicate to absorb it (below) still emits a standalone projection.
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

    duckdb::vector<sirius::logical_type> filter_types = plan->types;  // identity: keep all columns
    std::vector<cudf::size_type> output_indices;                      // empty ⇒ keep all columns
    if (op.HasProjectionMap()) {
      filter_types = sirius::from_duckdb_vec(op.types);  // the projected output types, in order
      output_indices.reserve(op.projection_map.size());
      for (duckdb::idx_t const idx : op.projection_map) {
        output_indices.push_back(static_cast<cudf::size_type>(idx));
      }
    }

    // Reject unsupported predicates before the physical filter stores them.
    auto predicate = sirius::ast::from_duckdb(*combined);
    if (predicate == nullptr) {
      throw duckdb::NotImplementedException("Unsupported filter predicate (falling back to CPU): " +
                                            combined->ToString());
    }
    auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(std::move(filter_types),
                                                                        std::move(predicate),
                                                                        op.estimated_cardinality,
                                                                        std::move(output_indices));
    filter->children.push_back(std::move(plan));
    plan = std::move(filter);
  } else if (op.HasProjectionMap()) {
    // Projection map with no predicate to fold it into: emit a standalone projection.
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
