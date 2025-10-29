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

#include "fallback.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"

namespace duckdb {

#define REGEXP_REPLACE_FUNC_STR "regexp_replace"

/**
 * So far we fall back to duckdb:
 *   - whenever we meet regular expressions.
 */
void FallbackChecker::Check() const {
  for (const auto& pipeline: pipelines) {
    for (const auto& op: pipeline->GetOperators()) {
      CheckOperator(op);
    }
  }
}

void FallbackChecker::CheckOperator(const GPUPhysicalOperator& op) const {
  switch (op.type) {
    case PhysicalOperatorType::PROJECTION: {
      CheckOperator(op.Cast<GPUPhysicalProjection>());
      break;
    }
  }
}

void FallbackChecker::CheckOperator(const GPUPhysicalProjection& op) const {
  for (const auto& expr: op.select_list) {
    CheckExpression(*expr);
  }
}

void FallbackChecker::CheckExpression(const Expression& expr) const {
  switch (expr.GetExpressionClass()) {
    case ExpressionClass::BOUND_BETWEEN: {
      CheckExpression(expr.Cast<BoundBetweenExpression>());
      break;
    }
    case ExpressionClass::BOUND_CASE: {
      CheckExpression(expr.Cast<BoundCaseExpression>());
      break;
    }
    case ExpressionClass::BOUND_CAST: {
      CheckExpression(expr.Cast<BoundCastExpression>());
      break;
    }
    case ExpressionClass::BOUND_COMPARISON: {
      CheckExpression(expr.Cast<BoundComparisonExpression>());
      break;
    }
    case ExpressionClass::BOUND_CONJUNCTION: {
      CheckExpression(expr.Cast<BoundConjunctionExpression>());
      break;
    }
    case ExpressionClass::BOUND_FUNCTION: {
      CheckExpression(expr.Cast<BoundFunctionExpression>());
      break;
    }
    case ExpressionClass::BOUND_OPERATOR: {
      CheckExpression(expr.Cast<BoundOperatorExpression>());
      break;
    }
  }
}

void FallbackChecker::CheckExpression(const BoundBetweenExpression& expr) const {
  CheckExpression(*expr.input);
  CheckExpression(*expr.lower);
  CheckExpression(*expr.upper);
}

void FallbackChecker::CheckExpression(const BoundCaseExpression& expr) const {
  for (const auto& case_check: expr.case_checks) {
    CheckExpression(*case_check.when_expr);
    CheckExpression(*case_check.then_expr);
  }
  CheckExpression(*expr.else_expr);
}

void FallbackChecker::CheckExpression(const BoundCastExpression& expr) const {
  CheckExpression(*expr.child);
}

void FallbackChecker::CheckExpression(const BoundComparisonExpression& expr) const {
  CheckExpression(*expr.left);
  CheckExpression(*expr.right);
}

void FallbackChecker::CheckExpression(const BoundConjunctionExpression& expr) const {
  for (const auto& child: expr.children) {
    CheckExpression(*child);
  }
}

void FallbackChecker::CheckExpression(const BoundFunctionExpression& expr) const {
  const auto& func_str = expr.function.name;
  if (func_str == REGEXP_REPLACE_FUNC_STR) {
    throw InternalException("[Fallback checker] unsupported expression: %s", REGEXP_REPLACE_FUNC_STR);
  }
  for (const auto& child: expr.children) {
    CheckExpression(*child);
  }
}

void FallbackChecker::CheckExpression(const BoundOperatorExpression& expr) const {
  for (const auto& child: expr.children) {
    CheckExpression(*child);
  }
}

}
