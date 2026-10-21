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

#include "expression/ast/from_duckdb.hpp"

#include <duckdb/planner/filter/expression_filter.hpp>

// sirius
#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/between.hpp"
#include "expression/ast/case_expr.hpp"
#include "expression/ast/cast.hpp"
#include "expression/ast/coalesce.hpp"
#include "expression/ast/comparison.hpp"
#include "expression/ast/conjunction.hpp"
#include "expression/ast/constant.hpp"
#include "expression/ast/function_call.hpp"
#include "expression/ast/in_list.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/unary_op.hpp"
#include "expression/date_trunc_unit.hpp"
#include "expression/function_id.hpp"
#include "expression/join_condition.hpp"  // sirius::comparison_type, sirius::from_duckdb(ExpressionType)
#include "expression/substring_slice.hpp"
#include "expression/value.hpp"         // sirius::from_duckdb(Value const&, logical_type const&)
#include "helper/type_conversions.hpp"  // sirius::from_duckdb(LogicalType const&)

// duckdb
#include <duckdb/common/assert.hpp>
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <re2/re2.h>

// standard library
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sirius::ast {

namespace {

// Translate a sequence of DuckDB child expressions (skipping the first @p start
// entries), short-circuiting to std::nullopt the moment any child is
// unsupported (i.e. from_duckdb returns null). Callers treat nullopt as "this
// whole expression is unsupported" and propagate a null node upward.
template <class Children>
std::optional<std::vector<std::unique_ptr<node>>> translate_children(Children const& children,
                                                                     std::size_t start = 0)
{
  std::vector<std::unique_ptr<node>> out;
  out.reserve(children.size() > start ? children.size() - start : 0);
  for (std::size_t i = start; i < children.size(); ++i) {
    auto translated = from_duckdb(*children[i]);
    if (!translated) { return std::nullopt; }
    out.push_back(std::move(translated));
  }
  return out;
}

std::unique_ptr<node> translate_reference(duckdb::BoundReferenceExpression const& expr)
{
  return std::make_unique<node>(
    reference{static_cast<uint32_t>(expr.Index()), sirius::from_duckdb(expr.GetReturnType())});
}

std::unique_ptr<node> translate_constant(duckdb::BoundConstantExpression const& expr)
{
  auto return_type = sirius::from_duckdb(expr.GetReturnType());
  auto payload     = sirius::from_duckdb(expr.GetValue(), return_type);
  return std::make_unique<node>(constant{std::move(payload), std::move(return_type)});
}

std::unique_ptr<node> translate_comparison(duckdb::BoundFunctionExpression const& expr)
{
  // Filter the BoundComparisonExpression ExpressionType down to the set that the
  // Sirius comparison node carries (sirius::comparison_type). The set mirrors
  // sirius::from_duckdb(duckdb::ExpressionType) in src/expression/join_condition.cpp.
  switch (expr.GetExpressionType()) {
    case duckdb::ExpressionType::COMPARE_EQUAL:
    case duckdb::ExpressionType::COMPARE_NOTEQUAL:
    case duckdb::ExpressionType::COMPARE_LESSTHAN:
    case duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO:
    case duckdb::ExpressionType::COMPARE_GREATERTHAN:
    case duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO:
    case duckdb::ExpressionType::COMPARE_DISTINCT_FROM:
    case duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM: break;
    default: return nullptr;
  }
  auto left  = from_duckdb(duckdb::BoundComparisonExpression::Left(expr));
  auto right = from_duckdb(duckdb::BoundComparisonExpression::Right(expr));
  if (!left || !right) { return nullptr; }
  return std::make_unique<node>(comparison{
    /*op=*/sirius::from_duckdb(expr.GetExpressionType()),
    /*left=*/std::move(left),
    /*right=*/std::move(right),
  });
}

std::unique_ptr<node> translate_conjunction(duckdb::BoundConjunctionExpression const& expr)
{
  conjunction::kind op{conjunction::kind::invalid};
  switch (expr.GetExpressionType()) {
    case duckdb::ExpressionType::CONJUNCTION_AND: op = conjunction::kind::op_and; break;
    case duckdb::ExpressionType::CONJUNCTION_OR: op = conjunction::kind::op_or; break;
    default: return nullptr;
  }
  auto children = translate_children(expr.GetChildren());
  if (!children) { return nullptr; }
  return std::make_unique<node>(conjunction{op, std::move(*children)});
}

std::unique_ptr<node> translate_between(duckdb::BoundFunctionExpression const& expr)
{
  auto input = from_duckdb(duckdb::BoundBetweenExpression::Input(expr));
  auto lower = from_duckdb(duckdb::BoundBetweenExpression::LowerBound(expr));
  auto upper = from_duckdb(duckdb::BoundBetweenExpression::UpperBound(expr));
  if (!input || !lower || !upper) { return nullptr; }
  return std::make_unique<node>(between{
    /*input=*/std::move(input),
    /*lower=*/std::move(lower),
    /*upper=*/std::move(upper),
    /*lower_inclusive=*/duckdb::BoundBetweenExpression::LowerInclusive(expr),
    /*upper_inclusive=*/duckdb::BoundBetweenExpression::UpperInclusive(expr),
  });
}

std::unique_ptr<node> translate_case(duckdb::BoundCaseExpression const& expr)
{
  std::vector<case_expr::when_then> cases;
  cases.reserve(expr.CaseChecks().size());
  for (auto const& check : expr.CaseChecks()) {
    auto when_node = from_duckdb(*check.when_expr);
    auto then_node = from_duckdb(*check.then_expr);
    if (!when_node || !then_node) { return nullptr; }
    cases.push_back(case_expr::when_then{std::move(when_node), std::move(then_node)});
  }
  auto else_node = from_duckdb(expr.Else());
  if (!else_node) { return nullptr; }
  return std::make_unique<node>(
    case_expr{std::move(cases), std::move(else_node), sirius::from_duckdb(expr.GetReturnType())});
}

std::unique_ptr<node> translate_cast(duckdb::BoundFunctionExpression const& expr)
{
  // cuDF does not implement DuckDB's temporal-numeric cast semantics. Decline the expression
  // before a same-shaped DATE carrier conversion can reach the representation tunnel; planning
  // then applies the configured fallback policy. Planner-generated restores bypass translation.
  auto const& source_type = duckdb::BoundCastExpression::Child(expr).GetReturnType();
  auto const& target_type = expr.GetReturnType();
  if ((source_type.IsTemporal() && target_type.IsNumeric()) ||
      (source_type.IsNumeric() && target_type.IsTemporal())) {
    return nullptr;
  }

  // cuDF casts can wrap where DuckDB CAST throws or TRY_CAST returns NULL. Reject
  // unsigned integer casts unless the target carrier fits the full source domain.
  // HUGEINT/UHUGEINT currently use INT64/UINT64 on the GPU.
  // TODO: Implement checked GPU integer casts: return NULL on overflow for TRY_CAST
  // and raise an error for CAST, allowing these conversions without CPU fallback.
  if (source_type.IsUnsigned()) {
    duckdb::idx_t target_value_bits = 0;
    switch (target_type.id()) {
      case duckdb::LogicalTypeId::TINYINT: target_value_bits = 7; break;
      case duckdb::LogicalTypeId::SMALLINT: target_value_bits = 15; break;
      case duckdb::LogicalTypeId::INTEGER: target_value_bits = 31; break;
      case duckdb::LogicalTypeId::BIGINT:
      case duckdb::LogicalTypeId::HUGEINT: target_value_bits = 63; break;
      case duckdb::LogicalTypeId::UTINYINT: target_value_bits = 8; break;
      case duckdb::LogicalTypeId::USMALLINT: target_value_bits = 16; break;
      case duckdb::LogicalTypeId::UINTEGER: target_value_bits = 32; break;
      case duckdb::LogicalTypeId::UBIGINT:
      case duckdb::LogicalTypeId::UHUGEINT: target_value_bits = 64; break;
      default: break;
    }
    if (target_value_bits != 0 &&
        duckdb::GetTypeIdSize(source_type.InternalType()) * 8 > target_value_bits) {
      return nullptr;
    }
  }

  auto child = from_duckdb(duckdb::BoundCastExpression::Child(expr));
  if (!child) { return nullptr; }
  return std::make_unique<node>(cast{
    /*child=*/std::move(child),
    /*target_type=*/sirius::from_duckdb(target_type),
    /*try_cast=*/duckdb::BoundCastExpression::IsTryCast(expr),
  });
}

std::optional<int64_t> bigint_constant(duckdb::Expression const& expr)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) { return std::nullopt; }
  auto const& value = expr.Cast<duckdb::BoundConstantExpression>().GetValue();
  if (value.IsNull() || value.type().id() != duckdb::LogicalTypeId::BIGINT) { return std::nullopt; }
  return value.GetValue<int64_t>();
}

// The GPU evaluator applies one slice to every row, so it needs constant BIGINT bounds that
// gpu_substring_slice can map onto DuckDB's semantics.
bool gpu_supports_substring(duckdb::BoundFunctionExpression const& expr)
{
  if (expr.GetChildren().size() != 2 && expr.GetChildren().size() != 3) { return false; }
  auto const offset = bigint_constant(*expr.GetChildren()[1]);
  auto const length = expr.GetChildren().size() == 3 ? bigint_constant(*expr.GetChildren()[2])
                                                     : substring_default_length;
  return offset && length && gpu_substring_slice(*offset, *length);
}

// The GPU evaluator rounds FLOAT and DOUBLE inputs to a constant number of decimal places.
bool gpu_supports_round(duckdb::BoundFunctionExpression const& expr)
{
  if (expr.GetChildren().size() != 1 && expr.GetChildren().size() != 2) { return false; }
  auto const input_type = expr.GetChildren()[0]->GetReturnType().id();
  if (input_type != duckdb::LogicalTypeId::FLOAT && input_type != duckdb::LogicalTypeId::DOUBLE) {
    return false;
  }
  if (expr.GetChildren().size() == 1) { return true; }
  auto const& precision = *expr.GetChildren()[1];
  if (precision.GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) { return false; }
  auto const& value = precision.Cast<duckdb::BoundConstantExpression>().GetValue();
  return !value.IsNull() && value.type().id() == duckdb::LogicalTypeId::INTEGER;
}

// The GPU evaluator takes the absolute value of numeric inputs.
bool gpu_supports_abs(duckdb::BoundFunctionExpression const& expr)
{
  if (expr.GetChildren().size() != 1) { return false; }
  auto const& type = expr.GetChildren()[0]->GetReturnType();
  // Everything numeric besides HUGEINT and UHUGEINT.
  return type.IsNumeric() && type.id() != duckdb::LogicalTypeId::HUGEINT &&
         type.id() != duckdb::LogicalTypeId::UHUGEINT;
}

std::unique_ptr<node> translate_function(duckdb::BoundFunctionExpression const& expr)
{
  // Optional scan predicates are hints; the required predicate remains in the plan.
  if (duckdb::ExpressionFilter::IsRootOptionalExpression(expr)) {
    return translate_constant(duckdb::BoundConstantExpression(duckdb::Value::BOOLEAN(true)));
  }
  if (duckdb::BoundComparisonExpression::IsComparison(expr)) { return translate_comparison(expr); }
  if (duckdb::BoundCastExpression::IsCast(expr)) { return translate_cast(expr); }
  if (expr.GetExpressionType() == duckdb::ExpressionType::COMPARE_BETWEEN) {
    return translate_between(expr);
  }
  auto func_id_opt =
    sirius::from_duckdb_function_name(expr.Function().GetName().GetIdentifierName());
  if (!func_id_opt.has_value()) { return nullptr; }
  if (*func_id_opt == function_id::date_trunc) {
    // Match the GPU evaluator's supported frequencies. It requires a constant string;
    // reject other forms here so neither unsupported units nor dynamic units reach execution.
    if (expr.GetChildren().size() != 2 ||
        expr.GetChildren()[0]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) {
      return nullptr;
    }
    auto const& frequency =
      expr.GetChildren()[0]->Cast<duckdb::BoundConstantExpression>().GetValue();
    if (frequency.IsNull() || frequency.type().id() != duckdb::LogicalTypeId::VARCHAR) {
      return nullptr;
    }
    auto const& unit = duckdb::StringValue::Get(frequency);
    if (!parse_gpu_date_trunc_unit(unit)) { return nullptr; }
  }
  if (*func_id_opt == function_id::constant_or_null) {
    // The GPU evaluator reads the result value from a constant first argument.
    if (expr.GetChildren().size() < 2 ||
        expr.GetChildren()[0]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) {
      return nullptr;
    }
  }
  if (*func_id_opt == function_id::substring && !gpu_supports_substring(expr)) { return nullptr; }
  if (*func_id_opt == function_id::round && !gpu_supports_round(expr)) { return nullptr; }
  if (*func_id_opt == function_id::abs && !gpu_supports_abs(expr)) { return nullptr; }
  if (*func_id_opt == function_id::regexp_replace) {
    // The GPU implementation supports constant patterns/replacements and DuckDB's default
    // options only. In particular, never silently discard the fourth (options) argument.
    if (expr.GetChildren().size() != 3) { return nullptr; }
    for (std::size_t i = 1; i < 3; ++i) {
      if (expr.GetChildren()[i]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_CONSTANT) {
        return nullptr;
      }
      auto const& value = expr.GetChildren()[i]->Cast<duckdb::BoundConstantExpression>().GetValue();
      if (value.IsNull() || value.type() != duckdb::LogicalType::VARCHAR) { return nullptr; }
    }
    auto const& pattern = duckdb::StringValue::Get(
      expr.GetChildren()[1]->Cast<duckdb::BoundConstantExpression>().GetValue());
    // RE2's shorthand classes and word boundaries are ASCII-only; cuDF's are Unicode-aware.
    // cuDF also does not implement POSIX classes. Skip escaped characters so a literal \\w
    // remains supported, but reject these constructs wherever they occur in the pattern.
    for (std::size_t i = 0; i < pattern.size(); ++i) {
      if (pattern[i] == '\\' && i + 1 < pattern.size()) {
        switch (pattern[++i]) {
          case 'w':
          case 'W':
          case 'd':
          case 'D':
          case 's':
          case 'S':
          case 'b':
          case 'B': return nullptr;
          default: break;
        }
      } else if (pattern.compare(i, 2, "[:") == 0) {
        return nullptr;
      } else if (pattern.compare(i, 2, "(?") == 0 && pattern.compare(i, 3, "(?:") != 0) {
        // cuDF only supports non-capturing groups among the (?...) forms. Inline flags such
        // as (?m) would also change the anchor semantics the evaluator relies on.
        return nullptr;
      }
    }
    duckdb_re2::RE2::Options options;
    options.set_log_errors(false);
    duckdb_re2::RE2 const regex(pattern, options);
    if (!regex.ok()) { return nullptr; }
    // cuDF can compile empty-only patterns such as (?:) and a{0} into programs that never
    // replace anything. RE2 bounds every possible match; two empty bounds prove that no
    // nonempty match is possible. One byte is enough to distinguish this from a* or a?.
    std::string min_match, max_match;
    if (regex.PossibleMatchRange(&min_match, &max_match, 1) && min_match.empty() &&
        max_match.empty()) {
      return nullptr;
    }
    auto const& replacement = duckdb::StringValue::Get(
      expr.GetChildren()[2]->Cast<duckdb::BoundConstantExpression>().GetValue());
    // cuDF's template syntax differs from RE2's. Only translate single-digit backreferences;
    // leave escaped backslashes and invalid rewrite escapes to DuckDB. Braced references are
    // reserved for the translated cuDF template below, but are literal text in DuckDB.
    bool has_backrefs = false;
    for (std::size_t i = 0; i < replacement.size(); ++i) {
      if (replacement[i] != '\\') { continue; }
      if (++i == replacement.size() || replacement[i] < '0' || replacement[i] > '9') {
        return nullptr;
      }
      has_backrefs = true;
    }
    if (has_backrefs && replacement.find("${") != std::string::npos) { return nullptr; }
    if (has_backrefs) {
      std::string error;
      // Validate against the original captures before adding the GPU wrapper groups.
      if (!regex.CheckRewriteString(replacement, &error)) { return nullptr; }
    }
  }
  auto arguments = translate_children(expr.GetChildren());
  if (!arguments) { return nullptr; }
  auto return_type = sirius::from_duckdb(expr.GetReturnType());
  return std::make_unique<node>(
    function_call{*func_id_opt, std::move(*arguments), std::move(return_type)});
}

std::unique_ptr<node> translate_aggregate(duckdb::BoundAggregateExpression const& aggr)
{
  auto agg_id_opt =
    sirius::from_duckdb_aggregate_name(aggr.Function().GetName().GetIdentifierName());
  if (!agg_id_opt.has_value()) { return nullptr; }
  auto arguments = translate_children(aggr.GetChildren());
  if (!arguments) { return nullptr; }
  auto return_type = sirius::from_duckdb(aggr.GetReturnType());
  return std::make_unique<node>(
    aggregate{*agg_id_opt, std::move(*arguments), std::move(return_type), aggr.IsDistinct()});
}

std::unique_ptr<node> translate_operator(duckdb::BoundOperatorExpression const& expr)
{
  auto const op_type = expr.GetExpressionType();

  // Unary operators share a single AST node; only the kind tag differs.
  unary_op::kind unary_kind{unary_op::kind::invalid};
  switch (op_type) {
    case duckdb::ExpressionType::OPERATOR_NOT: unary_kind = unary_op::kind::op_not; break;
    case duckdb::ExpressionType::OPERATOR_IS_NULL: unary_kind = unary_op::kind::op_is_null; break;
    case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL:
      unary_kind = unary_op::kind::op_is_not_null;
      break;
    // TRY needs per-row error suppression, which the GPU evaluator does not implement.
    case duckdb::ExpressionType::OPERATOR_TRY: return nullptr;
    default: break;
  }
  if (unary_kind != unary_op::kind::invalid) {
    D_ASSERT(!expr.GetChildren().empty());
    auto child = from_duckdb(*expr.GetChildren()[0]);
    if (!child) { return nullptr; }
    return std::make_unique<node>(unary_op{unary_kind, std::move(child)});
  }

  if (op_type == duckdb::ExpressionType::OPERATOR_COALESCE) {
    auto children = translate_children(expr.GetChildren());
    if (!children) { return nullptr; }
    return std::make_unique<node>(
      coalesce{std::move(*children), sirius::from_duckdb(expr.GetReturnType())});
  }

  if (op_type == duckdb::ExpressionType::COMPARE_IN ||
      op_type == duckdb::ExpressionType::COMPARE_NOT_IN) {
    D_ASSERT(!expr.GetChildren().empty());
    auto probe = from_duckdb(*expr.GetChildren()[0]);
    if (!probe) { return nullptr; }
    auto values = translate_children(expr.GetChildren(), /*start=*/1);
    if (!values) { return nullptr; }
    return std::make_unique<node>(in_list{
      /*probe=*/std::move(probe),
      /*values=*/std::move(*values),
      /*negated=*/op_type == duckdb::ExpressionType::COMPARE_NOT_IN,
    });
  }

  return nullptr;
}

}  // namespace

std::unique_ptr<node> from_duckdb(duckdb::Expression const& expr)
{
  switch (expr.GetExpressionClass()) {
    case duckdb::ExpressionClass::BOUND_AGGREGATE:
      return translate_aggregate(expr.Cast<duckdb::BoundAggregateExpression>());
    case duckdb::ExpressionClass::BOUND_CASE:
      return translate_case(expr.Cast<duckdb::BoundCaseExpression>());
    case duckdb::ExpressionClass::BOUND_CONJUNCTION:
      return translate_conjunction(expr.Cast<duckdb::BoundConjunctionExpression>());
    case duckdb::ExpressionClass::BOUND_CONSTANT:
      return translate_constant(expr.Cast<duckdb::BoundConstantExpression>());
    case duckdb::ExpressionClass::BOUND_FUNCTION:
      return translate_function(expr.Cast<duckdb::BoundFunctionExpression>());
    case duckdb::ExpressionClass::BOUND_OPERATOR:
      return translate_operator(expr.Cast<duckdb::BoundOperatorExpression>());
    case duckdb::ExpressionClass::BOUND_PARAMETER: return nullptr;
    case duckdb::ExpressionClass::BOUND_REF:
      return translate_reference(expr.Cast<duckdb::BoundReferenceExpression>());
    default:
      throw duckdb::InternalException("[sirius::ast::from_duckdb] Unknown ExpressionClass: %d",
                                      expr.GetExpressionClass());
  }
}

}  // namespace sirius::ast
