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

// sirius
#include <expression/ast/node.hpp>
#include <expression_evaluator/checked_cast.hpp>
#include <expression_evaluator/date_cast.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/logical_type.hpp>
#include <helper/numeric_narrowing.hpp>
#include <helper/timestamp_semantics.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

cudf::ast::ast_operator cast_op_to_ast(sirius::type_id id)
{
  switch (id) {
    case sirius::type_id::UBIGINT: return cudf::ast::ast_operator::CAST_TO_UINT64;
    case sirius::type_id::BIGINT: return cudf::ast::ast_operator::CAST_TO_INT64;
    case sirius::type_id::DOUBLE: return cudf::ast::ast_operator::CAST_TO_FLOAT64;
    default:
      throw invalid_input_exception(
        "[cast_op_to_ast] unsupported CAST target type id={}; cuDF AST supports "
        "UBIGINT, BIGINT, DOUBLE.",
        static_cast<int>(id));
  }
}

/// Numeric casts to integer and DECIMAL types, which cudf::cast truncates and wraps where DuckDB
/// rounds and range-checks.
bool needs_checked_cast(cudf::data_type source, cudf::data_type target)
{
  auto const integer_or_decimal = [](cudf::data_type t) {
    return cudf::is_integral_not_bool(t) || cudf::is_fixed_point(t);
  };
  return (integer_or_decimal(source) || cudf::is_floating_point(source)) &&
         integer_or_decimal(target);
}

/// DECIMAL to a DECIMAL with at least as many integer and fractional digits, such as the operand
/// widening DuckDB adds to decimal arithmetic. cudf::cast is exact here and no value can fail, so
/// the range check and its stream sync are skipped.
bool is_decimal_widening(sirius::logical_type const& source, sirius::logical_type const& target)
{
  return source.is_decimal() && target.is_decimal() &&
         source.decimal_scale() <= target.decimal_scale() &&
         source.decimal_precision() - source.decimal_scale() <=
           target.decimal_precision() - target.decimal_scale();
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::cast const& alt, evaluation_mode mode)
{
  auto const ast_supported = alt.lowers_to_cudf_ast();

  auto const ast_op_count = alt.cudf_ast_op_count();

  // Carrier restores must reach the materialized branch, the only path authorized to use the
  // physical representation tunnel. Semantic casts that cannot round or overflow may use the cuDF
  // AST path.
  if (ast_supported && _policy.strategy != expression_evaluator_strategy::MATERIALIZE &&
      (mode == evaluation_mode::AST || ast_op_count >= _min_ast_size)) {
    auto child            = evaluate(*alt.child, evaluation_mode::AST);
    auto const& cast_expr = _ast_tree.emplace<cudf::ast::operation>(
      cast_op_to_ast(alt.target_type.id()), child.get_expr());

    if (mode == evaluation_mode::AST) {
      //===----------1: AST Mode----------===//
      return evaluate_result(compose(cast_expr, {&child}));
    }
    //===----------2: MATERIALIZE Mode, evaluate node with AST----------===//
    auto result_column = evaluate_ast(cast_expr);

    release_temporaries({&child});
    return evaluate_result(std::move(result_column));
  }

  //===----------3: MATERIALIZE Mode, evaluate node with unary/binary ops----------===//
  auto const return_type = sirius::require_cudf_type(alt.target_type);
  auto child             = evaluate(*alt.child, evaluation_mode::MATERIALIZE);
  if (child.is_scalar()) {
    child = evaluate_result(
      cudf::make_column_from_scalar(child.get_scalar(), _input_table.num_rows(), _stream, _mr));
  }
  auto const input = child.get_column_view();
  std::unique_ptr<cudf::column> result_column;
  if (alt.kind == sirius::ast::cast_kind::carrier_restore) {
    // Only planner-certified carrier restoration may tunnel through the representation.
    result_column = sirius::cast_through_rep(input, return_type, _stream, _mr);
  } else if (input.type().id() == cudf::type_id::TIMESTAMP_DAYS &&
             (return_type.id() == cudf::type_id::TIMESTAMP_SECONDS ||
              return_type.id() == cudf::type_id::TIMESTAMP_MILLISECONDS ||
              return_type.id() == cudf::type_id::TIMESTAMP_MICROSECONDS ||
              return_type.id() == cudf::type_id::TIMESTAMP_NANOSECONDS)) {
    result_column = cast_date_to_timestamp(input, return_type, alt.try_cast, _stream, _mr);
  } else if ((input.type().id() == cudf::type_id::TIMESTAMP_NANOSECONDS ||
              input.type().id() == cudf::type_id::TIMESTAMP_MILLISECONDS ||
              input.type().id() == cudf::type_id::TIMESTAMP_SECONDS) &&
             return_type.id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    // Carrier restores were handled above; only semantic casts reach this branch.
    result_column = temporal::cast_to_microseconds_checked(input, _stream, _mr);
  } else if (alt.kind == sirius::ast::cast_kind::semantic &&
             needs_checked_cast(input.type(), return_type) &&
             !is_decimal_widening(alt.child->return_type(), alt.target_type)) {
    // HUGEINT and UHUGEINT run as INT64 and UINT64, so a value outside them may still be valid in
    // DuckDB: throw so the query replays on the CPU, even for TRY_CAST.
    auto const wide_target = alt.target_type.id() == sirius::type_id::HUGEINT ||
                             alt.target_type.id() == sirius::type_id::UHUGEINT;
    result_column = checked_cast(input,
                                 return_type,
                                 alt.target_type.decimal_precision(),
                                 alt.try_cast && !wide_target,
                                 _stream,
                                 _mr);
  } else {
    result_column = cudf::cast(input, return_type, _stream, _mr);
  }
  if (mode == evaluation_mode::AST) {
    // The parent is executing in AST mode, so add the materialized result to the AST tree.
    return materialize_as_ast_column(std::move(result_column));
  }
  return evaluate_result(std::move(result_column));
}

}  // namespace sirius
