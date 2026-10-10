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

/// Sources whose casts to DECIMAL and integer types go through the checked casts.
bool is_checked_numeric(cudf::type_id id)
{
  auto const type = cudf::data_type{id};
  return cudf::is_integral_not_bool(type) || cudf::is_floating_point(type) ||
         cudf::is_fixed_point(type);
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::cast const& alt, evaluation_mode mode)
{
  auto const ast_supported = alt.lowers_to_cudf_ast();
  auto const ast_op_count  = alt.cudf_ast_op_count();

  // Carrier restores must reach the materialized branch, the only path authorized to use the
  // physical representation tunnel. Semantic casts that cannot round or overflow may use the cuDF
  // AST path.
  if (ast_supported && _strategy != expression_evaluator_strategy::MATERIALIZE &&
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
  auto const return_type = sirius::get_cudf_type(alt.target_type);
  auto child             = evaluate(*alt.child, evaluation_mode::MATERIALIZE);
  if (child.is_scalar()) {
    child = evaluate_result(
      cudf::make_column_from_scalar(child.get_scalar(), _input_table.num_rows(), _stream, _mr));
  }
  std::unique_ptr<cudf::column> result_column;
  auto const source_type = child.get_column_view().type().id();
  if (alt.kind == sirius::ast::cast_kind::semantic &&
      (source_type == cudf::type_id::TIMESTAMP_NANOSECONDS ||
       source_type == cudf::type_id::TIMESTAMP_MILLISECONDS ||
       source_type == cudf::type_id::TIMESTAMP_SECONDS) &&
      return_type.id() == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    result_column = temporal::cast_to_microseconds_checked(child.get_column_view(), _stream, _mr);
  } else if (alt.kind == sirius::ast::cast_kind::semantic && is_checked_numeric(source_type) &&
             cudf::is_fixed_point(return_type)) {
    // cudf::cast truncates where DuckDB rounds, and does not check the target precision.
    result_column = cast_to_decimal(child.get_column_view(),
                                    return_type,
                                    alt.target_type.decimal_precision(),
                                    alt.try_cast,
                                    _stream,
                                    _mr);
  } else if (alt.kind == sirius::ast::cast_kind::semantic && is_checked_numeric(source_type) &&
             cudf::is_integral_not_bool(return_type)) {
    // cudf::cast truncates where DuckDB rounds, and wraps values that do not fit. HUGEINT and
    // UHUGEINT run as INT64 and UINT64, so a value outside them may still be valid in DuckDB:
    // throw so the query replays on the CPU, even for TRY_CAST.
    auto const wide_target = alt.target_type.id() == sirius::type_id::HUGEINT ||
                             alt.target_type.id() == sirius::type_id::UHUGEINT;
    result_column = cast_to_integer(
      child.get_column_view(), return_type, alt.try_cast && !wide_target, _stream, _mr);
  } else {
    // Only planner-certified carrier restoration may tunnel through a narrowed representation.
    result_column = alt.kind == sirius::ast::cast_kind::carrier_restore
                      ? sirius::cast_through_rep(child.get_column_view(), return_type, _stream, _mr)
                      : cudf::cast(child.get_column_view(), return_type, _stream, _mr);
  }
  if (mode == evaluation_mode::AST) {
    // The parent is executing in AST mode, so add the materialized result to the AST tree.
    return materialize_as_ast_column(std::move(result_column));
  }
  return evaluate_result(std::move(result_column));
}

}  // namespace sirius
