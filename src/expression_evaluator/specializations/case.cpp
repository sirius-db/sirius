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
#include <expression/function_id.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/reduction.hpp>

// standard library
#include <ranges>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

evaluate_result expression_evaluator::evaluate(sirius::ast::case_expr const& alt,
                                               evaluation_mode mode)
{
  //===----------MATERIALIZE (AST breaker)----------===//
  // CASE cannot be represented as a cudf AST operation, so we always materialize it. If the caller
  // requested AST mode, we materialize the result and wrap it as a temporary column that the
  // parent's AST tree can reference.
  std::unique_ptr<cudf::column> output;

  // First, evaluate the ELSE
  auto current_result = evaluate(*alt.else_, evaluation_mode::MATERIALIZE);

  // Iterate in reverse so the THEN of the first true WHEN is copied to the
  // output column last (overwrites any later match). SQL CASE semantics:
  // first matching WHEN wins.
  D_ASSERT(!alt.cases.empty());

  for (auto const& case_check : std::views::reverse(alt.cases)) {
    // Execute the WHEN expression to get a boolean array intermediate.
    auto current_mask = evaluate(*case_check.when_, evaluation_mode::MATERIALIZE);

    // Check for the implicit error() function that DuckDB inserts as a CASE THEN.
    if (case_check.then_->holds<sirius::ast::function_call>() &&
        case_check.then_->get<sirius::ast::function_call>().function() ==
          sirius::function_id::error) {
      // If the THEN is true anywhere, raise the error eagerly.
      bool throw_error = false;
      if (current_mask.is_scalar()) {
        auto const& bool_scalar =
          static_cast<cudf::scalar_type_t<bool> const&>(current_mask.get_scalar());
        if (bool_scalar.is_valid(_stream)) { throw_error = bool_scalar.value(_stream); }
      } else {
        auto any_result = cudf::reduce(current_mask.get_column_view(),
                                       *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                       cudf::data_type(cudf::type_id::BOOL8),
                                       _stream,
                                       _mr);
        throw_error     = static_cast<cudf::scalar_type_t<bool>*>(any_result.get())->value(_stream);
      }
      if (throw_error) {
        throw internal_exception(
          "[expression_evaluator:case]: More than one row returned by a subquery used as an "
          "expression.");
      }
      continue;
    }

    // Otherwise, evaluate the THEN and selectively copy to the output.
    auto current_then = evaluate(*case_check.then_, evaluation_mode::MATERIALIZE);
    if (current_result.is_scalar()) {
      // This can only possibly happen when i = num_checks - 1
      if (current_then.is_scalar()) {
        output = cudf::copy_if_else(current_then.get_scalar(),
                                    current_result.get_scalar(),
                                    current_mask.get_column_view(),
                                    _stream,
                                    _mr);
      } else {
        output = cudf::copy_if_else(current_then.get_column_view(),
                                    current_result.get_scalar(),
                                    current_mask.get_column_view(),
                                    _stream,
                                    _mr);
      }
    } else if (current_then.is_scalar()) {
      output = cudf::copy_if_else(current_then.get_scalar(),
                                  current_result.get_column_view(),
                                  current_mask.get_column_view(),
                                  _stream,
                                  _mr);
    } else {
      output = cudf::copy_if_else(current_then.get_column_view(),
                                  current_result.get_column_view(),
                                  current_mask.get_column_view(),
                                  _stream,
                                  _mr);
    }
    current_result = evaluate_result(std::move(output));
  }
  if (mode == evaluation_mode::AST) {
    // The caller wants an AST node.
    // Since at least one copy_if_else has been executed, current_result must have an owned column.
    if (!current_result.is_owned_column()) {
      throw internal_exception(
        "[expression_evaluator:case]: Expected an owned column after executing CASE "
        "expression.");
    }
    return materialize_as_ast_column(std::move(current_result.release_column()));
  }
  return current_result;
}

}  // namespace sirius
