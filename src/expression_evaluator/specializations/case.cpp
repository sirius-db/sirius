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
#include <cudf/cudf_utils.hpp>

#include <expression/ast/node.hpp>
#include <expression/function_id.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/unary.hpp>

// standard library
#include <algorithm>
#include <optional>
#include <ranges>

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

namespace {

/// True for the implicit error() THEN that DuckDB inserts for a scalar subquery returning more
/// than one row.
bool is_error_branch(sirius::ast::node const& then)
{
  return then.holds<sirius::ast::function_call>() &&
         then.get<sirius::ast::function_call>().function() == sirius::function_id::error;
}

[[noreturn]] void raise_subquery_error()
{
  throw internal_exception(
    "[expression_evaluator:case]: More than one row returned by a subquery used as an "
    "expression.");
}

}  // namespace

evaluate_result expression_evaluator::evaluate(sirius::ast::case_expr const& alt,
                                               evaluation_mode mode)
{
  D_ASSERT(!alt.cases.empty());

  // The eager pass tests an error() THEN's WHEN on every row, ignoring earlier WHENs, which is
  // exact only for the first WHEN. DuckDB generates no other, but SQL can spell one.
  auto const eager_is_exact = std::ranges::none_of(
    alt.cases | std::views::drop(1), [](auto const& c) { return is_error_branch(*c.then_); });
  auto current_result = alt.eager
                          ? evaluate_case_eagerly(alt)
                          : with_selective_retry([&] { return evaluate_case_eagerly(alt); },
                                                 [&] { return evaluate_case_selectively(alt); },
                                                 eager_is_exact);

  //===----------MATERIALIZE (AST breaker)----------===//
  // CASE cannot be represented as a cudf AST operation, so we always materialize it. If the caller
  // requested AST mode, we wrap the result as a temporary column that the parent's AST tree can
  // reference.
  if (mode == evaluation_mode::AST) {
    // An eager pass whose every THEN is error() returns ELSE as it is: a scalar, or a borrowed
    // view of an input column. The temporary column must own a full column.
    if (current_result.is_scalar()) {
      current_result = evaluate_result(cudf::make_column_from_scalar(
        current_result.get_scalar(), _input_table.num_rows(), _stream, _mr));
    } else if (current_result.is_column_view()) {
      current_result = evaluate_result(
        std::make_unique<cudf::column>(current_result.get_column_view(), _stream, _mr));
    }
    return materialize_as_ast_column(current_result.release_column());
  }
  return current_result;
}

evaluate_result expression_evaluator::evaluate_case_eagerly(sirius::ast::case_expr const& alt)
{
  std::unique_ptr<cudf::column> output;

  // First, evaluate the ELSE
  auto current_result = evaluate(*alt.else_, evaluation_mode::MATERIALIZE);

  // Iterate in reverse so the THEN of the first true WHEN is copied to the
  // output column last (overwrites any later match). SQL CASE semantics:
  // first matching WHEN wins.
  for (auto const& case_check : std::views::reverse(alt.cases)) {
    // Execute the WHEN expression to get a boolean array intermediate.
    auto current_mask = evaluate(*case_check.when_, evaluation_mode::MATERIALIZE);

    // Check for the implicit error() function that DuckDB inserts as a CASE THEN.
    if (is_error_branch(*case_check.then_)) {
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
      if (throw_error) { raise_subquery_error(); }
      continue;
    }

    // A folded WHEN can be a scalar, including NULL (which selects ELSE).
    if (current_mask.is_scalar()) {
      current_mask = evaluate_result(cudf::make_column_from_scalar(
        current_mask.get_scalar(), _input_table.num_rows(), _stream, _mr));
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
  return current_result;
}

std::unique_ptr<cudf::column> expression_evaluator::evaluate_case_selectively(
  sirius::ast::case_expr const& alt)
{
  // DuckDB semantics: each WHEN sees only the rows no earlier WHEN matched, where NULL does not
  // match, and evaluates as a predicate; each THEN sees the rows its WHEN matched, and ELSE the
  // rest. A branch that no row reaches is not evaluated. Every mask below is a null-free BOOL8
  // column over all input rows.
  auto const num_rows = _input_table.num_rows();
  auto const type     = require_cudf_type(alt.return_type());
  // No row reaches a branch of an empty input; ELSE gives the empty result its type.
  if (num_rows == 0) { return evaluate_on_selected_rows(*alt.else_, nullptr, type); }
  auto const condition = cudf::data_type{cudf::type_id::BOOL8};
  auto const no        = cudf::numeric_scalar<bool>(false, true, _stream, _mr);

  std::unique_ptr<cudf::column> result;
  auto const assign = [&](sirius::ast::node const& branch, cudf::column_view const& rows) {
    auto value = evaluate_on_selected_rows(branch, &rows, type);
    if (!value) { return; }
    if (!result) { result = sirius::make_null_column(value->view(), num_rows, _stream, _mr); }
    result = scatter_selected(value->view(), result->view(), rows);
  };

  // Rows no WHEN has matched yet; empty while that is every row.
  std::unique_ptr<cudf::column> unmatched;
  bool exhausted = false;
  for (auto const& case_check : alt.cases) {
    auto const candidates =
      unmatched ? std::optional<cudf::column_view>{unmatched->view()} : std::nullopt;
    auto matched = evaluate_on_selected_rows(
      *case_check.when_, candidates ? &*candidates : nullptr, condition, /*predicate=*/true);
    // Every row has matched an earlier WHEN, so no later WHEN or ELSE is evaluated.
    if (!matched) {
      exhausted = true;
      break;
    }
    if (matched->has_nulls()) { matched = cudf::replace_nulls(matched->view(), no, _stream, _mr); }
    if (candidates) {
      auto const none = cudf::make_column_from_scalar(no, num_rows, _stream, _mr);
      matched         = scatter_selected(matched->view(), none->view(), *candidates);
    }

    if (is_error_branch(*case_check.then_)) {
      auto const any = cudf::reduce(matched->view(),
                                    *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                    condition,
                                    _stream,
                                    _mr);
      if (static_cast<cudf::scalar_type_t<bool> const&>(*any).value(_stream)) {
        raise_subquery_error();
      }
    } else {
      assign(*case_check.then_, matched->view());
    }

    unmatched = candidates
                  ? cudf::copy_if_else(no, *candidates, matched->view(), _stream, _mr)
                  : cudf::unary_operation(matched->view(), cudf::unary_operator::NOT, _stream, _mr);
  }
  if (!exhausted) { assign(*alt.else_, unmatched->view()); }
  // Each row reached a THEN or ELSE that produced it, or an error() THEN that raised.
  if (!result) {
    throw internal_exception("[expression_evaluator:case]: no branch produced the CASE rows");
  }
  return result;
}

}  // namespace sirius
