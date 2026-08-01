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

#include <expression/ast/aggregate.hpp>  // sirius::ast::aggregate
#include <expression/ast/node.hpp>       // sirius::ast::node alternatives
#include <expression_evaluator/expression_evaluator.hpp>
#include <sirius/exception.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <data/data_batch_utils.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <memory>
#include <numeric>
#include <variant>

namespace {
/// Returns true if cudf::cast supports this type (fixed-width only; no STRING/LIST/STRUCT/etc.).
bool IsFixedWidth(cudf::data_type const& type)
{
  auto const id = type.id();
  return id != cudf::type_id::STRING && id != cudf::type_id::LIST && id != cudf::type_id::STRUCT &&
         id != cudf::type_id::DICTIONARY32 && id != cudf::type_id::EMPTY;
}
}  // namespace

namespace sirius {
using data_batch      = cucascade::data_batch;
using evaluate_result = expression_evaluator::evaluate_result;
using ast_result      = expression_evaluator::ast_result;

//===----------------------------------------------------------------------===//
// ast_result
//===----------------------------------------------------------------------===//

expression_evaluator::ast_result::ast_result(expr_ref e,
                                             std::vector<std::vector<std::size_t>> scalar_indices,
                                             std::vector<std::vector<std::size_t>> column_indices)
  : expr(e)
{
  auto const total_scalars = std::accumulate(
    scalar_indices.begin(),
    scalar_indices.end(),
    std::size_t(0),
    [](std::size_t sum, const std::vector<std::size_t>& vec) { return sum + vec.size(); });
  temp_scalar_indices.reserve(total_scalars);
  for (auto& vec : scalar_indices) {
    temp_scalar_indices.insert(temp_scalar_indices.end(),
                               std::make_move_iterator(vec.begin()),
                               std::make_move_iterator(vec.end()));
  }

  auto const total_columns = std::accumulate(
    column_indices.begin(),
    column_indices.end(),
    std::size_t(0),
    [](std::size_t sum, const std::vector<std::size_t>& vec) { return sum + vec.size(); });
  temp_column_indices.reserve(total_columns);
  for (auto& vec : column_indices) {
    temp_column_indices.insert(temp_column_indices.end(),
                               std::make_move_iterator(vec.begin()),
                               std::make_move_iterator(vec.end()));
  }
}

//===----------------------------------------------------------------------===//
// evaluate_result
//===----------------------------------------------------------------------===//

std::reference_wrapper<cudf::ast::expression const>
expression_evaluator::evaluate_result::get_expr() const
{
  if (!is_ast()) {
    throw std::runtime_error("[evaluate_result] Attempted to get expr from materialized result");
  }
  return std::get<ast_result>(payload).expr;
}

std::vector<std::size_t> expression_evaluator::evaluate_result::get_temp_scalar_indices() const
{
  if (is_ast()) { return std::get<ast_result>(payload).temp_scalar_indices; }
  return {};
}

std::vector<std::size_t> expression_evaluator::evaluate_result::get_temp_column_indices() const
{
  if (is_ast()) { return std::get<ast_result>(payload).temp_column_indices; }
  return {};
}

cudf::scalar const& expression_evaluator::evaluate_result::get_scalar() const
{
  if (!is_scalar()) {
    throw std::runtime_error(
      "[evaluate_result] Attempted to get scalar from non-scalar evaluate_result");
  }
  auto const& scalar_payload = std::get<std::unique_ptr<cudf::scalar>>(payload);
  return *scalar_payload;
}

cudf::column_view expression_evaluator::evaluate_result::get_column_view() const
{
  if (is_ast() || is_scalar()) {
    throw std::runtime_error(
      "[evaluate_result] Attempted to get column view from non-column evaluate_result");
  }
  if (std::holds_alternative<cudf::column_view>(payload)) {
    return std::get<cudf::column_view>(payload);
  }
  auto const& column_payload = std::get<std::unique_ptr<cudf::column>>(payload);
  return column_payload->view();
}

std::unique_ptr<cudf::column> expression_evaluator::evaluate_result::release_column()
{
  if (std::holds_alternative<std::unique_ptr<cudf::column>>(payload)) {
    return std::move(std::get<std::unique_ptr<cudf::column>>(payload));
  }
  throw std::runtime_error(
    "[evaluate_result] Attempted to get column from evaluate_result that does not hold a column");
}

//===----------------------------------------------------------------------===//
// expression_evaluator
//===----------------------------------------------------------------------===//

expression_evaluator::expression_evaluator(
  duckdb::vector<std::unique_ptr<sirius::ast::node>> const& expressions,
  rmm::device_async_resource_ref resource_ref,
  rmm::cuda_stream_view stream,
  expression_evaluator_strategy strategy,
  std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  _ast_expressions.reserve(expressions.size());
  for (auto const& expr : expressions) {
    _ast_expressions.push_back(expr.get());
  }
}

expression_evaluator::expression_evaluator(sirius::ast::node const& expression,
                                           rmm::device_async_resource_ref resource_ref,
                                           rmm::cuda_stream_view stream,
                                           expression_evaluator_strategy strategy,
                                           std::size_t min_ast_size)
  : expression_evaluator(&expression, resource_ref, stream, strategy, min_ast_size)
{
}

expression_evaluator::expression_evaluator(sirius::ast::node const* expression,
                                           rmm::device_async_resource_ref resource_ref,
                                           rmm::cuda_stream_view stream,
                                           expression_evaluator_strategy strategy,
                                           std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  _ast_expressions.push_back(expression);
}

expression_evaluator::expression_evaluator(std::vector<sirius::ast::node const*> expressions,
                                           rmm::device_async_resource_ref resource_ref,
                                           rmm::cuda_stream_view stream,
                                           expression_evaluator_strategy strategy,
                                           std::size_t min_ast_size)
  : _ast_expressions(std::move(expressions)),
    _strategy(strategy),
    _mr(resource_ref),
    _stream(stream),
    _min_ast_size(min_ast_size)
{
}

std::unique_ptr<cudf::column> expression_evaluator::evaluate_ast(expr_ref root_expr)
{
  std::vector<cudf::column_view> combined_column_views;
  combined_column_views.reserve(_input_table.num_columns() + _temp_columns.size());
  for (int column_idx = 0; column_idx < _input_table.num_columns(); ++column_idx) {
    combined_column_views.push_back(_input_table.column(column_idx));
  }

  // We must be careful not to add invalidated temporary columns, as cuDF will throw when
  // constructing the table_view. Since the input table has a nonzero number of columns, we can
  // use the 0th column to produce a dummy column_view to maintain the integrity of the column
  // indices (note that this dummy column will never be referenced).
  for (auto const& temp_column : _temp_columns) {
    if (temp_column) {
      combined_column_views.push_back(temp_column->view());
    } else {
      combined_column_views.push_back(_input_table.column(0));
    }
  }

  cudf::table_view combined_table_view(combined_column_views);
  if (_strategy == expression_evaluator_strategy::AST_INTERPRET) {
    return cudf::compute_column(combined_table_view, root_expr.get(), _stream, _mr);
  } else {
    return cudf::compute_column_jit(combined_table_view, root_expr.get(), _stream, _mr);
  }
}

evaluate_result expression_evaluator::materialize_as_ast_column(
  std::unique_ptr<cudf::column> column)
{
  auto const col_idx         = _input_table.num_columns() + _temp_columns.size();
  auto const temp_column_idx = _temp_columns.size();
  _temp_columns.push_back(std::move(column));
  auto const& col_ref = _ast_tree.emplace<cudf::ast::column_reference>(col_idx);
  return evaluate_result(ast_result(col_ref, std::vector<std::size_t>{}, {temp_column_idx}));
}

void expression_evaluator::release_temporaries(std::vector<std::size_t> const& scalar_indices,
                                               std::vector<std::size_t> const& column_indices)
{
  for (auto const idx : scalar_indices) {
    _temp_scalars[idx].reset();
  }
  for (auto const idx : column_indices) {
    _temp_columns[idx].reset();
  }
}

void expression_evaluator::release_temporaries(
  std::vector<std::vector<std::size_t>> const& scalar_indices,
  std::vector<std::vector<std::size_t>> const& column_indices)
{
  for (auto const& vec : scalar_indices) {
    for (auto const idx : vec) {
      _temp_scalars[idx].reset();
    }
  }
  for (auto const& vec : column_indices) {
    for (auto const idx : vec) {
      _temp_columns[idx].reset();
    }
  }
}

std::unique_ptr<cudf::table> expression_evaluator::evaluate(cudf::table_view input)
{
  _output_columns.clear();
  _output_columns.reserve(_ast_expressions.size());

  // Reset AST state from any previous invocation so that the tree, temp scalars, and temp columns
  // do not accumulate stale nodes across calls.
  _ast_tree = cudf::ast::tree{};
  _temp_scalars.clear();
  _temp_columns.clear();

  // Get the table_view from the input_batch
  _input_table = std::move(input);

  // Per-result column post-processing. The node's kind and result type are read
  // natively from the Sirius AST (no DuckDB round-trip).
  auto post_process = [this](sirius::ast::node const& expr, evaluate_result result) {
    if (expr.holds<sirius::ast::reference>()) {
      // Bound reference: pass column through without type check
      _output_columns.push_back(
        std::make_unique<cudf::column>(result.get_column_view(), _stream, _mr));
    } else {
      // Cast the `result` from libcudf to the node's return type if `result` has a different
      // type. E.g., `extract(year from col)` from libcudf returns int16_t but the SQL type is
      // int64_t. Only use cudf::cast when both types are fixed-width (cast does not support
      // STRING/LIST/STRUCT).
      auto const cudf_return_type = sirius::get_cudf_type(expr.return_type());
      std::unique_ptr<cudf::column> result_column;
      if (result.is_scalar()) {
        result_column =
          cudf::make_column_from_scalar(result.get_scalar(), _input_table.num_rows(), _stream, _mr);
      } else if (result.is_column_view()) {
        result_column = std::make_unique<cudf::column>(result.get_column_view(), _stream, _mr);
      } else {
        result_column = result.release_column();
      }
      if (result_column->type().id() != cudf_return_type.id()) {
        // Cast is only valid for fixed-width types (no STRING/LIST/STRUCT/etc.).
        if (IsFixedWidth(result_column->type()) && IsFixedWidth(cudf_return_type)) {
          result_column = cudf::cast(result_column->view(), cudf_return_type, _stream, _mr);
        } else {
          throw internal_exception("[expression_evaluator] Unsupported type conversion: {} to {}",
                                   cudf::type_to_name(result_column->type()),
                                   cudf::type_to_name(cudf_return_type));
        }
      }
      _output_columns.push_back(std::move(result_column));
    }
  };

  // Iterate _ast_expressions and route through the std::visit dispatcher, then
  // post-process each result using the node's native kind + return type.
  for (auto const* ast_expr : _ast_expressions) {
    if (!ast_expr) {
      // This is a Sirius bug, not a user error: a null slot means from_duckdb
      // declined an expression as unsupported, but the planner still routed it
      // to the GPU executor.  This hard-fails the query instead of falling back to CPU.
      // TODO fix to ensure the planner never builds a GPU projection containing unsupported
      // expressions.
      throw internal_exception(
        "[expression_evaluator] null expression in select list — "
        "from_duckdb returned nullptr for an unsupported expression; "
        "cannot evaluate on GPU");
    }
    auto result = evaluate(*ast_expr, evaluation_mode::MATERIALIZE);
    post_process(*ast_expr, std::move(result));
  }

  return std::make_unique<cudf::table>(std::move(_output_columns));
}

std::unique_ptr<cudf::column> expression_evaluator::compute_mask(cudf::table_view input)
{
  D_ASSERT(_ast_expressions.size() == 1);
  // A filter predicate must be BOOLEAN. Read the type natively from the Sirius AST.
  D_ASSERT(_ast_expressions[0]->return_type().id() == sirius::type_id::BOOLEAN);

  return std::move(evaluate(input)->release()[0]);
}

std::unique_ptr<cudf::table> expression_evaluator::select(
  cudf::table_view input, std::span<cudf::size_type const> output_indices)
{
  // A cuDF table with zero columns cannot carry a row count, so a pure-filter result with no
  // output columns (e.g. count(*) over a filtered scan) is unrepresentable here. Such callers
  // must use the all-columns select() overload so the surviving row count is preserved.
  if (output_indices.empty()) {
    throw internal_exception(
      "[expression_evaluator] select(): output_indices must be non-empty; use the "
      "all-columns select() overload for count(*)-style filters with no output columns");
  }
  auto mask = compute_mask(input);
  return cudf::apply_boolean_mask(
    input.select(output_indices.begin(), output_indices.end()), mask->view(), _stream, _mr);
}

std::unique_ptr<cudf::table> expression_evaluator::select(cudf::table_view input)
{
  auto mask = compute_mask(input);
  return cudf::apply_boolean_mask(input, mask->view(), _stream, _mr);
}

evaluate_result expression_evaluator::evaluate(sirius::ast::aggregate const& /*expr*/,
                                               evaluation_mode /*mode*/)
{
  throw not_implemented_exception(
    "[expression_evaluator] aggregate nodes are not executed via the expression executor "
    "(#863).");
}

evaluate_result expression_evaluator::evaluate(sirius::ast::node const& expr, evaluation_mode mode)
{
  // Every sirius::ast alternative has a matching private evaluate(alt, mode)
  // overload, so overload resolution enforces dispatch completeness at compile
  // time (a new alternative without an overload fails to compile here).
  return std::visit(
    [this, mode](auto const& alt) -> evaluate_result { return this->evaluate(alt, mode); }, expr.v);
}

}  // namespace sirius
