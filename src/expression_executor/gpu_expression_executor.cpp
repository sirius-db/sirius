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

#include <expression/ast/node.hpp>       // sirius::ast::node + 11 alternatives
#include <expression/ast/to_duckdb.hpp>  // sirius::ast::to_duckdb (per-alternative overloads + node dispatcher)
#include <expression/expression_internal.hpp>
#include <expression/function_id.hpp>
#include <expression_executor/ast_supported_types.hpp>
#include <expression_executor/gpu_expression_executor.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <data/data_batch_utils.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/expression/bound_between_expression.hpp>
#include <duckdb/planner/expression/bound_case_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

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
using data_batch     = cucascade::data_batch;
using execute_result = gpu_expression_executor::execute_result;
using ast_result     = gpu_expression_executor::ast_result;

//===----------------------------------------------------------------------===//
// ast_result
//===----------------------------------------------------------------------===//

gpu_expression_executor::ast_result::ast_result(
  expr_ref e,
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
// execute_result
//===----------------------------------------------------------------------===//

std::reference_wrapper<cudf::ast::expression const>
gpu_expression_executor::execute_result::get_expr() const
{
  if (!is_ast()) {
    throw std::runtime_error("[execute_result] Attempted to get expr from materialized result");
  }
  return std::get<ast_result>(payload).expr;
}

std::vector<std::size_t> gpu_expression_executor::execute_result::get_temp_scalar_indices() const
{
  if (is_ast()) { return std::get<ast_result>(payload).temp_scalar_indices; }
  return {};
}

std::vector<std::size_t> gpu_expression_executor::execute_result::get_temp_column_indices() const
{
  if (is_ast()) { return std::get<ast_result>(payload).temp_column_indices; }
  return {};
}

cudf::scalar const& gpu_expression_executor::execute_result::get_scalar() const
{
  if (!is_scalar()) {
    throw std::runtime_error(
      "[execute_result] Attempted to get scalar from non-scalar execute_result");
  }
  auto const& scalar_payload = std::get<std::unique_ptr<cudf::scalar>>(payload);
  return *scalar_payload;
}

cudf::column_view gpu_expression_executor::execute_result::get_column_view() const
{
  if (is_ast() || is_scalar()) {
    throw std::runtime_error(
      "[execute_result] Attempted to get column view from non-column execute_result");
  }
  if (std::holds_alternative<cudf::column_view>(payload)) {
    return std::get<cudf::column_view>(payload);
  }
  auto const& column_payload = std::get<std::unique_ptr<cudf::column>>(payload);
  return column_payload->view();
}

std::unique_ptr<cudf::column> gpu_expression_executor::execute_result::release_column()
{
  if (std::holds_alternative<std::unique_ptr<cudf::column>>(payload)) {
    return std::move(std::get<std::unique_ptr<cudf::column>>(payload));
  }
  throw std::runtime_error(
    "[execute_result] Attempted to get column from execute_result that does not hold a column");
}

//===----------------------------------------------------------------------===//
// gpu_expression_executor
//===----------------------------------------------------------------------===//

gpu_expression_executor::gpu_expression_executor(
  duckdb::vector<sirius::expression> const& expressions,
  rmm::device_async_resource_ref resource_ref,
  rmm::cuda_stream_view stream,
  expression_executor_strategy strategy,
  std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  _expressions.reserve(expressions.size());
  for (auto const& expr : expressions) {
    _expressions.push_back(sirius::unwrap(expr));
  }
}

gpu_expression_executor::gpu_expression_executor(sirius::expression const& expression,
                                                 rmm::device_async_resource_ref resource_ref,
                                                 rmm::cuda_stream_view stream,
                                                 expression_executor_strategy strategy,
                                                 std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  _expressions.push_back(sirius::unwrap(expression));
}

gpu_expression_executor::gpu_expression_executor(duckdb::Expression const* expression,
                                                 rmm::device_async_resource_ref resource_ref,
                                                 rmm::cuda_stream_view stream,
                                                 expression_executor_strategy strategy,
                                                 std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  _expressions.push_back(expression);
}

gpu_expression_executor::gpu_expression_executor(sirius::ast::node const* expression,
                                                 rmm::device_async_resource_ref resource_ref,
                                                 rmm::cuda_stream_view stream,
                                                 expression_executor_strategy strategy,
                                                 std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  _ast_expressions.push_back(expression);
}

std::unique_ptr<cudf::column> gpu_expression_executor::execute_ast(expr_ref root_expr)
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
  if (_strategy == expression_executor_strategy::AST_INTERPRET) {
    return cudf::compute_column(combined_table_view, root_expr.get(), _stream, _mr);
  } else {
    return cudf::compute_column_jit(combined_table_view, root_expr.get(), _stream, _mr);
  }
}

execute_result gpu_expression_executor::materialize_as_ast_column(
  std::unique_ptr<cudf::column> column)
{
  auto const col_idx         = _input_table.num_columns() + _temp_columns.size();
  auto const temp_column_idx = _temp_columns.size();
  _temp_columns.push_back(std::move(column));
  auto const& col_ref = _ast_tree.emplace<cudf::ast::column_reference>(col_idx);
  return execute_result(ast_result(col_ref, std::vector<std::size_t>{}, {temp_column_idx}));
}

void gpu_expression_executor::release_temporaries(std::vector<std::size_t> const& scalar_indices,
                                                  std::vector<std::size_t> const& column_indices)
{
  for (auto const idx : scalar_indices) {
    _temp_scalars[idx].reset();
  }
  for (auto const idx : column_indices) {
    _temp_columns[idx].reset();
  }
}

void gpu_expression_executor::release_temporaries(
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

std::unique_ptr<cudf::table> gpu_expression_executor::execute(cudf::table_view input)
{
  D_ASSERT(_expressions.empty() != _ast_expressions.empty());
  _output_columns.clear();
  _output_columns.reserve(std::max(_expressions.size(), _ast_expressions.size()));

  // Reset AST state from any previous invocation so that the tree, temp scalars, and temp columns
  // do not accumulate stale nodes across calls.
  _ast_tree = cudf::ast::tree{};
  _temp_scalars.clear();
  _temp_columns.clear();

  // Get the table_view from the input_batch
  _input_table = std::move(input);

  // Per-result column post-processing — shared between the DuckDB and the
  // AST branches below. The DuckDB-typed expression is the parity reference;
  // the AST branch round-trips through sirius::ast::to_duckdb to recover the
  // expression_class + return_type fields for the same post-processing path.
  auto post_process = [this](duckdb::Expression const& expr, execute_result result) {
    if (expr.expression_class == duckdb::ExpressionClass::BOUND_REF) {
      // BOUND_REF: pass column through without type check
      _output_columns.push_back(
        std::make_unique<cudf::column>(result.get_column_view(), _stream, _mr));
    } else {
      // Cast the `result` from libcudf to `return_type` if `result` has a different type.
      // E.g., `extract(year from col)` from libcudf returns int16_t but duckdb requires int64_t
      // Only use cudf::cast when both types are fixed-width (cast does not support
      // STRING/LIST/STRUCT).
      auto const cudf_return_type = GetCudfType(expr.return_type);
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
          throw duckdb::InternalException(
            "[gpu_expression_executor] Unsupported type conversion: " +
            cudf::type_to_name(result_column->type()) + " to " +
            cudf::type_to_name(cudf_return_type));
        }
      }
      _output_columns.push_back(std::move(result_column));
    }
  };

  if (!_ast_expressions.empty()) {
    // AST path: iterate _ast_expressions and route through the std::visit dispatcher.
    // The per-result post-processing recovers expression_class + return_type by
    // round-tripping through sirius::ast::to_duckdb so the column post-processing
    // matches the DuckDB branch exactly.
    for (auto const* ast_expr : _ast_expressions) {
      auto result    = execute(*ast_expr, execution_mode::MATERIALIZE);
      auto duck_expr = sirius::ast::to_duckdb(*ast_expr);
      post_process(*duck_expr, std::move(result));
    }
  } else {
    // DuckDB path: existing call sites unchanged.
    for (auto& _expression : _expressions) {
      auto const& expr = *_expression;
      auto result      = execute(expr, execution_mode::MATERIALIZE);
      post_process(expr, std::move(result));
    }
  }

  return std::make_unique<cudf::table>(std::move(_output_columns), _stream, _mr);
}

std::unique_ptr<cudf::table> gpu_expression_executor::select(cudf::table_view input)
{
  D_ASSERT(_expressions.empty() != _ast_expressions.empty());
  D_ASSERT(_expressions.size() + _ast_expressions.size() == 1);
  if (!_ast_expressions.empty()) {
    // AST branch: round-trip the single node through sirius::ast::to_duckdb so the
    // BOOLEAN return-type assertion matches the DuckDB branch exactly (debug-only).
    auto duck_expr = sirius::ast::to_duckdb(*_ast_expressions[0]);
    D_ASSERT(duck_expr->return_type == duckdb::LogicalType::BOOLEAN);
  } else {
    auto const& expr = *_expressions[0];
    D_ASSERT(expr.return_type == duckdb::LogicalType::BOOLEAN);
  }

  // Call execute(input_batch) to set _input_table and produce the boolean mask as a single column
  auto mask_batch = execute(input);
  auto mask_view  = mask_batch->view().column(0);

  // Apply the boolean mask to filter the input batch
  return cudf::apply_boolean_mask(input, mask_view, _stream, _mr);
}

execute_result gpu_expression_executor::execute(duckdb::Expression const& expr, execution_mode mode)
{
  switch (expr.GetExpressionClass()) {
    case duckdb::ExpressionClass::BOUND_BETWEEN:
      return execute(expr.Cast<duckdb::BoundBetweenExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_CASE:
      return execute(expr.Cast<duckdb::BoundCaseExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_CAST:
      return execute(expr.Cast<duckdb::BoundCastExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_CONSTANT:
      return execute(expr.Cast<duckdb::BoundConstantExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_COMPARISON:
      return execute(expr.Cast<duckdb::BoundComparisonExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_CONJUNCTION:
      return execute(expr.Cast<duckdb::BoundConjunctionExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_FUNCTION:
      return execute(expr.Cast<duckdb::BoundFunctionExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_OPERATOR:
      return execute(expr.Cast<duckdb::BoundOperatorExpression>(), mode);
    case duckdb::ExpressionClass::BOUND_REF:
      return execute(expr.Cast<duckdb::BoundReferenceExpression>(), mode);
    default:
      throw duckdb::InternalException(
        "[gpu_expression_executor] execute called on an expression [{}] with unsupported "
        "expression class: {}",
        expr.ToString(),
        static_cast<int>(expr.GetExpressionClass()));
  }
}

std::size_t gpu_expression_executor::count_ast_ops(duckdb::Expression const& expr) const
{
  switch (expr.GetExpressionClass()) {
    case duckdb::ExpressionClass::BOUND_BETWEEN: {
      auto const& between_expr = expr.Cast<duckdb::BoundBetweenExpression>();
      // 2 comparison ops + 1 AND op + the ops in the children
      return 3 + count_ast_ops(*between_expr.input) + count_ast_ops(*between_expr.lower) +
             count_ast_ops(*between_expr.upper);
    }
    case duckdb::ExpressionClass::BOUND_CASE: {
      // No AST support
      return 0;
    }
    case duckdb::ExpressionClass::BOUND_CAST: {
      auto const& cast_expr = expr.Cast<duckdb::BoundCastExpression>();
      if (std::find(supported_ast_cast_types.begin(),
                    supported_ast_cast_types.end(),
                    expr.return_type.id()) != supported_ast_cast_types.end()) {
        return 1 + count_ast_ops(*cast_expr.child);
      }
      return 0;
    }
    case duckdb::ExpressionClass::BOUND_COMPARISON: {
      auto const& comp_expr = expr.Cast<duckdb::BoundComparisonExpression>();
      return 1 + count_ast_ops(*comp_expr.left) + count_ast_ops(*comp_expr.right);
    }
    case duckdb::ExpressionClass::BOUND_CONJUNCTION: {
      auto const& conj_expr = expr.Cast<duckdb::BoundConjunctionExpression>();
      auto count =
        conj_expr.children.size() - 1;  // Number of AND/OR ops is one less than number of children
      for (const auto& child : conj_expr.children) {
        count += count_ast_ops(*child);
      }
      return count;
    }
    case duckdb::ExpressionClass::BOUND_CONSTANT: {
      // Base case
      return 0;
    }
    case duckdb::ExpressionClass::BOUND_FUNCTION: {
      auto const& func_expr = expr.Cast<duckdb::BoundFunctionExpression>();

      /// First we check if the output type is decimal, since cuDF ASTs choke on intermediate
      /// decimal results currently.
      /// TODO: Fix when the following bug fix is in:
      /// https://github.com/rapidsai/cudf/pull/21996
      if (func_expr.return_type.id() == duckdb::LogicalTypeId::DECIMAL) { return 0; }

      // Check the set of supported AST functions. If the function is supported, count 1 for the
      // function itself plus the ops in the children.
      auto const resolved_id_opt = sirius::from_duckdb_function_name(func_expr.function.name);
      if (resolved_id_opt.has_value() &&
          std::find(supported_ast_functions.begin(),
                    supported_ast_functions.end(),
                    *resolved_id_opt) != supported_ast_functions.end()) {
        std::size_t count = 1;
        for (const auto& child : func_expr.children) {
          count += count_ast_ops(*child);
        }
        return count;
      }
      return 0;
    }
    case duckdb::ExpressionClass::BOUND_OPERATOR: {
      auto const& op_expr = expr.Cast<duckdb::BoundOperatorExpression>();
      switch (op_expr.type) {
        case duckdb::ExpressionType::COMPARE_IN:  // Fallthrough
        case duckdb::ExpressionType::COMPARE_NOT_IN: {
          auto const num_ors = op_expr.children.size() - 2;
          auto const num_eqs = op_expr.children.size() - 1;
          auto count =
            num_ors + num_eqs + (op_expr.type == duckdb::ExpressionType::COMPARE_NOT_IN ? 1 : 0);
          for (const auto& child : op_expr.children) {
            count += count_ast_ops(*child);
          }
          return count;
        }
        case duckdb::ExpressionType::OPERATOR_COALESCE: return 0;
        case duckdb::ExpressionType::OPERATOR_TRY:
          throw duckdb::NotImplementedException(
            "[gpu_expression_executor] count_ast_ops called on an unsupported TRY operator "
            "expression.");
        case duckdb::ExpressionType::OPERATOR_NOT: return 1 + count_ast_ops(*op_expr.children[0]);
        case duckdb::ExpressionType::OPERATOR_IS_NULL:
          return 1 + count_ast_ops(*op_expr.children[0]);
        case duckdb::ExpressionType::OPERATOR_IS_NOT_NULL:
          return 2 + count_ast_ops(*op_expr.children[0]);
        default:
          throw duckdb::InternalException(
            "[gpu_expression_executor] count_ast_ops called on an operator expression [{}] with "
            "unsupported operator type: {}",
            op_expr.ToString(),
            static_cast<int>(op_expr.type));
      }
    }
    case duckdb::ExpressionClass::BOUND_REF: {
      // Base case
      return 0;
    }
    default:
      throw duckdb::InternalException(
        "[gpu_expression_executor] count_ast_ops called on an expression [{}] with unsupported "
        "expression class: {}",
        expr.ToString(),
        static_cast<int>(expr.GetExpressionClass()));
  }
}

execute_result gpu_expression_executor::execute(sirius::ast::node const& expr, execution_mode mode)
{
  return std::visit(
    [this, mode](auto const& alt) -> execute_result { return this->execute(alt, mode); }, expr.v);
}

execute_result gpu_expression_executor::execute(sirius::ast::reference const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::reference shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::constant const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::constant shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::comparison const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::comparison shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::conjunction const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::conjunction shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::between const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::between shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::case_expr const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::case_expr shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::cast const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::cast shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::unary_op const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::unary_op shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::coalesce const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::coalesce shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::in_list const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::in_list shim not yet implemented");
}

execute_result gpu_expression_executor::execute(sirius::ast::function_call const& /*alt*/,
                                                execution_mode /*mode*/)
{
  throw duckdb::NotImplementedException(
    "[gpu_expression_executor] sirius::ast::function_call shim not yet implemented");
}

std::size_t gpu_expression_executor::count_ast_ops(sirius::ast::node const& expr) const
{
  // Round-trip shim — the per-specialization migration phase
  // (https://github.com/sirius-db/sirius/issues/699) replaces this with native
  // AST traversal one specialization at a time.
  return count_ast_ops(*sirius::ast::to_duckdb(expr));
}

}  // namespace sirius
