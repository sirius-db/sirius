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
#include <expression_executor/gpu_dispatcher.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <expression_executor/gpu_expression_executor_state.hpp>
#include <gpu_buffer_manager.hpp>
#include <gpu_columns.hpp>
#include <operator/gpu_materialize.hpp>

// cucascade
#include <cucascade/data/gpu_data_representation.hpp>
#include <data/data_batch_utils.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/common/types.hpp>

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

namespace {
/// Returns true if cudf::cast supports this type (fixed-width only; no STRING/LIST/STRUCT/etc.).
bool IsFixedWidth(cudf::data_type const& type)
{
  auto const id = type.id();
  return id != cudf::type_id::STRING && id != cudf::type_id::LIST && id != cudf::type_id::STRUCT &&
         id != cudf::type_id::DICTIONARY32 && id != cudf::type_id::EMPTY;
}
}  // namespace

namespace sirius::experimental {
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
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> const& expressions,
  expression_executor_strategy strategy,
  rmm::device_async_resource_ref resource_ref,
  rmm::cuda_stream_view stream,
  std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  for (auto const& expr : expressions) {
    _expressions.push_back(expr.get());
  }
}

gpu_expression_executor::gpu_expression_executor(duckdb::Expression const* expression,
                                                 expression_executor_strategy strategy,
                                                 rmm::device_async_resource_ref resource_ref,
                                                 rmm::cuda_stream_view stream,
                                                 std::size_t min_ast_size)
  : _strategy(strategy), _mr(resource_ref), _stream(stream), _min_ast_size(min_ast_size)
{
  _expressions.push_back(expression);
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
  auto col_ref = _ast_tree.emplace<cudf::ast::column_reference>(col_idx);
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

std::shared_ptr<data_batch> gpu_expression_executor::execute(
  std::shared_ptr<data_batch> input_batch)
{
  if (!input_batch) { return input_batch; }
  D_ASSERT(!_expressions.empty());
  _output_columns.clear();
  _output_columns.reserve(_expressions.size());

  // Reset AST state from any previous invocation so that the tree, temp scalars, and temp columns
  // do not accumulate stale nodes across calls.
  _ast_tree = cudf::ast::tree{};
  _temp_scalars.clear();
  _temp_columns.clear();

  // Get the table_view from the input_batch
  auto const& input_rep = input_batch->get_data()->cast<cucascade::gpu_table_representation>();
  _input_table          = input_rep.get_table().view();

  // Execute the expressions and emit results into _output_columns
  for (auto& _expression : _expressions) {
    auto const& expr = *_expression;
    auto result      = execute(expr, execution_mode::MATERIALIZE);

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
  }

  // Create the data representation
  std::unique_ptr<cucascade::idata_representation> output_data_rep =
    std::make_unique<cucascade::gpu_table_representation>(
      std::make_unique<cudf::table>(std::move(_output_columns), _stream, _mr),
      *input_batch->get_memory_space());

  // Create the data batch and return
  auto const batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data_rep));
}

std::shared_ptr<data_batch> gpu_expression_executor::select(std::shared_ptr<data_batch> input_batch)
{
  D_ASSERT(_expressions.size() == 1);
  auto const& expr = *_expressions[0];
  D_ASSERT(expr.return_type == duckdb::LogicalType::BOOLEAN);

  // Call execute(input_batch) to set _input_table and produce the boolean mask as a single column
  auto mask_batch = execute(input_batch);
  auto& mask_repr = mask_batch->get_data()->cast<cucascade::gpu_table_representation>();
  auto mask_view  = mask_repr.get_table().view().column(0);

  // Apply the boolean mask to filter the input batch
  auto output_table = cudf::apply_boolean_mask(_input_table, mask_view, _stream, _mr);
  std::unique_ptr<cucascade::idata_representation> output_data_rep =
    std::make_unique<cucascade::gpu_table_representation>(std::move(output_table),
                                                          *input_batch->get_memory_space());

  // Create the data batch and return
  auto const batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data_rep));
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
      if (std::find(supported_ast_functions.begin(),
                    supported_ast_functions.end(),
                    func_expr.function.name) != supported_ast_functions.end()) {
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
        case duckdb::ExpressionType::OPERATOR_COALESCE:
          /// KEVIN: TODO
          throw duckdb::NotImplementedException(
            "[gpu_expression_executor] count_ast_ops called on an unsupported COALESCE operator "
            "expression.");
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
}  // namespace sirius::experimental

namespace duckdb {
namespace sirius {

GpuExpressionExecutor::GpuExpressionExecutor(const Expression& expr,
                                             rmm::device_async_resource_ref resource_ref)
  : resource_ref(resource_ref)
{
  AddExpression(expr);
}

GpuExpressionExecutor::GpuExpressionExecutor(const vector<unique_ptr<Expression>>& expressions,
                                             rmm::device_async_resource_ref resource_ref)
  : resource_ref(resource_ref)
{
  D_ASSERT(expressions.size() > 0);

  for (const auto& expr : expressions) {
    AddExpression(*expr);
  }
}

void GpuExpressionExecutor::AddExpression(const Expression& expr)
{
  // Add the given expression to the list of expressions this executor is responsible for
  expressions.push_back(&expr);

  // Initialize the executor state of the expression and add to the list of executor states
  auto state = std::make_unique<GpuExpressionExecutorState>();
  Initialize(expr, *state);
  states.push_back(std::move(state));
}

void GpuExpressionExecutor::ClearExpressions()
{
  states.clear();
  expressions.clear();
}

void GpuExpressionExecutor::Initialize(const Expression& expr, GpuExpressionExecutorState& state)
{
  // Set the executor of the executor state to this GpuExpressionExecutor
  state.executor = this;

  // Initialize the state of the expression
  state.root_state = InitializeState(expr, state);
}

void GpuExpressionExecutor::SetInputColumns(const GPUIntermediateRelation& input_relation)
{
  input_count           = 0;
  has_null_input_column = false;

  // Shallow copy the columns
  input_columns = input_relation.columns;

  // Set the input count
  if (input_columns.empty()) {
    input_count = 1;
  } else {
    // All columns that are not null should have the same count
    for (const auto& col : input_columns) {
      const auto temp_count = col == nullptr ? 0
                              : col->row_ids == nullptr
                                ? static_cast<cudf::size_type>(col->column_length)
                                : static_cast<cudf::size_type>(col->row_id_count);
      if (temp_count > 0) {
        input_count = temp_count;
      } else {
        has_null_input_column = true;
      }
    }
  }
}

// Helper template function for HasNullLeaf()
template <typename ExpressionT>
bool GpuExpressionExecutor::HasNullLeafLoop(const ExpressionT& expr) const
{
  for (const auto& child : expr.children) {
    if (HasNullLeaf(*child)) { return true; }
  }
  return false;
}

bool GpuExpressionExecutor::HasNullLeaf(const Expression& expr) const
{
  // Check if the expression is a null reference
  switch (expr.GetExpressionClass()) {
    case ExpressionClass::BOUND_BETWEEN: {
      const auto& between_expr = expr.Cast<BoundBetweenExpression>();
      return HasNullLeaf(*between_expr.input) || HasNullLeaf(*between_expr.lower) ||
             HasNullLeaf(*between_expr.upper);
    }
    case ExpressionClass::BOUND_CASE: {
      const auto& case_expr = expr.Cast<BoundCaseExpression>();
      for (const auto& case_check : case_expr.case_checks) {
        if (HasNullLeaf(*case_check.when_expr) || HasNullLeaf(*case_check.then_expr)) {
          return true;
        }
      }
      return HasNullLeaf(*case_expr.else_expr);
    }
    case ExpressionClass::BOUND_CAST: {
      const auto& cast_expr = expr.Cast<BoundCastExpression>();
      return HasNullLeaf(*cast_expr.child);
    }
    case ExpressionClass::BOUND_COMPARISON: {
      const auto& comp_expr = expr.Cast<BoundComparisonExpression>();
      return HasNullLeaf(*comp_expr.left) || HasNullLeaf(*comp_expr.right);
    }
    case ExpressionClass::BOUND_CONJUNCTION: {
      return HasNullLeafLoop(expr.Cast<BoundConjunctionExpression>());
    }
    case ExpressionClass::BOUND_CONSTANT: {
      // Base case
      return false;
    }
    case ExpressionClass::BOUND_FUNCTION: {
      return HasNullLeafLoop(expr.Cast<BoundFunctionExpression>());
    }
    case ExpressionClass::BOUND_OPERATOR: {
      return HasNullLeafLoop(expr.Cast<BoundOperatorExpression>());
    }
    case ExpressionClass::BOUND_REF: {
      // Base case
      const auto& ref_expr = expr.Cast<BoundReferenceExpression>();
      const auto& col      = input_columns[ref_expr.index];
      return col == nullptr || col->data_wrapper.data == nullptr;
    }
    default:
      throw InternalException("HasNullLeaf called on an expression [" + expr.ToString() +
                              "] with unsupported expression class!");
  }
  return false;
}

void GpuExpressionExecutor::Execute(const GPUIntermediateRelation& input_relation,
                                    GPUIntermediateRelation& output_relation,
                                    rmm::cuda_stream_view stream)
{
  D_ASSERT(expressions.size() == output_relation.columns.size());
  D_ASSERT(!expressions.empty());

  execution_stream = stream;
  SetInputColumns(input_relation);

  // Loop over expressions to execute
  for (idx_t i = 0; i < expressions.size(); ++i) {
    const auto& expr = *expressions[i];

    // If the expression is a reference, just pass it through
    if (expr.expression_class == ExpressionClass::BOUND_REF) {
      auto input_idx             = expr.Cast<BoundReferenceExpression>().index;
      output_relation.columns[i] = input_relation.columns[input_idx];
      continue;
    }

    // Make placeholder output column
    output_relation.columns[i] = make_shared_ptr<GPUColumn>(
      0, convertLogicalTypeToColumnType(expr.return_type), nullptr, nullptr);

    // Skip execution if the input count is zero or if there is a null leaf
    if (input_count == 0 || (has_null_input_column && HasNullLeaf(expr))) { continue; }

    // Otherwise, execute the expression
    auto result = ExecuteExpression(i);

    // Cast the `result` from libcudf to `return_type` if `result` has different types.
    // E.g., `extract(year from col)` from libcudf returns int16_t but duckdb requires int64_t
    // Only use cudf::cast when both types are fixed-width (cast does not support
    // STRING/LIST/STRUCT).
    auto cudf_return_type = GetCudfType(expressions[i]->return_type);
    if (result->type().id() != cudf_return_type.id()) {
      if (IsFixedWidth(result->type()) && IsFixedWidth(cudf_return_type)) {
        result = cudf::cast(result->view(), cudf_return_type, execution_stream, resource_ref);
      } else {
        throw InternalException("GpuExpressionExecutor: Unsupported type conversion: " +
                                cudf::type_to_name(result->type()) + " to " +
                                cudf::type_to_name(cudf_return_type));
      }
    }

    // Transfer to output relation (zero copy)
    output_relation.columns[i]->setFromCudfColumn(*result,
                                                  false,  // How to know?
                                                  nullptr,
                                                  0,
                                                  &GPUBufferManager::GetInstance());
  }
}

std::shared_ptr<cucascade::data_batch> GpuExpressionExecutor::execute(
  std::shared_ptr<cucascade::data_batch> input_batch, rmm::cuda_stream_view stream)
{
  assert(!expressions.empty());

  use_data_batch_apis = true;
  execution_stream    = stream;
  output_columns.clear();
  output_columns.resize(expressions.size());

  // Retrieve the table_view from the data_batch
  auto& input_data_rep = input_batch->get_data()->cast<cucascade::gpu_table_representation>();
  input_table          = input_data_rep.get_table().view();
  input_count          = static_cast<cudf::size_type>(input_table.num_rows());

  for (size_t i = 0; i < expressions.size(); ++i) {
    auto const& expr = *expressions[i];
    auto result      = ExecuteExpression(i);
    // BOUND_REF: pass column through without type check (same as Execute(GPUIntermediateRelation)).
    // The column is the actual input column; no cast is valid for string/non-fixed-width.
    if (expr.expression_class != ExpressionClass::BOUND_REF) {
      auto cudf_return_type = GetCudfType(expressions[i]->return_type);
      // Cast the `result` from libcudf to `return_type` if `result` has a different type.
      // E.g., `extract(year from col)` from libcudf returns int16_t but duckdb requires int64_t
      // Only use cudf::cast when both types are fixed-width (cast does not support
      // STRING/LIST/STRUCT).
      if (result->type().id() != cudf_return_type.id()) {
        if (IsFixedWidth(result->type()) && IsFixedWidth(cudf_return_type)) {
          result = cudf::cast(result->view(), cudf_return_type, execution_stream, resource_ref);
        } else {
          throw InternalException("GpuExpressionExecutor: Unsupported type conversion: " +
                                  cudf::type_to_name(result->type()) + " to " +
                                  cudf::type_to_name(cudf_return_type));
        }
      }
    }
    output_columns[i] = std::move(result);
  }

  // Create the data representation
  std::unique_ptr<cucascade::idata_representation> output_data_rep =
    std::make_unique<cucascade::gpu_table_representation>(
      std::move(
        std::make_unique<cudf::table>(std::move(output_columns), execution_stream, resource_ref)),
      *input_batch->get_memory_space());

  // Create the data batch and return
  auto const batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data_rep));
}

void GpuExpressionExecutor::Select(GPUIntermediateRelation& input_relation,
                                   GPUIntermediateRelation& output_relation,
                                   rmm::cuda_stream_view stream)
{
  D_ASSERT(expressions.size() == 1);
  D_ASSERT(expressions[0]->return_type == LogicalType::BOOLEAN);

  execution_stream = stream;
  SetInputColumns(input_relation);

  // If the input count is zero or if there is a null leaf, just materialize
  if (input_count == 0 || (has_null_input_column && HasNullLeaf(*expressions[0]))) {
    HandleMaterializeRowIDs(
      input_relation, output_relation, 0, nullptr, &GPUBufferManager::GetInstance(), true);
    return;
  }

  // Execute the boolean expression
  auto bitmap = ExecuteExpression(0);

  // Need to convert `null` values in bitmap to `false`, to be consistent with SQL `where` clause
  if (bitmap->null_count() > 0) {
    cudf::numeric_scalar<bool> false_scalar(false, execution_stream, resource_ref);
    bitmap = cudf::replace_nulls(bitmap->view(), false_scalar, execution_stream, resource_ref);
  }

  // Generate the selection vector
  auto [row_ids, count] = GpuDispatcher::DispatchSelect(bitmap->view(), resource_ref);

  // Compact
  HandleMaterializeRowIDs(
    input_relation, output_relation, count, row_ids, &GPUBufferManager::GetInstance(), true);
}

std::shared_ptr<cucascade::data_batch> GpuExpressionExecutor::select(
  std::shared_ptr<cucascade::data_batch> input_batch, rmm::cuda_stream_view stream)
{
  assert(expressions.size() == 1);
  assert(expressions[0]->return_type == LogicalType::BOOLEAN);

  use_data_batch_apis = true;
  execution_stream    = stream;
  output_columns.clear();

  // Retrieve the table_view from the data_batch
  auto& input_data_rep = input_batch->get_data()->cast<cucascade::gpu_table_representation>();
  input_table          = input_data_rep.get_table().view();
  input_count          = static_cast<cudf::size_type>(input_table.num_rows());

  // Get the bitmap
  auto bitmap = ExecuteExpression(0);

  // (debug traces removed)

  // Apply the bitmap
  auto output_table =
    cudf::apply_boolean_mask(input_table, bitmap->view(), execution_stream, resource_ref);
  std::unique_ptr<cucascade::idata_representation> output_data_rep =
    std::make_unique<cucascade::gpu_table_representation>(std::move(output_table),
                                                          *input_batch->get_memory_space());

  // Create the data batch and return
  auto const batch_id = ::sirius::get_next_batch_id();
  return std::make_shared<cucascade::data_batch>(batch_id, std::move(output_data_rep));
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::ExecuteExpression(idx_t expr_idx)
{
  D_ASSERT(expr_idx < expressions.size());

  return Execute(*expressions[expr_idx], states[expr_idx]->root_state.get());
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const Expression& expr,
                                                             GpuExpressionState* state)
{
  switch (expr.GetExpressionClass()) {
    case ExpressionClass::BOUND_BETWEEN: return Execute(expr.Cast<BoundBetweenExpression>(), state);
    case ExpressionClass::BOUND_CASE: return Execute(expr.Cast<BoundCaseExpression>(), state);
    case ExpressionClass::BOUND_CAST: return Execute(expr.Cast<BoundCastExpression>(), state);
    case ExpressionClass::BOUND_COMPARISON:
      return Execute(expr.Cast<BoundComparisonExpression>(), state);
    case ExpressionClass::BOUND_CONJUNCTION:
      return Execute(expr.Cast<BoundConjunctionExpression>(), state);
    case ExpressionClass::BOUND_CONSTANT:
      return Execute(expr.Cast<BoundConstantExpression>(), state);
    case ExpressionClass::BOUND_FUNCTION:
      return Execute(expr.Cast<BoundFunctionExpression>(), state);
    case ExpressionClass::BOUND_OPERATOR:
      return Execute(expr.Cast<BoundOperatorExpression>(), state);
    case ExpressionClass::BOUND_PARAMETER:
      throw NotImplementedException("Execute[BOUND_PARAMETER]: Not yet implemented!");
    case ExpressionClass::BOUND_REF: return Execute(expr.Cast<BoundReferenceExpression>(), state);
    default:
      throw InternalException("Execute called on an expression [" + expr.ToString() +
                              "] with unsupported expression class!");
  }
}

std::unique_ptr<GpuExpressionState> GpuExpressionExecutor::InitializeState(
  const Expression& expr, GpuExpressionExecutorState& state)
{
  switch (expr.GetExpressionClass()) {
    case ExpressionClass::BOUND_BETWEEN:
      return InitializeState(expr.Cast<BoundBetweenExpression>(), state);
    case ExpressionClass::BOUND_CASE:
      return InitializeState(expr.Cast<BoundCaseExpression>(), state);
    case ExpressionClass::BOUND_CAST:
      return InitializeState(expr.Cast<BoundCastExpression>(), state);
    case ExpressionClass::BOUND_COMPARISON:
      return InitializeState(expr.Cast<BoundComparisonExpression>(), state);
    case ExpressionClass::BOUND_CONJUNCTION:
      return InitializeState(expr.Cast<BoundConjunctionExpression>(), state);
    case ExpressionClass::BOUND_CONSTANT:
      return InitializeState(expr.Cast<BoundConstantExpression>(), state);
    case ExpressionClass::BOUND_FUNCTION:
      return InitializeState(expr.Cast<BoundFunctionExpression>(), state);
    case ExpressionClass::BOUND_OPERATOR:
      return InitializeState(expr.Cast<BoundOperatorExpression>(), state);
    case ExpressionClass::BOUND_PARAMETER:
      throw NotImplementedException("InitializeState[BOUND_PARAMETER]: Not yet implemented!");
    case ExpressionClass::BOUND_REF:
      return InitializeState(expr.Cast<BoundReferenceExpression>(), state);
    default: throw InternalException("InitializeState: Unknown ExpressionClass!");
  }
}

}  // namespace sirius
}  // namespace duckdb
