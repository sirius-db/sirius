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

#pragma once

// sirius
#include <config.hpp>
#include <expression_executor/gpu_expression_executor_state.hpp>
#include <gpu_buffer_manager.hpp>
#include <gpu_columns.hpp>

// duckdb
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

// cucascades
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository_manager.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// standard library
#include <array>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

namespace sirius::experimental {

static std::array<duckdb::LogicalTypeId, 3> constexpr supported_ast_cast_types{
  {duckdb::LogicalTypeId::UBIGINT, duckdb::LogicalTypeId::BIGINT, duckdb::LogicalTypeId::DOUBLE}};
static std::array<std::string_view, 6> constexpr supported_ast_functions{
  "+", "-", "*", "/", "//", "%"};

class gpu_expression_executor {
  using expr_ref = std::reference_wrapper<cudf::ast::expression const>;

 public:
  using data_batch              = cucascade::data_batch;
  using data_repository_manager = cucascade::data_repository_manager<std::shared_ptr<data_batch>>;

  struct ast_result {
    expr_ref expr;
    std::vector<std::size_t> temp_scalar_indices;  ///< The indices of the temp scalars that need to
                                                   ///< be kept alive for this AST expression.
    std::vector<std::size_t> temp_column_indices;  ///< The indices of the temp columns that need to
                                                   ///< be kept alive for this AST expression.

    ast_result(expr_ref e) : expr(e) {}
    ast_result(expr_ref e,
               std::vector<std::size_t> scalar_indices,
               std::vector<std::size_t> column_indices)
      : expr(e),
        temp_scalar_indices(std::move(scalar_indices)),
        temp_column_indices(std::move(column_indices))
    {
    }
    ast_result(expr_ref e,
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
  };

  struct execute_result {
    std::variant<ast_result,
                 cudf::column_view,
                 std::unique_ptr<cudf::scalar>,
                 std::unique_ptr<cudf::column>>
      payload;

    //===----------Constructors----------===//
    execute_result() = delete;
    execute_result(ast_result ast_payload) : payload(std::move(ast_payload)) {}
    execute_result(cudf::column_view column_view_payload) : payload(column_view_payload) {}
    execute_result(std::unique_ptr<cudf::scalar> scalar_payload)
      : payload(std::move(scalar_payload))
    {
    }
    execute_result(std::unique_ptr<cudf::column> column_payload)
      : payload(std::move(column_payload))
    {
    }

    [[nodiscard]] bool is_ast() const { return std::holds_alternative<ast_result>(payload); }
    [[nodiscard]] expr_ref get_expr() const
    {
      if (!is_ast()) {
        throw std::runtime_error("[execute_result] Attempted to get expr from materialized result");
      }
      return std::get<ast_result>(payload).expr;
    }
    [[nodiscard]] std::vector<std::size_t> get_temp_scalar_indices() const
    {
      if (is_ast()) { return std::get<ast_result>(payload).temp_scalar_indices; }
      return {};
    }
    [[nodiscard]] std::vector<std::size_t> get_temp_column_indices() const
    {
      if (is_ast()) { return std::get<ast_result>(payload).temp_column_indices; }
      return {};
    }
    [[nodiscard]] bool is_scalar() const
    {
      return std::holds_alternative<std::unique_ptr<cudf::scalar>>(payload);
    }
    [[nodiscard]] cudf::scalar const& get_scalar() const
    {
      if (!is_scalar()) {
        throw std::runtime_error(
          "[execute_result] Attempted to get scalar from non-scalar execute_result");
      }
      auto const& scalar_payload = std::get<std::unique_ptr<cudf::scalar>>(payload);
      return *scalar_payload;
    }
    [[nodiscard]] cudf::column_view get_column_view() const
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
  };

  enum class execution_mode {
    AST,
    MATERIALIZE,
  };

  gpu_expression_executor(rmm::device_async_resource_ref = cudf::get_current_device_resource_ref(),
                          rmm::cuda_stream_view stream   = cudf::get_default_stream(),
                          std::size_t min_ast_size       = 2);

  std::shared_ptr<data_batch> execute(std::shared_ptr<data_batch> input_batch,
                                      duckdb::Expression const& expr);

 private:
  rmm::device_async_resource_ref _mr;
  rmm::cuda_stream_view _stream;
  std::size_t _min_ast_size;
  cudf::table_view _input_table;

  cudf::ast::tree _ast_tree;
  std::vector<std::unique_ptr<cudf::scalar>> _temp_scalars;
  std::vector<std::unique_ptr<cudf::column>> _temp_columns;

  std::unique_ptr<cudf::column> execute_ast(expr_ref root_expr)
  {
    cudf::table temp_table(std::move(_temp_columns));
    cudf::table_view combined_table_view({_input_table, temp_table.view()});
    auto result_column =
      cudf::compute_column_jit(combined_table_view, root_expr.get(), _stream, _mr);
    _temp_columns = temp_table.release();
    return std::move(result_column);
  };

  void release_temporaries(std::vector<std::size_t> const& scalar_indices,
                           std::vector<std::size_t> const& column_indices)
  {
    for (auto const idx : scalar_indices) {
      _temp_scalars[idx].reset();
    }
    for (auto const idx : column_indices) {
      _temp_columns[idx].reset();
    }
  }
  void release_temporaries(std::vector<std::vector<std::size_t>> const& scalar_indices,
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

  execute_result execute(duckdb::Expression const& expr, execution_mode mode = execution_mode::AST);

  // Leaf expression nodes
  execute_result execute(duckdb::BoundReferenceExpression const& expr, execution_mode mode);
  execute_result execute(duckdb::BoundConstantExpression const& expr, execution_mode mode);

  // Interior expression nodes
  execute_result execute(duckdb::BoundBetweenExpression const& expr, execution_mode mode);
  execute_result execute(duckdb::BoundCaseExpression const& expr, execution_mode mode);
  execute_result execute(duckdb::BoundCastExpression const& expr, execution_mode mode);
  execute_result execute(duckdb::BoundComparisonExpression const& expr, execution_mode mode);
  execute_result execute(duckdb::BoundConjunctionExpression const& expr, execution_mode mode);
  execute_result execute(duckdb::BoundFunctionExpression const& expr, execution_mode mode);
  execute_result execute(duckdb::BoundOperatorExpression const& expr, execution_mode mode);

  [[nodiscard]] std::size_t count_ast_ops(duckdb::Expression const& expr) const;
};

}  // namespace sirius::experimental

namespace duckdb {
namespace sirius {

//===----------------------------------------------------------------------===//
// GpuExpressionExecutor
//===----------------------------------------------------------------------===//

/**
 * @brief The GpuExpressionExecutor is responsible for evaluating expressions on the GPU.
 */
struct GpuExpressionExecutor {
  using data_batch              = cucascade::data_batch;
  using data_repository_manager = cucascade::data_repository_manager<std::shared_ptr<data_batch>>;

  //===----------Constructor/Destructor(s)----------===//
  /**
   * @brief Constructs an expression executor with a single expression
   *
   * @param expr The expression to evaluate
   * @param resource_ref The rmm::device_async_resource_ref to pass to cudf APIs for allocations
   */
  GpuExpressionExecutor(
    const Expression& expr,
    rmm::device_async_resource_ref resource_ref = cudf::get_current_device_resource_ref());

  /**
   * @brief Constructs an expression executor with a set of expressions
   *
   * @param expressions The expressions to evaluate
   * @param resource_ref The rmm::device_async_resource_ref to pass to cudf APIs for allocations
   */
  GpuExpressionExecutor(
    const vector<unique_ptr<Expression>>& expressions,
    rmm::device_async_resource_ref resource_ref = cudf::get_current_device_resource_ref());

  //===----------Fields----------===//
  std::vector<const Expression*> expressions;  ///< The expressions to execute
  std::vector<std::unique_ptr<GpuExpressionExecutorState>>
    states;  ///< The execution states associated with each expression to execute
  std::vector<shared_ptr<GPUColumn>>
    input_columns;                              ///< The input columns for expression evaluation
  rmm::device_async_resource_ref resource_ref;  ///< The allocator to pass to cudf APIs
  cudf::size_type input_count;                  ///< The row count of the input table
  bool has_null_input_column;                   ///< Whether some input column is null
  rmm::cuda_stream_view execution_stream;       ///< THe stream in which to execute operations

  //===----------Fields for New Execution Model----------===//
  bool use_data_batch_apis =
    false;  ///< Whether to use the data_batch APIs in executing bound references
  std::vector<std::unique_ptr<cudf::column>>
    output_columns;              ///< The columns generated by the executed expressions
  cudf::table_view input_table;  ///< The input table

  //===----------Methods----------===//
  void AddExpression(const Expression& expr);
  void ClearExpressions();

  // Set the root state of the executor to the given expression
  void Initialize(const Expression& expr, GpuExpressionExecutorState& state);

  // Set the input count and columns for the expression executor
  void SetInputColumns(const GPUIntermediateRelation& input_relation);

  // Before evaluating an expression, check the leaves for nullptrs
  // (Assumes the input columns have already been set)
  [[nodiscard]] bool HasNullLeaf(const Expression& expr) const;
  template <typename ExpressionT>
  bool HasNullLeafLoop(const ExpressionT& expr) const;

  // Execute the set of expressions with the given input relation and store the result in the
  // output relation (Provides the main interface with client code for Projections).
  void Execute(const GPUIntermediateRelation& input_relation,
               GPUIntermediateRelation& output_relation,
               rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  // Execute the set of expressions with the given input relation and compact into the output
  // relation based on the resulting selection vector (Provides the main interface with client
  // code for Filters).
  void Select(GPUIntermediateRelation& input_relation,
              GPUIntermediateRelation& output_relation,
              rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Executes the current set of expressions against the given input batch and emits a new
   * output batch holding the results.
   *
   * @param input_batch The input batch against which to evaluate expressions
   * @param stream The stream in which to execute the operations in the expression tree
   *
   * @return std::shared_ptr<cucascade::data_batch> The result of the evaluated expressions
   *
   * @note It is required that there is only one boolean expression in the current expression set.
   */
  std::shared_ptr<data_batch> execute(std::shared_ptr<data_batch> input_batch,
                                      rmm::cuda_stream_view stream);

  /**
   * @brief Evaluates a boolean expression and filters the input batch according to the result.
   *
   * @param input_batch The input batch against which to evaluate the expression
   * @param stream The stream in which to execute the operations in the expression tree
   *
   * @return std::shared_ptr<cucascade::data_batch> The input batch filtered by the boolean
   * expression
   */
  std::shared_ptr<cucascade::data_batch> select(std::shared_ptr<data_batch> input_batch,
                                                rmm::cuda_stream_view stream);

  // Execute the expression at the given index and return the result
  std::unique_ptr<cudf::column> ExecuteExpression(idx_t expression_idx);

  //----------Execute + Specializations----------//
  std::unique_ptr<cudf::column> Execute(const Expression& expr, GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundBetweenExpression& expr,
                                        GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundCaseExpression& expr, GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundCastExpression& expr, GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundComparisonExpression& expr,
                                        GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundConjunctionExpression& expr,
                                        GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundConstantExpression& expr,
                                        GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundFunctionExpression& expr,
                                        GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundOperatorExpression& expr,
                                        GpuExpressionState* state);
  std::unique_ptr<cudf::column> Execute(const BoundReferenceExpression& expr,
                                        GpuExpressionState* state);

  //===----------Initialize State + Specializations----------===//
  static std::unique_ptr<GpuExpressionState> InitializeState(const Expression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundBetweenExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundCaseExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundCastExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundComparisonExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundConjunctionExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundConstantExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundFunctionExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundOperatorExpression& expr,
                                                             GpuExpressionExecutorState& state);
  static std::unique_ptr<GpuExpressionState> InitializeState(const BoundReferenceExpression& expr,
                                                             GpuExpressionExecutorState& state);
};

}  // namespace sirius
}  // namespace duckdb
