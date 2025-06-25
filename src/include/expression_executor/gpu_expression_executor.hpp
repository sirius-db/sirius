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

#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"
#include <cudf/types.hpp>
#include <memory>
#include <vector>

namespace duckdb
{
namespace sirius
{

/// CONFIG ///
// Whether to use CuDF or Sirius for string functions
#define USE_CUDF_EXPR false
/// CONFIG ///

//----------GpuExpressionExecutor----------//
struct GpuExpressionExecutor
{

  //----------Constructor/Destructor(s)----------//
  GpuExpressionExecutor() = default;
  explicit GpuExpressionExecutor(const Expression& expr);
  explicit GpuExpressionExecutor(const vector<unique_ptr<Expression>>& expressions);

  //----------Public Fields----------//
  // The expressions of the executor
  std::vector<const Expression*> expressions;
  // The executor states for the expressions this executor is responsible for
  std::vector<std::unique_ptr<GpuExpressionExecutorState>> states;
  // The input (argument) columns for the current physical operator
  std::vector<shared_ptr<GPUColumn>> input_columns;
  // The memory resource
  rmm::device_async_resource_ref resource_ref = GPUBufferManager::GetInstance().mr;
  // The input count for the current relation (needed for materializing constants)
  cudf::size_type input_count;
  // Whether some input column is empty
  bool has_null_input_column;
  // The stream in which to execute the given set of expressions
  rmm::cuda_stream_view execution_stream;
  // Static flag indicating whether to use cudf or sirius for string functions
  static constexpr bool use_cudf = USE_CUDF_EXPR;

  //----------Methods----------//
  void AddExpression(const Expression& expr);
  void ClearExpressions();

  // Set the root state of the executor to the given expression
  void Initialize(const Expression& expr, GpuExpressionExecutorState& state);

  // Set the input count and columns for the expression executor
  void SetInputColumns(const GPUIntermediateRelation& input_relation);

  // Before evaluating an expression, check the leaves for nullptrs
  // (Assumes the input columns have already been set)
  bool HasNullLeaf(const Expression& expr) const;
  template <typename ExpressionT>
  bool HasNullLeafLoop(const ExpressionT& expr) const;

  // Execute the set of expressions with the given input relation and store the result in the output
  // relation (Provides the main interface with client code for Projections).
  void Execute(const GPUIntermediateRelation& input_relation,
               GPUIntermediateRelation& output_relation,
               rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  // Execute the set of expressions with the given input relation and compact into the output
  // relation based on the resulting selection vector (Provides the main interface with client code
  // for Filters).
  void Select(GPUIntermediateRelation& input_relation,
              GPUIntermediateRelation& output_relation,
              rmm::cuda_stream_view stream = rmm::cuda_stream_default);

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

  //----------Initialize State + Specializations----------//
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

} // namespace sirius
} // namespace duckdb