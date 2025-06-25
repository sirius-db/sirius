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

#include "expression_executor/gpu_expression_executor_state.hpp"
#include "expression_executor/gpu_expression_executor.hpp"

namespace duckdb
{
namespace sirius
{

void GpuExpressionState::AddChild(const Expression& child_expr)
{
  // Types
  types.push_back(GetCudfType(child_expr.return_type));

  // Children states
  auto child_state = GpuExpressionExecutor::InitializeState(child_expr, root);
  child_states.push_back(std::move(child_state));
}

GpuExpressionState::GpuExpressionState(const Expression& expr, GpuExpressionExecutorState& root)
    : expr(expr)
    , root(root)
{}

GpuExpressionExecutorState::GpuExpressionExecutorState()
{}

} // namespace sirius
} // namespace duckdb