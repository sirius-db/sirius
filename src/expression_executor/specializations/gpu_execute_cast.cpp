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

#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"

#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_checks.hpp>

namespace duckdb {
namespace sirius {

std::unique_ptr<GpuExpressionState> GpuExpressionExecutor::InitializeState(
  const BoundCastExpression& expr, GpuExpressionExecutorState& root)
{
  auto result = make_uniq<GpuExpressionState>(expr, root);
  result->AddChild(*expr.child);
  return std::move(result);
}

// Note that constants are CASTed before they reach the execution engine
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundCastExpression& expr,
                                                             GpuExpressionState* state)
{
  auto return_type = GetCudfType(expr.return_type);

  // Resolve the child
  auto* child_state = state->child_states[0].get();
  auto child        = Execute(*expr.child, child_state);

  // cuDF's parquet reader may read DECIMAL columns as STRING when the parquet
  // encoding isn't natively supported. Handle STRING→DECIMAL conversion using
  // cudf::strings::to_fixed_point instead of cudf::cast (which doesn't support it).
  if (child->view().type().id() == cudf::type_id::STRING && cudf::is_fixed_point(return_type)) {
    return cudf::strings::to_fixed_point(
      cudf::strings_column_view(child->view()), return_type, execution_stream, resource_ref);
  }

  return cudf::cast(child->view(), return_type, execution_stream, resource_ref);
}

}  // namespace sirius
}  // namespace duckdb
