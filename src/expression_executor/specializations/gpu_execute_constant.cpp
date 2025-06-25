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

#include "cuda_stream_view.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include <type_traits>

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundConstantExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  return std::make_unique<GpuExpressionState>(expr, root);
}

// Helper template functor to reduce bloat
template <typename T>
struct MakeColumnFromConstant
{
  static std::unique_ptr<cudf::column>
  Do(const BoundConstantExpression& expr, cudf::size_type count, rmm::device_async_resource_ref mr, rmm::cuda_stream_view stream)
  {
    if constexpr (std::is_same<T, std::string>())
    {
      cudf::string_scalar scalar(expr.value.GetValue<std::string>(),
                                 true,
                                 stream,
                                 mr);
      return cudf::make_column_from_scalar(scalar, count, stream, mr);
    }
    else if constexpr (std::is_same_v<T, numeric::decimal32> || std::is_same_v<T, numeric::decimal64>)
    {
      auto type = expr.value.type();
      if (type.id() != LogicalTypeId::DECIMAL) {
        throw InternalException("Invalid duckdb type for decimal constant: %d", static_cast<int>(type.id()));
      }
      // cudf decimal type uses negative scale
      auto scalar = cudf::fixed_point_scalar<T>(expr.value.GetValueUnsafe<typename T::rep>(),
                                                numeric::scale_type{-DecimalType::GetScale(type)},
                                                true,
                                                stream,
                                                mr);
      return cudf::make_column_from_scalar(scalar, count, stream, mr);
    }
    else
    {
      auto scalar =
        cudf::numeric_scalar<T>(expr.value.GetValue<T>(), true, stream, mr);
      return cudf::make_column_from_scalar(scalar, count, stream, mr);
    }
  }
};

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundConstantExpression& expr,
                                                             GpuExpressionState* state)
{
  D_ASSERT(expr.value.type() == expr.return_type);

  // In many cases, this column materialization is pruned away
  auto cudf_type = GpuExpressionState::GetCudfType(expr.return_type);
  switch (cudf_type.id())
  {
    case cudf::type_id::INT32:
      return MakeColumnFromConstant<int32_t>::Do(expr, input_count, resource_ref, execution_stream);
    case cudf::type_id::UINT64:
      return MakeColumnFromConstant<uint64_t>::Do(expr, input_count, resource_ref, execution_stream);
    case cudf::type_id::FLOAT32:
      return MakeColumnFromConstant<float_t>::Do(expr, input_count, resource_ref, execution_stream);
    case cudf::type_id::FLOAT64:
      return MakeColumnFromConstant<double_t>::Do(expr, input_count, resource_ref, execution_stream);
    case cudf::type_id::BOOL8:
      return MakeColumnFromConstant<bool>::Do(expr, input_count, resource_ref, execution_stream);
    case cudf::type_id::STRING:
      return MakeColumnFromConstant<std::string>::Do(expr, input_count, resource_ref, execution_stream);
    case cudf::type_id::DECIMAL32:
      return MakeColumnFromConstant<numeric::decimal32>::Do(expr, input_count, resource_ref, execution_stream);
    case cudf::type_id::DECIMAL64:
      return MakeColumnFromConstant<numeric::decimal64>::Do(expr, input_count, resource_ref, execution_stream);
    default:
      throw InternalException("Execute[Constant]: Unknown cudf type: %d", static_cast<int>(cudf_type.id()));
  }
}

} // namespace sirius
} // namespace duckdb