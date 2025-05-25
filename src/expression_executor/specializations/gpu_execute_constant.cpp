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
  Do(const BoundConstantExpression& expr, cudf::size_type count, rmm::device_async_resource_ref mr)
  {
    if constexpr (std::is_same<T, std::string>())
    {
      cudf::string_scalar scalar(expr.value.GetValue<std::string>(),
                                 true,
                                 cudf::get_default_stream(),
                                 mr);
      return cudf::make_column_from_scalar(scalar, count, cudf::get_default_stream(), mr);
    }
    else
    {
      auto scalar =
        cudf::numeric_scalar<T>(expr.value.GetValue<T>(), true, cudf::get_default_stream(), mr);
      return cudf::make_column_from_scalar(scalar, count, cudf::get_default_stream(), mr);
    }
  }
};

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundConstantExpression& expr,
                                                             GpuExpressionState* state)
{
  D_ASSERT(expr.value.type() == expr.return_type);

  // In many cases, this is pruned away
  switch (GpuExpressionState::GetCudfType(expr.return_type).id())
  {
    case cudf::type_id::INT32:
      return MakeColumnFromConstant<int32_t>::Do(expr, input_count, resource_ref);
    case cudf::type_id::UINT64:
      return MakeColumnFromConstant<uint64_t>::Do(expr, input_count, resource_ref);
    case cudf::type_id::FLOAT32:
      return MakeColumnFromConstant<float_t>::Do(expr, input_count, resource_ref);
    case cudf::type_id::FLOAT64:
      return MakeColumnFromConstant<double_t>::Do(expr, input_count, resource_ref);
    case cudf::type_id::BOOL8:
      return MakeColumnFromConstant<bool>::Do(expr, input_count, resource_ref);
    case cudf::type_id::STRING:
      return MakeColumnFromConstant<std::string>::Do(expr, input_count, resource_ref);
    default:
      throw InternalException("Execute[Constant]: Unknown type!");
  }
}

} // namespace sirius
} // namespace duckdb