#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include <cudf/unary.hpp>

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundCastExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = make_uniq<GpuExpressionState>(expr, root);
  result->AddChild(*expr.child);
  return std::move(result);
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundCastExpression& expr,
                                                             GpuExpressionState* state)
{
  // Resolve the child
  auto* child_state = state->child_states[0].get();
  auto child        = Execute(*expr.child, child_state);

  auto type_id = GpuExpressionState::GetCudfType(expr.return_type).id();
  return cudf::cast(child->view(),
                    cudf::data_type{type_id},
                    cudf::get_default_stream(),
                    resource_ref);
}

} // namespace sirius
} // namespace duckdb
