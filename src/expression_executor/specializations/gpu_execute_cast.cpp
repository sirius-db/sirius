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


// Note that constants are CASTed before they reach the execution engine
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundCastExpression& expr,
                                                             GpuExpressionState* state)
{
  auto return_type_id = GpuExpressionState::GetCudfType(expr.return_type).id();

  // Resolve the child
  auto* child_state = state->child_states[0].get();
  auto child = Execute(*expr.child, child_state);

  // Execute the cast
  return cudf::cast(child->view(),
                    cudf::data_type{return_type_id},
                    execution_stream,
                    resource_ref);
}

} // namespace sirius
} // namespace duckdb
