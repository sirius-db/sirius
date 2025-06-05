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