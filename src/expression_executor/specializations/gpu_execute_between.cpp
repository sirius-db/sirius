#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include <cudf/binaryop.hpp>

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundBetweenExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = make_uniq<GpuExpressionState>(expr, root);
  result->AddChild(*expr.input);
  result->AddChild(*expr.lower);
  result->AddChild(*expr.upper);
  return std::move(result);
}

// KEVIN: potential optimization path: if lower and upper bounds are constants, skip Execute() for
// them
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundBetweenExpression& expr,
                                                             GpuExpressionState* state)
{
  // Resolve the children
  auto input = Execute(*expr.input, state->child_states[0].get());
  auto lower = Execute(*expr.lower, state->child_states[1].get());
  auto upper = Execute(*expr.upper, state->child_states[2].get());

  // CuDF does not provide native support for ternary BETWEEN.
  auto lower_cmp = cudf::binary_operation(input->view(),
                                          lower->view(),
                                          cudf::binary_operator::GREATER_EQUAL,
                                          cudf::data_type{cudf::type_id::BOOL8},
                                          cudf::get_default_stream(),
                                          resource_ref);
  auto upper_cmp = cudf::binary_operation(input->view(),
                                          upper->view(),
                                          cudf::binary_operator::LESS_EQUAL,
                                          cudf::data_type{cudf::type_id::BOOL8},
                                          cudf::get_default_stream(),
                                          resource_ref);
  return cudf::binary_operation(lower_cmp->view(),
                                upper_cmp->view(),
                                cudf::binary_operator::LOGICAL_AND,
                                cudf::data_type{cudf::type_id::BOOL8},
                                cudf::get_default_stream(),
                                resource_ref);
}

} // namespace sirius
} // namespace duckdb
