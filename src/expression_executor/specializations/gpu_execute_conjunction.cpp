#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include <cudf/binaryop.hpp>

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundConjunctionExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = make_uniq<GpuExpressionState>(expr, root);
  for (auto& child : expr.children)
  {
    result->AddChild(*child);
  }
  return std::move(result);
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundConjunctionExpression& expr,
                                                             GpuExpressionState* state)
{
  D_ASSERT(output_column.data_wrapper.type == ColumnType::BOOLEAN);
  auto return_type = GpuExpressionState::GetCudfType(expr.return_type);

  std::unique_ptr<cudf::column> output_column;

  // Resolve the children
  for (idx_t i = 0; i < expr.children.size(); i++)
  {
    D_ASSERT(state->intermediate_columns[i].data_wrapper.type == ColumnType::BOOLEAN);

    auto current_result = Execute(*expr.children[i], state->child_states[i].get());

    if (i == 0)
    {
      // Nothing to compare
      output_column = std::move(current_result);
    }
    else
    {
      // AND/OR current result with output collecte so far
      switch (expr.GetExpressionType())
      {
        case ExpressionType::CONJUNCTION_AND:
          output_column = cudf::binary_operation(current_result->view(),
                                                 output_column->view(),
                                                 cudf::binary_operator::LOGICAL_AND,
                                                 return_type,
                                                 cudf::get_default_stream(),
                                                 resource_ref);
          break;
        case ExpressionType::CONJUNCTION_OR:
          output_column = cudf::binary_operation(current_result->view(),
                                                 output_column->view(),
                                                 cudf::binary_operator::LOGICAL_OR,
                                                 return_type,
                                                 cudf::get_default_stream(),
                                                 resource_ref);
          break;
        default:
          std::cout << "EXPRESSION TYPE: " << static_cast<int64_t>(expr.GetExpressionType())
                    << "\n";
          throw InternalException("Execute[Conjunction]: Unknown conjunction type!");
      }
    }
  }

  return std::move(output_column);
}

} // namespace sirius
} // namespace duckdb
