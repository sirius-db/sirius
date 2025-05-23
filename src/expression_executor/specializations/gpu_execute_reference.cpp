#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "gpu_materialize.hpp"

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundReferenceExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  return std::make_unique<GpuExpressionState>(expr, root);
}

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundReferenceExpression& expr,
                                                             GpuExpressionState* state)
{
  // Materialize first, if necessary
  // TODO: materialize directly to a std::unique_ptr<cudf::column> to avoid unecessary copy
  auto input_column = input_columns[expr.index];
  if (input_column->row_ids != nullptr)
  {
    std::cout << "MATERIALIZING...\n";
    auto dummy_reference_expression = BoundReferenceExpression(LogicalType(LogicalTypeId::ANY), 0);
    input_column                    = HandleMaterializeExpression(input_column,
                                               dummy_reference_expression /* Unused */,
                                               &GPUBufferManager::GetInstance());
    std::cout << "DONE MATERIALIZING...\n";
  }

  // Perform a deep copy (in some cases, this copy is pruned)
  std::cout << "MAKING COLUMN...\n";
  auto input_column_view = input_column->convertToCudfColumn();
  std::cout << "VIEW MADE...\n";
  return std::make_unique<cudf::column>(input_column_view,
                                        cudf::get_default_stream(),
                                        resource_ref);
}

} // namespace sirius
} // namespace duckdb
