#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_dispatcher.hpp"
#include "expression_executor/gpu_expression_executor.hpp"

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
  auto input_column = input_columns[expr.index];
  if (input_column->row_ids != nullptr)
  {
    if (input_column->row_id_count > INT32_MAX)
    {
      throw NotImplementedException(
        "Execute[Reference]: row_id_counts larger than int32_t are not supported in libcudf");
    }
    // DispatchMaterialize will handle byte overflow from the materialized offsets
    return GpuDispatcher::DispatchMaterialize(input_column.get(), resource_ref);
  }
  if (input_column->data_wrapper.type == ColumnType::VARCHAR &&
      input_column->data_wrapper.num_bytes > INT32_MAX)
  {
    throw NotImplementedException(
      "Execute[Reference]: string offsets larger than int32_t are not supported in libcudf");
  }
  if (input_column->column_length > INT32_MAX)
  {
    throw NotImplementedException(
      "Execute[Reference]: input column length larger than int32_t are not supported in libcudf");
  }

  // Perform a deep copy (necessary, since memory ownership cannot be transferred away from the gpu
  // buffer manager)
  auto input_column_view = input_column->convertToCudfColumn();
  return std::make_unique<cudf::column>(input_column_view,
                                        cudf::get_default_stream(),
                                        resource_ref);
}

} // namespace sirius
} // namespace duckdb
