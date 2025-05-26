#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include <algorithm>
#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/search.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/unary.hpp>
#include <rmm/device_uvector.hpp>

namespace duckdb
{
namespace sirius
{

std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundOperatorExpression& expr,
                                       GpuExpressionExecutorState& root)
{
  auto result = std::make_unique<GpuExpressionState>(expr, root);
  for (auto& child : expr.children)
  {
    result->AddChild(*child);
  }
  return std::move(result);
}

// Helper template functor to reduce bloat
template <typename T>
struct ExecuteNumericIn
{
  static std::unique_ptr<cudf::column> Do(const BoundOperatorExpression& expr,
                                          const cudf::column_view& input_view,
                                          rmm::device_async_resource_ref mr)
  {
    std::vector<T> children_vals;
    for (idx_t child = 1; child < expr.children.size(); ++child)
    {
      const auto& child_expression = expr.children[child]->Cast<BoundConstantExpression>();
      children_vals.push_back(child_expression.value.GetValue<T>());
    }
    rmm::device_uvector<T> children_vals_d(children_vals.size(), rmm::cuda_stream_default, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(children_vals_d.data(),
                                  children_vals.data(),
                                  children_vals.size() * sizeof(T),
                                  cudaMemcpyHostToDevice,
                                  cudf::get_default_stream()));
    cudf::column_view children_view(input_view.type(),
                                    static_cast<cudf::size_type>(children_vals.size()),
                                    children_vals_d.data(),
                                    nullptr,
                                    0,
                                    0);
    return cudf::contains(children_view, input_view, cudf::get_default_stream(), mr);
  }
};
// For strings
struct ExecuteStringIn
{
  static std::unique_ptr<cudf::column> Do(const BoundOperatorExpression& expr,
                                          const cudf::column_view& input_view,
                                          rmm::device_async_resource_ref mr)
  {
    auto num_strings = static_cast<cudf::size_type>(expr.children.size() - 1);
    auto num_offsets = num_strings + 1;

    // We need to convert to cudf/arrow format...
    std::vector<char> chars;
    std::vector<cudf::size_type> offsets;
    cudf::size_type offset = 0;
    for (idx_t child = 1; child < expr.children.size(); ++child)
    {
      const auto& child_expression = expr.children[child]->Cast<BoundConstantExpression>();
      const auto& child_string     = child_expression.value.GetValue<std::string>();
      chars.insert(chars.end(), child_string.begin(), child_string.end());
      offsets.push_back(offset);
      offset += static_cast<cudf::size_type>(child_string.size());
    }
    offsets.push_back(offset);

    // Allocate buffers and copy to device
    rmm::device_uvector<char> chars_buffer(offset, cudf::get_default_stream(), mr);
    rmm::device_uvector<cudf::size_type> offsets_buffer(num_offsets,
                                                        cudf::get_default_stream(),
                                                        mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(chars_buffer.data(),
                                  chars.data(),
                                  chars.size() * sizeof(char),
                                  cudaMemcpyHostToDevice,
                                  cudf::get_default_stream()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_buffer.data(),
                                  offsets.data(),
                                  offsets.size() * sizeof(cudf::size_type),
                                  cudaMemcpyHostToDevice,
                                  cudf::get_default_stream()));

    // Make CuDF things
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                      num_offsets,
                                                      std::move(offsets_buffer).release(),
                                                      rmm::device_buffer{},
                                                      0);
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(std::move(offsets_col));
    auto in_strings_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                         num_strings,
                                                         std::move(chars_buffer).release(),
                                                         rmm::device_buffer{},
                                                         0,
                                                         std::move(children));

    // Execute the search
    return cudf::contains(in_strings_col->view(), input_view, cudf::get_default_stream(), mr);
  }
};

std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundOperatorExpression& expr,
                                                             GpuExpressionState* state)
{
  auto expression_type = expr.GetExpressionType();
  auto return_type     = GpuExpressionState::GetCudfType(expr.return_type);

  if (expression_type == ExpressionType::COMPARE_IN ||
      expression_type == ExpressionType::COMPARE_NOT_IN)
  {
    if (expr.children.size() < 2)
    {
      throw InvalidInputException("Execute[BOUND_OPERATOR]: IN needs at least two children!");
    }

    // Evaluate the left side
    auto left      = Execute(*expr.children[0], state->child_states[0].get());
    auto left_type = left->type();

    // Optimization: special handling for case where RHS are all constants
    if (std::all_of(expr.children.begin() + 1, expr.children.end(), [](const auto& child) {
          return child->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT;
        }))
    {
      // All types should be the same
      switch (left_type.id())
      {
        case cudf::type_id::INT32:
          return ExecuteNumericIn<int32_t>::Do(expr, left->view(), resource_ref);
        case cudf::type_id::UINT64:
          return ExecuteNumericIn<uint64_t>::Do(expr, left->view(), resource_ref);
        case cudf::type_id::FLOAT32:
          return ExecuteNumericIn<float_t>::Do(expr, left->view(), resource_ref);
        case cudf::type_id::FLOAT64:
          return ExecuteNumericIn<double_t>::Do(expr, left->view(), resource_ref);
        case cudf::type_id::BOOL8:
          return ExecuteNumericIn<uint8_t>::Do(expr, left->view(), resource_ref);
        case cudf::type_id::STRING:
          return ExecuteStringIn::Do(expr, left->view(), resource_ref);
        default:
          std::cout << "UNKNOWN TYPE: " << static_cast<int32_t>(left->type().id()) << "\n";
          throw NotImplementedException("Execute[IN_CONSTANTS]: Unimplemented type!");
      }
    }

    // For every child, OR the result of the comparison with the left to get the overall result.
    std::unique_ptr<cudf::column> intermediate_result = nullptr;
    for (idx_t child = 1; child < expr.children.size(); ++child)
    {
      // Resolve the child
      auto comparator = Execute(*expr.children[child], state->child_states[child].get());
      auto comparison_result = cudf::binary_operation(left->view(),
                                                      comparator->view(),
                                                      cudf::binary_operator::EQUAL,
                                                      return_type,
                                                      cudf::get_default_stream(),
                                                      resource_ref);

      if (child == 1)
      {
        // First child: Move to result
        intermediate_result = std::move(comparison_result);
      }
      else
      {
        // Otherwise OR together
        intermediate_result = cudf::binary_operation(intermediate_result->view(),
                                                     comparison_result->view(),
                                                     cudf::binary_operator::LOGICAL_OR,
                                                     return_type,
                                                     cudf::get_default_stream(),
                                                     resource_ref);
      }
    }

    // NOT IN?
    if (expression_type == ExpressionType::COMPARE_NOT_IN)
    {
      // Negate the result and return
      return cudf::unary_operation(intermediate_result->view(),
                                   cudf::unary_operator::NOT,
                                   cudf::get_default_stream(),
                                   resource_ref);
    }
    else
    {
      // Return the result
      return std::move(intermediate_result);
    }
  }
  else if (expression_type == ExpressionType::OPERATOR_COALESCE)
  {
    throw NotImplementedException("Execute[OPERATOR_COALESCE]: Not yet implemented!");
  }
  else if (expr.children.size() == 1)
  {
    // Resolve child
    auto child = Execute(*expr.children[0], state->child_states[0].get());

    switch (expr.GetExpressionType())
    {
      case ExpressionType::OPERATOR_NOT:
        return cudf::unary_operation(child->view(),
                                     cudf::unary_operator::NOT,
                                     cudf::get_default_stream(),
                                     resource_ref);
      case ExpressionType::OPERATOR_IS_NULL:
        throw NotImplementedException("Execute[OPERATOR_IS_NULL]: Not yet implemented!");
      case ExpressionType::OPERATOR_IS_NOT_NULL:
        throw NotImplementedException("Execute[OPERATOR_IS_NOT_NULL]: Not yet implemented!");
      default:
        throw NotImplementedException("Execute[OPERATOR]: Unimplemented operator type with 1 "
                                      "child!");
    }
  }

  // If we've gotten this far, something ain't right
  throw NotImplementedException("Execute[OPERATOR]: Unimplemented operator type!");
}

} // namespace sirius
} // namespace duckdb
