#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "expression_executor/gpu_dispatcher.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "expression_executor/gpu_expression_executor_state.hpp"
#include "gpu_physical_strings_matching.hpp"
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>
#include <string>

namespace duckdb
{
namespace sirius
{

// There has to be a better way to extract the function semantics than string comparison!
//----------Function Strings----------//
#define ADD_FUNC_STR "+"
#define SUB_FUNC_STR "-"
#define MUL_FUNC_STR "*"
#define DIV_FUNC_STR "/"
#define INT_DIV_FUNC_STR "//"
#define MOD_FUNC_STR "%"
#define SUBSTRING_FUNC_STR_1 "substring"
#define SUBSTRING_FUNC_STR_2 "substr"
#define LIKE_FUNC_STR "~~"
#define NOT_LIKE_FUNC_STR "!~~"
#define CONTAINS_FUNC_STR "contains"
#define PREFIX_FUNC_STR "prefix"
#define ERROR_FUNC_STR "error"

#define SPLIT_DELIMITER "%"

//----------InitializeState----------//
std::unique_ptr<GpuExpressionState>
GpuExpressionExecutor::InitializeState(const BoundFunctionExpression& expr,
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
template <StringMatchingType MatchType, bool UseCudf>
struct StringMatchingDispatcher
{
  // The executor
  GpuExpressionExecutor* gpu_expression_executor;

  // Constructor
  explicit StringMatchingDispatcher(GpuExpressionExecutor* gpu_expression_executor)
      : gpu_expression_executor(gpu_expression_executor)
  {}

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 2);
    D_ASSERT(expr.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);

    auto input = gpu_expression_executor->Execute(*expr.children[0], state->child_states[0].get());
    const auto& match_str_expr = expr.children[1]->Cast<BoundConstantExpression>();
    const auto& match_str      = match_str_expr.value.GetValue<std::string>();

    if constexpr (UseCudf)
    {
      //----------Using CuDF----------//
      cudf::strings_column_view input_view(input->view());
      if constexpr (MatchType == StringMatchingType::LIKE ||
                    MatchType == StringMatchingType::NOT_LIKE)
      {
        std::vector<std::string> match_terms = string_split(match_str, SPLIT_DELIMITER);

        auto like = cudf::strings::like(cudf::strings_column_view(input_view),
                                        cudf::string_scalar(match_str),
                                        cudf::string_scalar(""),
                                        cudf::get_default_stream(),
                                        gpu_expression_executor->resource_ref);

        // LIKE or NOT LIKE?
        if constexpr (MatchType == StringMatchingType::LIKE)
        {
          return std::move(like);
        }
        else
        {
          // Negate the match result
          return cudf::unary_operation(like->view(),
                                       cudf::unary_operator::NOT,
                                       cudf::get_default_stream(),
                                       gpu_expression_executor->resource_ref);
        }
      }
      else if constexpr (MatchType == StringMatchingType::CONTAINS)
      {
        const auto match_str_scalar = cudf::string_scalar(match_str,
                                                          true,
                                                          cudf::get_default_stream(),
                                                          gpu_expression_executor->resource_ref);
        return cudf::strings::contains(input_view,
                                       match_str_scalar,
                                       cudf::get_default_stream(),
                                       gpu_expression_executor->resource_ref);
      }
      else if constexpr (MatchType == StringMatchingType::PREFIX)
      {
        const auto match_str_scalar = cudf::string_scalar(match_str,
                                                          true,
                                                          cudf::get_default_stream(),
                                                          gpu_expression_executor->resource_ref);
        return cudf::strings::starts_with(input_view,
                                          match_str_scalar,
                                          cudf::get_default_stream(),
                                          gpu_expression_executor->resource_ref);
      }
    }
    else
    {
      //----------Using Sirius----------//
      return GpuDispatcher::DispatchStringMatching<MatchType>(
        input->view(),
        match_str,
        gpu_expression_executor->resource_ref);
    }
  }
};

//----------Execute----------//
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundFunctionExpression& expr,
                                                             GpuExpressionState* state)
{
  const auto& function_expression_state = state->Cast<GpuExpressionState>();
  const auto& return_type               = GpuExpressionState::GetCudfType(expr.return_type);
  const auto& func_str                  = expr.function.name;

  //----------Numeric Binary Functions----------//
  // Lambda for numeric binary operators
  auto binary_function = [this, &expr, state, return_type](
                           cudf::binary_operator bin_op) -> std::unique_ptr<cudf::column> {
    // Resolve children
    auto left  = Execute(*expr.children[0], state->child_states[0].get());
    auto right = Execute(*expr.children[1], state->child_states[1].get());

    return cudf::binary_operation(left->view(),
                                  right->view(),
                                  bin_op,
                                  return_type,
                                  cudf::get_default_stream(),
                                  resource_ref);
  };

  // Execute this function
  if (func_str == ADD_FUNC_STR)
  {
    return binary_function(cudf::binary_operator::ADD);
  }
  else if (func_str == SUB_FUNC_STR)
  {
    return binary_function(cudf::binary_operator::SUB);
  }
  else if (func_str == MUL_FUNC_STR)
  {
    return binary_function(cudf::binary_operator::MUL);
  }
  else if (func_str == DIV_FUNC_STR || func_str == INT_DIV_FUNC_STR)
  {
    // For non-integer division on integer types, DuckDB inserts a CAST
    return binary_function(cudf::binary_operator::DIV);
  }
  else if (func_str == MOD_FUNC_STR)
  {
    return binary_function(cudf::binary_operator::MOD);
  }
  else if (func_str == ERROR_FUNC_STR)
  {
    throw InternalException("Execute[Function]: error() should be handled by Execute[Case]!");
  }

  //----------String Functions----------//
  if (func_str == SUBSTRING_FUNC_STR_1 || func_str == SUBSTRING_FUNC_STR_2)
  {
    // We assume the start and len arguments are constants (seems to be the case in DuckDB)
    D_ASSERT(expr.children.size() == 3);
    D_ASSERT(expr.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);
    D_ASSERT(expr.children[2]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);

    const auto& start_expr = expr.children[1]->Cast<BoundConstantExpression>();
    const auto& len_expr   = expr.children[2]->Cast<BoundConstantExpression>();

    auto input = Execute(*expr.children[0], state->child_states[0].get());

    if constexpr (use_cudf)
    {
      cudf::strings_column_view input_view(input->view());
      const auto cudf_start = start_expr.value.GetValue<cudf::size_type>() - 1;
      const auto cudf_end   = len_expr.value.GetValue<cudf::size_type>() + cudf_start;

      return cudf::strings::slice_strings(input_view, cudf_start, cudf_end);
    }
    else
    {
      const auto sirius_start = start_expr.value.GetValue<uint64_t>() - 1;
      const auto sirius_len   = len_expr.value.GetValue<uint64_t>();

      return GpuDispatcher::DispatchSubstring(input->view(),
                                              sirius_start,
                                              sirius_len,
                                              resource_ref);
    }
  }
  else if (func_str == LIKE_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::LIKE, use_cudf> dispatcher(this);
    return dispatcher(expr, state);
  }
  else if (func_str == NOT_LIKE_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::NOT_LIKE, use_cudf> dispatcher(this);
    return dispatcher(expr, state);
  }
  else if (func_str == CONTAINS_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::CONTAINS, use_cudf> dispatcher(this);
    return dispatcher(expr, state);
  }
  else if (func_str == PREFIX_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::PREFIX, use_cudf> dispatcher(this);
    return dispatcher(expr, state);
  }

  // If we've gotten this far, we've encountered a unimplemented function type
  std::cerr << "UNKNOWN FUNCTION STRING: " << func_str << "\n";
  throw InternalException("Execute[Function]: Unknown function type!");
}

} // namespace sirius
} // namespace duckdb
