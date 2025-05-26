#include "duckdb/common/assert.hpp"
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

//----------StringMatchingDispatcher----------//
// Helper template functor for string matching operations to reduce bloat in Execute()
template <StringMatchingType MatchType, bool UseCudf>
struct StringMatchingDispatcher
{
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit StringMatchingDispatcher(GpuExpressionExecutor& exec)
      : executor(exec)
  {}

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 2);
    D_ASSERT(expr.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT);

    auto input                 = executor.Execute(*expr.children[0], state->child_states[0].get());
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
                                        executor.resource_ref);

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
                                       executor.resource_ref);
        }
      }
      else if constexpr (MatchType == StringMatchingType::CONTAINS)
      {
        const auto match_str_scalar =
          cudf::string_scalar(match_str, true, cudf::get_default_stream(), executor.resource_ref);
        return cudf::strings::contains(input_view,
                                       match_str_scalar,
                                       cudf::get_default_stream(),
                                       executor.resource_ref);
      }
      else if constexpr (MatchType == StringMatchingType::PREFIX)
      {
        const auto match_str_scalar =
          cudf::string_scalar(match_str, true, cudf::get_default_stream(), executor.resource_ref);
        return cudf::strings::starts_with(input_view,
                                          match_str_scalar,
                                          cudf::get_default_stream(),
                                          executor.resource_ref);
      }
    }
    else
    {
      //----------Using Sirius----------//
      return GpuDispatcher::DispatchStringMatching<MatchType>(input->view(),
                                                              match_str,
                                                              executor.resource_ref);
    }
  }
};

//----------NumericBinaryFunctionDispatcher----------//
template <cudf::binary_operator BinOp>
struct NumericBinaryFunctionDispatcher
{
  // The executor
  GpuExpressionExecutor& executor;

  // Constructor
  explicit NumericBinaryFunctionDispatcher(GpuExpressionExecutor& exec)
      : executor(exec)
  {}

  // Left scalar binary operator
  template <typename T>
  std::unique_ptr<cudf::column> DoLeftScalarBinaryOp(const T& left_value,
                                                     const cudf::column_view& right,
                                                     const cudf::data_type& return_type)
  {
    auto left_numeric_scalar =
      cudf::numeric_scalar(left_value, true, cudf::get_default_stream(), executor.resource_ref);
    return cudf::binary_operation(left_numeric_scalar,
                                  right,
                                  BinOp,
                                  return_type,
                                  cudf::get_default_stream(),
                                  executor.resource_ref);
  }

  // Right scalar binary operator
  template <typename T>
  std::unique_ptr<cudf::column> DoRightScalarBinaryOp(const cudf::column_view& left,
                                                      const T& right_value,
                                                      const cudf::data_type& return_type)
  {
    auto right_numeric_scalar =
      cudf::numeric_scalar(right_value, true, cudf::get_default_stream(), executor.resource_ref);
    return cudf::binary_operation(left,
                                  right_numeric_scalar,
                                  BinOp,
                                  return_type,
                                  cudf::get_default_stream(),
                                  executor.resource_ref);
  }

  // Dispatch operator
  std::unique_ptr<cudf::column> operator()(const BoundFunctionExpression& expr,
                                           GpuExpressionState* state)
  {
    D_ASSERT(expr.children.size() == 2);
    const auto& return_type = GpuExpressionState::GetCudfType(expr.return_type);

    // Resolve children
    if (expr.children[0]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT)
    {
      // LHS is a constant, so skip its column materialization
      const auto& left_value = expr.children[0]->Cast<BoundConstantExpression>().value;
      const auto& right      = executor.Execute(*expr.children[1], state->child_states[1].get());

      switch (GpuExpressionState::GetCudfType(expr.children[0]->return_type).id())
      {
        case cudf::type_id::INT32:
          return DoLeftScalarBinaryOp(left_value.GetValue<int32_t>(), right->view(), return_type);
        case cudf::type_id::UINT64:
          return DoLeftScalarBinaryOp(left_value.GetValue<uint64_t>(), right->view(), return_type);
        case cudf::type_id::FLOAT32:
          return DoLeftScalarBinaryOp(left_value.GetValue<float_t>(), right->view(), return_type);
        case cudf::type_id::FLOAT64:
          return DoLeftScalarBinaryOp(left_value.GetValue<double_t>(), right->view(), return_type);
        case cudf::type_id::BOOL8:
          throw NotImplementedException("Execute[Function]: Boolean types not supported for "
                                        "numeric binary operations!");
        default:
          throw InternalException("Execute[Function]: Unknown constant type for binary "
                                  "operation!");
      }
    }
    else if (expr.children[1]->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT)
    {
      // RHS is a constant, so skip its column materialization
      const auto& right_value = expr.children[1]->Cast<BoundConstantExpression>().value;
      const auto& left        = executor.Execute(*expr.children[0], state->child_states[0].get());

      switch (GpuExpressionState::GetCudfType(expr.children[1]->return_type).id())
      {
        case cudf::type_id::INT32:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<int32_t>(), return_type);
        case cudf::type_id::UINT64:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<uint64_t>(), return_type);
        case cudf::type_id::FLOAT32:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<float_t>(), return_type);
        case cudf::type_id::FLOAT64:
          return DoRightScalarBinaryOp(left->view(), right_value.GetValue<double_t>(), return_type);
        case cudf::type_id::BOOL8:
          throw NotImplementedException("Execute[Function]: Boolean types not supported for "
                                        "numeric binary operations!");
        default:
          throw InternalException("Execute[Function]: Unknown constant type for binary "
                                  "operation!");
      }
    }

    // NEITHER side is a constant, so we need to execute both children
    auto left  = executor.Execute(*expr.children[0], state->child_states[0].get());
    auto right = executor.Execute(*expr.children[1], state->child_states[1].get());

    // Execute the binary operation
    return cudf::binary_operation(left->view(),
                                  right->view(),
                                  BinOp,
                                  GpuExpressionState::GetCudfType(expr.return_type),
                                  cudf::get_default_stream(),
                                  executor.resource_ref);
  }
};

//----------Execute----------//
std::unique_ptr<cudf::column> GpuExpressionExecutor::Execute(const BoundFunctionExpression& expr,
                                                             GpuExpressionState* state)
{
  const auto& function_expression_state = state->Cast<GpuExpressionState>();
  const auto& func_str                  = expr.function.name;

  //----------Numeric Binary Functions----------//
  if (func_str == ADD_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::ADD> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == SUB_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::SUB> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == MUL_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::MUL> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == DIV_FUNC_STR || func_str == INT_DIV_FUNC_STR)
  {
    // For non-integer division on integer types, DuckDB inserts a CAST
    NumericBinaryFunctionDispatcher<cudf::binary_operator::DIV> binary_function(*this);
    return binary_function(expr, state);
  }
  else if (func_str == MOD_FUNC_STR)
  {
    NumericBinaryFunctionDispatcher<cudf::binary_operator::MOD> binary_function(*this);
    return binary_function(expr, state);
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
    StringMatchingDispatcher<StringMatchingType::LIKE, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == NOT_LIKE_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::NOT_LIKE, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == CONTAINS_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::CONTAINS, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }
  else if (func_str == PREFIX_FUNC_STR)
  {
    StringMatchingDispatcher<StringMatchingType::PREFIX, use_cudf> dispatcher(*this);
    return dispatcher(expr, state);
  }

  // If we've gotten this far, we've encountered a unimplemented function type
  std::cerr << "UNKNOWN FUNCTION STRING: " << func_str << "\n";
  throw InternalException("Execute[Function]: Unknown function type!");
}

} // namespace sirius
} // namespace duckdb
