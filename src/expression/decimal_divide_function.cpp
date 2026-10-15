/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "expression/decimal_divide_function.hpp"

#include "helper/decimal_divide.hpp"

// duckdb
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/optional.hpp>
#include <duckdb/common/types/decimal.hpp>
#include <duckdb/common/vector_operations/binary_executor.hpp>
#include <duckdb/execution/expression_executor.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/transaction/meta_transaction.hpp>

// standard library
#include <cstdint>
#include <string>

namespace sirius {

namespace {

using duckdb::LogicalType;
using duckdb::LogicalTypeId;

/// The type the bind casts an operand to. The GPU kernel reads 32-, 64- and 128-bit operands, so
/// narrow integers widen to INTEGER, integers beyond BIGINT become DECIMAL(p, 0), and a DECIMAL
/// of precision 4 or less (DuckDB stores it as INT16, which the GPU does not support) widens to
/// precision 9.
LogicalType operand_type(LogicalType const& type, char const* operand)
{
  switch (type.id()) {
    case LogicalTypeId::INTEGER_LITERAL:
      return operand_type(duckdb::IntegerLiteral::GetType(type), operand);
    case LogicalTypeId::SQLNULL:
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::UTINYINT:
    case LogicalTypeId::USMALLINT:
    case LogicalTypeId::INTEGER: return LogicalType::INTEGER;
    case LogicalTypeId::UINTEGER:
    case LogicalTypeId::BIGINT: return LogicalType::BIGINT;
    case LogicalTypeId::UBIGINT: return LogicalType::DECIMAL(20, 0);
    case LogicalTypeId::HUGEINT:
    case LogicalTypeId::UHUGEINT: return LogicalType::DECIMAL(38, 0);
    case LogicalTypeId::DECIMAL: {
      auto const width = duckdb::DecimalType::GetWidth(type);
      auto const scale = duckdb::DecimalType::GetScale(type);
      return width <= 4 ? LogicalType::DECIMAL(9, scale) : type;
    }
    default:
      throw duckdb::BinderException("%s: the %s must be a DECIMAL or an integer, not %s",
                                    decimal_divide_function_name,
                                    operand,
                                    type.ToString());
  }
}

int fractional_digits(LogicalType const& type)
{
  return type.id() == LogicalTypeId::DECIMAL ? duckdb::DecimalType::GetScale(type) : 0;
}

// DuckDB v2 settles argument and return types in the resolve_types callback; a bind callback may
// not change them.
void resolve_decimal_divide_types(duckdb::ResolveScalarFunctionTypesInput& input)
{
  auto& context              = input.GetClientContext();
  auto const& scale_argument = input.GetArgument(2);
  if (scale_argument.HasParameter()) { throw duckdb::ParameterNotResolvedException(); }
  auto const& scale_type = scale_argument.GetReturnType();
  bool const integral =
    scale_type.IsIntegral() || scale_type.id() == LogicalTypeId::INTEGER_LITERAL;
  if (!integral || !scale_argument.IsFoldable()) {
    throw duckdb::BinderException("%s: the result scale must be an integer constant",
                                  decimal_divide_function_name);
  }
  auto const scale_value = duckdb::ExpressionExecutor::EvaluateScalar(context, scale_argument)
                             .DefaultCastAs(LogicalType::INTEGER);
  if (scale_value.IsNull()) {
    throw duckdb::BinderException("%s: the result scale must not be NULL",
                                  decimal_divide_function_name);
  }
  auto const result_scale = duckdb::IntegerValue::Get(scale_value);
  if (result_scale < 0 || result_scale > decimal_divide_max_digits) {
    throw duckdb::BinderException("%s: the result scale must be between 0 and %d, not %d",
                                  decimal_divide_function_name,
                                  decimal_divide_max_digits,
                                  result_scale);
  }

  auto& bound_function             = input.GetBoundFunction();
  bound_function.GetArguments()[0] = operand_type(input.GetArgumentType(0), "dividend");
  bound_function.GetArguments()[1] = operand_type(input.GetArgumentType(1), "divisor");
  // The scale stays a constant argument, so a plan copy (serialize, deserialize) binds the call
  // again the same way. Execution reads the scale from the return type.
  bound_function.SetReturnType(
    LogicalType::DECIMAL(decimal_divide_max_digits, static_cast<uint8_t>(result_scale)));
}

__int128_t to_int128(int32_t value) { return value; }
__int128_t to_int128(int64_t value) { return value; }
__int128_t to_int128(duckdb::hugeint_t value)
{
  return static_cast<__int128_t>((static_cast<unsigned __int128>(value.upper) << 64) | value.lower);
}

duckdb::hugeint_t to_hugeint(__int128_t value)
{
  duckdb::hugeint_t result;
  result.lower = static_cast<uint64_t>(value);
  result.upper = static_cast<int64_t>(value >> 64);
  return result;
}

std::string operand_string(int32_t value, LogicalType const& type)
{
  return type.id() == LogicalTypeId::DECIMAL
           ? duckdb::Decimal::ToString(
               value, duckdb::DecimalType::GetWidth(type), duckdb::DecimalType::GetScale(type))
           : std::to_string(value);
}
std::string operand_string(int64_t value, LogicalType const& type)
{
  return type.id() == LogicalTypeId::DECIMAL
           ? duckdb::Decimal::ToString(
               value, duckdb::DecimalType::GetWidth(type), duckdb::DecimalType::GetScale(type))
           : std::to_string(value);
}
std::string operand_string(duckdb::hugeint_t value, LogicalType const& type)
{
  return duckdb::Decimal::ToString(
    value, duckdb::DecimalType::GetWidth(type), duckdb::DecimalType::GetScale(type));
}

template <typename Dividend, typename Divisor>
void execute_typed(duckdb::DataChunk& args, int result_scale, duckdb::Vector& result)
{
  auto const& dividend_type = args.data[0].GetType();
  auto const& divisor_type  = args.data[1].GetType();
  auto const exponent =
    result_scale + fractional_digits(divisor_type) - fractional_digits(dividend_type);
  // A lambda that returns an empty optional makes that row NULL.
  duckdb::BinaryExecutor::Execute<Dividend, Divisor, duckdb::hugeint_t>(
    args.data[0],
    args.data[1],
    result,
    args.size(),
    [&](Dividend dividend, Divisor divisor) -> duckdb::optional<duckdb::hugeint_t> {
      __int128_t quotient = 0;
      switch (decimal_divide_round(to_int128(dividend), to_int128(divisor), exponent, &quotient)) {
        case decimal_divide_status::ok: break;
        case decimal_divide_status::divide_by_zero: return duckdb::nullopt;
        case decimal_divide_status::overflow:
          throw duckdb::OutOfRangeException(
            "%s: the quotient of %s and %s needs more than %d digits at scale %d",
            decimal_divide_function_name,
            operand_string(dividend, dividend_type),
            operand_string(divisor, divisor_type),
            decimal_divide_max_digits,
            result_scale);
      }
      return to_hugeint(quotient);
    });
}

template <typename Dividend>
void execute_with_dividend(duckdb::DataChunk& args, int result_scale, duckdb::Vector& result)
{
  switch (args.data[1].GetType().InternalType()) {
    case duckdb::PhysicalType::INT32:
      return execute_typed<Dividend, int32_t>(args, result_scale, result);
    case duckdb::PhysicalType::INT64:
      return execute_typed<Dividend, int64_t>(args, result_scale, result);
    case duckdb::PhysicalType::INT128:
      return execute_typed<Dividend, duckdb::hugeint_t>(args, result_scale, result);
    default:
      throw duckdb::InternalException("%s: unexpected divisor type %s",
                                      decimal_divide_function_name,
                                      args.data[1].GetType().ToString());
  }
}

void execute_decimal_divide(duckdb::DataChunk& args,
                            duckdb::ExpressionState& state,
                            duckdb::Vector& result)
{
  auto const& return_type = state.expr.Cast<duckdb::BoundFunctionExpression>().GetReturnType();
  auto const result_scale = static_cast<int>(duckdb::DecimalType::GetScale(return_type));
  switch (args.data[0].GetType().InternalType()) {
    case duckdb::PhysicalType::INT32:
      return execute_with_dividend<int32_t>(args, result_scale, result);
    case duckdb::PhysicalType::INT64:
      return execute_with_dividend<int64_t>(args, result_scale, result);
    case duckdb::PhysicalType::INT128:
      return execute_with_dividend<duckdb::hugeint_t>(args, result_scale, result);
    default:
      throw duckdb::InternalException("%s: unexpected dividend type %s",
                                      decimal_divide_function_name,
                                      args.data[0].GetType().ToString());
  }
}

}  // namespace

duckdb::ScalarFunction get_decimal_divide_function()
{
  // Every argument is ANY: the bind accepts every DECIMAL width and integer type for the operands
  // and chooses their casts, and takes the scale from an integer constant of any width (a
  // Substrait plan may type it i32 or i64). The return type is set by the bind.
  duckdb::ScalarFunction function(decimal_divide_function_name,
                                  {LogicalType::ANY, LogicalType::ANY, LogicalType::ANY},
                                  LogicalType::DECIMAL(decimal_divide_max_digits, 0),
                                  execute_decimal_divide);
  function.SetResolveTypesCallback(resolve_decimal_divide_types);
  function.SetFallible();
  // DuckDB folds a call with a NULL constant operand to NULL before the bind runs, typed with the
  // unbound return type. Bind every call, so a NULL result is DECIMAL(38, result_scale) too; the
  // executor still makes a NULL operand a NULL result.
  function.SetNullHandling(duckdb::FunctionNullHandling::SPECIAL_HANDLING);
  return function;
}

void register_decimal_divide_function(duckdb::DatabaseInstance& instance)
{
  auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = duckdb::Catalog::GetSystemCatalog(instance);
  duckdb::CreateScalarFunctionInfo info(get_decimal_divide_function());
  // Idempotent: the extension load and the FFI's embedded database may both register.
  info.on_conflict = duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
  catalog.CreateFunction(transaction, info);
}

}  // namespace sirius
