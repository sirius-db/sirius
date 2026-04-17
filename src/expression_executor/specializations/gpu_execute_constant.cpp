/*
 * Copyright 2025, Sirius Contributors.
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

// sirius

#include <expression_executor/gpu_expression_executor.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <type_traits>

namespace {
using execute_result = ::sirius::gpu_expression_executor::execute_result;
using ast_node       = ::sirius::gpu_expression_executor::ast_result;
using execution_mode = ::sirius::gpu_expression_executor::execution_mode;

template <typename T>
execute_result make_execute_result_from_scalar(
  cudf::ast::tree& ast_tree,
  std::vector<std::unique_ptr<cudf::scalar>>& temp_scalars,
  T device_scalar,
  execution_mode mode)
{
  if (mode == execution_mode::AST) {
    auto const& expr_ref       = ast_tree.emplace<cudf::ast::literal>(*device_scalar);
    auto const temp_scalar_idx = temp_scalars.size();
    temp_scalars.push_back(std::move(device_scalar));
    return execute_result(ast_node(expr_ref, {temp_scalar_idx}, std::vector<std::size_t>{}));
  }
  return execute_result(std::move(device_scalar));
}

}  // namespace


namespace sirius {
using execute_result = gpu_expression_executor::execute_result;

execute_result gpu_expression_executor::execute(duckdb::BoundConstantExpression const& expr,
                                                execution_mode mode)
{
  assert(expr.value.type() == expr.return_type);

  auto const cudf_type = duckdb::GetCudfType(expr.value.type());
  switch (cudf_type.id()) {
    case cudf::type_id::INT8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int8_t>>(
        expr.value.GetValue<int8_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::INT16: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int16_t>>(
        expr.value.GetValue<int16_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::INT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(
        expr.value.GetValue<int32_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::INT64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int64_t>>(
        expr.value.GetValue<int64_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::UINT8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint8_t>>(
        expr.value.GetValue<uint8_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::UINT16: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint16_t>>(
        expr.value.GetValue<uint16_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::UINT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint32_t>>(
        expr.value.GetValue<uint32_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::UINT64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint64_t>>(
        expr.value.GetValue<uint64_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::FLOAT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<float_t>>(
        expr.value.GetValue<float_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::FLOAT64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<double_t>>(
        expr.value.GetValue<double_t>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::BOOL8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<bool>>(
        expr.value.GetValue<bool>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
        cudf::duration_D{expr.value.GetValue<duckdb::date_t>().days}, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_s>>(
        cudf::duration_s{expr.value.GetValue<duckdb::timestamp_sec_t>().value}, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ms>>(
        cudf::duration_ms{expr.value.GetValue<duckdb::timestamp_ms_t>().value}, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
        cudf::duration_us{expr.value.GetValue<duckdb::timestamp_tz_t>().value}, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ns>>(
        cudf::duration_ns{expr.value.GetValue<duckdb::timestamp_ns_t>().value}, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL32: {
      auto const scale = numeric::scale_type{-duckdb::DecimalType::GetScale(expr.return_type)};
      auto scalar      = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        expr.value.GetValueUnsafe<numeric::decimal32::rep>(), scale, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL64: {
      auto const scale = numeric::scale_type{-duckdb::DecimalType::GetScale(expr.return_type)};
      auto scalar      = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        expr.value.GetValueUnsafe<numeric::decimal64::rep>(), scale, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL128: {
      auto const scale = numeric::scale_type{-duckdb::DecimalType::GetScale(expr.return_type)};
      duckdb::hugeint_t hugeint_value = expr.value.GetValueUnsafe<duckdb::hugeint_t>();
      __int128_t int128_value = (__int128_t(hugeint_value.upper) << 64) | hugeint_value.lower;
      auto scalar             = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
        int128_value, scale, true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    case cudf::type_id::STRING: {
      auto scalar = std::make_unique<cudf::string_scalar>(
        expr.value.GetValue<std::string>(), true, _stream, _mr);
      return make_execute_result_from_scalar(_ast_tree, _temp_scalars, std::move(scalar), mode);
    }
    default:
      throw duckdb::NotImplementedException("[gpu_expression_executor] Unsupported scalar type: %s",
                                            expr.return_type.ToString());
  }
}

}  // namespace sirius
