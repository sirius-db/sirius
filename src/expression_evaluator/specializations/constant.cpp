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
#include <expression/ast/node.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <helper/logical_type.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/exception.hpp>

// cudf
#include <cudf/ast/expressions.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <string>
#include <variant>

namespace {
using evaluate_result = ::sirius::expression_evaluator::evaluate_result;
using ast_node        = ::sirius::expression_evaluator::ast_result;
using evaluation_mode = ::sirius::expression_evaluator::evaluation_mode;

}  // namespace

namespace sirius {
using evaluate_result = expression_evaluator::evaluate_result;

evaluate_result expression_evaluator::evaluate(sirius::ast::constant const& alt,
                                               evaluation_mode mode)
{
  auto const cudf_type = sirius::get_cudf_type(alt.return_type());
  bool const is_valid  = !std::holds_alternative<sirius::null_value>(alt.payload);

  switch (cudf_type.id()) {
    case cudf::type_id::INT8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int8_t>>(
        is_valid ? std::get<int8_t>(alt.payload) : int8_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::INT16: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int16_t>>(
        is_valid ? std::get<int16_t>(alt.payload) : int16_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::INT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int32_t>>(
        is_valid ? std::get<int32_t>(alt.payload) : int32_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::INT64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<int64_t>>(
        is_valid ? std::get<int64_t>(alt.payload) : int64_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::UINT8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint8_t>>(
        is_valid ? std::get<uint8_t>(alt.payload) : uint8_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::UINT16: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint16_t>>(
        is_valid ? std::get<uint16_t>(alt.payload) : uint16_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::UINT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint32_t>>(
        is_valid ? std::get<uint32_t>(alt.payload) : uint32_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::UINT64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<uint64_t>>(
        is_valid ? std::get<uint64_t>(alt.payload) : uint64_t{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::FLOAT32: {
      auto scalar = std::make_unique<cudf::numeric_scalar<float>>(
        is_valid ? std::get<float>(alt.payload) : float{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::FLOAT64: {
      auto scalar = std::make_unique<cudf::numeric_scalar<double>>(
        is_valid ? std::get<double>(alt.payload) : double{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::BOOL8: {
      auto scalar = std::make_unique<cudf::numeric_scalar<bool>>(
        is_valid ? std::get<bool>(alt.payload) : false, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_D>>(
        cudf::duration_D{is_valid ? std::get<sirius::date_value>(alt.payload).days : 0},
        is_valid,
        _stream,
        _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_SECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_s>>(
        cudf::duration_s{is_valid ? std::get<sirius::timestamp_sec_value>(alt.payload).value : 0},
        is_valid,
        _stream,
        _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_MILLISECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ms>>(
        cudf::duration_ms{is_valid ? std::get<sirius::timestamp_ms_value>(alt.payload).value : 0},
        is_valid,
        _stream,
        _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_us>>(
        cudf::duration_us{is_valid ? std::get<sirius::timestamp_us_value>(alt.payload).value : 0},
        is_valid,
        _stream,
        _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::TIMESTAMP_NANOSECONDS: {
      auto scalar = std::make_unique<cudf::timestamp_scalar<cudf::timestamp_ns>>(
        cudf::duration_ns{is_valid ? std::get<sirius::timestamp_ns_value>(alt.payload).value : 0},
        is_valid,
        _stream,
        _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL32: {
      auto const scale =
        numeric::scale_type{-static_cast<int32_t>(alt.return_type().decimal_scale())};
      auto const rep =
        is_valid ? std::get<sirius::decimal32>(alt.payload).value : numeric::decimal32::rep{};
      auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
        rep, scale, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL64: {
      auto const scale =
        numeric::scale_type{-static_cast<int32_t>(alt.return_type().decimal_scale())};
      auto const rep =
        is_valid ? std::get<sirius::decimal64>(alt.payload).value : numeric::decimal64::rep{};
      auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
        rep, scale, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::DECIMAL128: {
      auto const scale =
        numeric::scale_type{-static_cast<int32_t>(alt.return_type().decimal_scale())};
      __int128_t const rep =
        is_valid ? std::get<sirius::decimal128>(alt.payload).value : __int128_t{};
      auto scalar = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
        rep, scale, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    case cudf::type_id::STRING: {
      auto scalar = std::make_unique<cudf::string_scalar>(
        is_valid ? std::get<std::string>(alt.payload) : std::string{}, is_valid, _stream, _mr);
      return finish_scalar(std::move(scalar), mode);
    }
    default:
      throw not_implemented_exception("[expression_evaluator] Unsupported scalar type: %s",
                                      alt.return_type().to_string());
  }
}

}  // namespace sirius
