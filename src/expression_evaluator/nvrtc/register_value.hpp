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

#pragma once

#include <cudf/types.hpp>

#include <helper/logical_type.hpp>
#include <sirius/exception.hpp>

#include <string>

namespace sirius::nvrtc {

/**
 * @brief Returns whether a cuDF data type is a fixed-width type representable in GPU registers.
 */
[[nodiscard]] inline bool is_supported_fixed_width(cudf::data_type type) noexcept
{
  switch (type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::BOOL8:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return true;
    default: return false;
  }
}

/**
 * @brief Returns the CUDA C++ storage type name for a fixed-width cuDF data type.
 */
[[nodiscard]] inline std::string cuda_type_name(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT8: return "int8_t";
    case cudf::type_id::INT16: return "int16_t";
    case cudf::type_id::INT32: return "int32_t";
    case cudf::type_id::INT64: return "int64_t";
    case cudf::type_id::UINT8: return "uint8_t";
    case cudf::type_id::UINT16: return "uint16_t";
    case cudf::type_id::UINT32: return "uint32_t";
    case cudf::type_id::UINT64: return "uint64_t";
    case cudf::type_id::FLOAT32: return "float";
    case cudf::type_id::FLOAT64: return "double";
    case cudf::type_id::BOOL8: return "bool";
    case cudf::type_id::TIMESTAMP_DAYS: return "int32_t";
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "int64_t";
    case cudf::type_id::DECIMAL32: return "int32_t";
    case cudf::type_id::DECIMAL64: return "int64_t";
    case cudf::type_id::DECIMAL128: return "__int128_t";
    default:
      throw not_implemented_exception("[nvrtc] Unsupported cuDF type for CUDA codegen: {}",
                                      cudf::type_to_name(type));
  }
}

/**
 * @brief Compile-time descriptor of a per-row Register Value in generated CUDA device code.
 *
 * Each Register Value corresponds to a local `reg_val<T>` variable holding `(val, valid, overflow)`
 * in Thread-Private Registers for the current row, together with static guarantees used to elide
 * redundant null masks or overflow checks at the Sink Consumer.
 */
struct register_value {
  std::string var_name;           ///< Identifier of the local `reg_val<T>` variable in CUDA code
  sirius::logical_type sql_type;  ///< Declared SQL logical type of the expression
  cudf::data_type device_type;    ///< Physical cuDF type of the register payload
  bool is_statically_non_null{false};  ///< True if `valid` is guaranteed true for all rows in batch
  bool can_overflow{false};            ///< True if `overflow` may become true on any valid row
};

}  // namespace sirius::nvrtc
