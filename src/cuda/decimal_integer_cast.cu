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

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <expression_evaluator/decimal_integer_cast.hpp>
#include <sirius/exception.hpp>

#include <cstdint>
#include <limits>

namespace sirius {
namespace {

struct integer_bounds {
  __int128_t minimum;
  __int128_t maximum;
};

integer_bounds target_bounds(logical_type const& target)
{
  switch (target.id()) {
    case type_id::TINYINT:
      return {std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()};
    case type_id::SMALLINT:
      return {std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()};
    case type_id::INTEGER:
      return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
    case type_id::BIGINT:
      return {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()};
    case type_id::HUGEINT:
      return {std::numeric_limits<__int128_t>::min(), std::numeric_limits<__int128_t>::max()};
    case type_id::UTINYINT: return {0, std::numeric_limits<uint8_t>::max()};
    case type_id::USMALLINT: return {0, std::numeric_limits<uint16_t>::max()};
    case type_id::UINTEGER: return {0, std::numeric_limits<uint32_t>::max()};
    case type_id::UBIGINT:
    case type_id::UHUGEINT: return {0, std::numeric_limits<uint64_t>::max()};
    default: throw invalid_input_exception("Expected an integer cast target");
  }
}

template <typename Rep>
std::unique_ptr<cudf::column> convert(cudf::column_view const& input,
                                      logical_type const& target,
                                      bool try_cast,
                                      __int128_t divisor,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const bounds = target_bounds(target);
  // Rounding to an integer never needs a wider representation than the input. Keep that width
  // until the checked final cast; widening every DECIMAL32/64 to DECIMAL128 is unnecessary.
  auto rounded = cudf::make_fixed_point_column(
    cudf::data_type{input.type().id(), 0}, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto validity = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  int const supported_carrier = 0;
  rmm::device_scalar<int> unsupported_carrier(supported_carrier, stream, mr);
  auto* unsupported   = unsupported_carrier.data();
  auto const uhugeint = target.id() == type_id::UHUGEINT;
  auto const* data    = input.data<Rep>();
  auto const* mask    = input.null_mask();
  auto const offset   = input.offset();
  auto* valid         = validity->mutable_view().data<bool>();
  auto const begin    = thrust::make_counting_iterator<cudf::size_type>(0);
  thrust::transform(rmm::exec_policy(stream, mr),
                    begin,
                    begin + input.size(),
                    rounded->mutable_view().data<Rep>(),
                    [=] __device__(cudf::size_type i) -> Rep {
                      valid[i] = false;
                      if (mask && !cudf::bit_is_set(mask, offset + i)) { return 0; }
                      auto const raw       = static_cast<__int128_t>(data[i]);
                      auto result          = raw / divisor;
                      auto const remainder = raw % divisor;
                      // Quotient/remainder avoids adding half a divisor to a value near its
                      // representation limit. 10^38 and its half both fit signed 128 bits.
                      if (divisor > 1) {
                        if (remainder >= divisor / 2) { ++result; }
                        if (remainder <= -divisor / 2) { --result; }
                      }
                      if (result < bounds.minimum || result > bounds.maximum) {
                        // A positive UHUGEINT beyond UINT64 is valid SQL, not a TRY_CAST NULL.
                        if (uhugeint && result > bounds.maximum) { atomicExch(unsupported, 1); }
                        return 0;
                      }
                      valid[i] = true;
                      return static_cast<Rep>(result);
                    });
  auto [output_mask, null_count] = cudf::bools_to_mask(validity->view(), stream, mr);
  if ((!try_cast && null_count > input.null_count()) ||
      (uhugeint && unsupported_carrier.value(stream) != 0)) {
    // CASE evaluates branches eagerly on the GPU. Replay also preserves DuckDB's behavior when
    // the overflowing cast belongs to a branch the query does not select.
    throw not_implemented_exception(
      "DECIMAL value out of range for {} on the GPU; falling back to DuckDB", target.to_string());
  }
  rounded->set_null_mask(std::move(*output_mask), null_count);
  auto const output_type = require_cudf_type(target);
  if (rounded->type() == output_type) { return rounded; }
  return cudf::cast(rounded->view(), output_type, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> cast_decimal_to_integer(cudf::column_view const& input,
                                                      logical_type const& target,
                                                      bool try_cast,
                                                      ::cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const scale = -static_cast<int64_t>(input.type().scale());
  if (!cudf::is_fixed_point(input.type()) || scale < 0 || scale > 38 || !target.is_integer()) {
    throw invalid_input_exception("Expected a SQL DECIMAL input and integer cast target");
  }
  if (input.is_empty()) { return cudf::make_empty_column(require_cudf_type(target)); }
  __int128_t divisor = 1;
  for (int64_t i = 0; i < scale; ++i) {
    divisor *= 10;
  }
  switch (input.type().id()) {
    case cudf::type_id::DECIMAL32:
      return convert<int32_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::DECIMAL64:
      return convert<int64_t>(input, target, try_cast, divisor, stream, mr);
    case cudf::type_id::DECIMAL128:
      return convert<__int128_t>(input, target, try_cast, divisor, stream, mr);
    default: throw internal_exception("Unexpected DECIMAL representation");
  }
}

}  // namespace sirius
