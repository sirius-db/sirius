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

#pragma once

// Shared device helpers for the membership probe kernels (IN-list, small IN-list, Bloom): probe
// keys arrive at whatever integer carrier the consumer decoded, and may carry a prior keep-mask.

// cudf
#include <cudf/types.hpp>

// cccl
#include <cuda/std/limits>

// standard library
#include <cstdint>

namespace sirius::op::detail {

/// Lossless conversion into the key domain. Widening always succeeds; narrowing only when @p value
/// is representable, and a non-representable value can never equal a stored key.
template <class KeyT, class ProbeT>
__device__ __forceinline__ bool probe_key_convert(ProbeT value, KeyT& out) noexcept
{
  if constexpr (sizeof(ProbeT) <= sizeof(KeyT)) {
    out = static_cast<KeyT>(value);
    return true;
  } else {
    if (value < static_cast<ProbeT>(cuda::std::numeric_limits<KeyT>::min()) ||
        value > static_cast<ProbeT>(cuda::std::numeric_limits<KeyT>::max())) {
      return false;
    }
    out = static_cast<KeyT>(value);
    return true;
  }
}

/// @p words is packed 1 bit/row (bit `row % 32` of word `row / 32`, 1 = keep); null = no prior.
__device__ __forceinline__ bool prior_mask_keeps(std::uint32_t const* words,
                                                 cudf::size_type row) noexcept
{
  return words == nullptr ||
         ((words[static_cast<std::size_t>(row) >> 5] >> (static_cast<std::uint32_t>(row) & 31U)) &
          1U) != 0U;
}

/// Invokes @p fn with a value-initialized instance of the signed integer type behind @p t, or
/// returns false without invoking it. Covers every carrier a key column can be narrowed to; any
/// other type is a semantic mismatch, not a width one, and the filter declines.
template <class Fn>
bool dispatch_signed_integer_probe(cudf::data_type t, Fn&& fn)
{
  switch (t.id()) {
    case cudf::type_id::INT8: fn(std::int8_t{}); return true;
    case cudf::type_id::INT16: fn(std::int16_t{}); return true;
    case cudf::type_id::INT32: fn(std::int32_t{}); return true;
    case cudf::type_id::INT64: fn(std::int64_t{}); return true;
    default: return false;
  }
}

}  // namespace sirius::op::detail
