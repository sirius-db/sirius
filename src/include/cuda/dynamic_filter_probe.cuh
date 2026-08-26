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

// Shared device/host helpers for the membership dynamic-filter probe kernels
// (IN-list, small IN-list, Bloom). Two concerns live here so the three .cu
// implementations stay in lockstep:
//
//  1. Heterogeneous probe keys. A consumer may probe with a column whose
//     integer carrier is narrower or wider than the build-key type — e.g. a
//     compressed-materialization pin stores BIGINT l_partkey as an INT32
//     carrier while the set was built from the native INT64 p_partkey.
//     @ref probe_key_convert widens/narrows per element in-kernel, so no
//     materialized cast of the probe column is ever needed.
//
//  2. Prior keep-masks. A probe may receive the packed keep-mask of the
//     conjuncts already evaluated (fused scan-filter wave 1); rows the prior
//     mask killed skip the set/Bloom lookup entirely. @ref prior_mask_keeps
//     reads the standard packed convention (bit `row % 32` of word
//     `row / 32`, 1 = keep) shared by cuDF bitmasks and the fused
//     selection-wave masks.

// cudf
#include <cudf/types.hpp>

// cccl
#include <cuda/std/limits>

// standard library
#include <cstdint>

namespace sirius::op::detail {

/**
 * @brief Lossless per-element conversion of a probe value into the filter's key domain.
 *
 * Widening always succeeds. Narrowing succeeds only when @p value is representable in @c KeyT;
 * a non-representable value can never equal a stored key, so callers treat a failed conversion
 * as a definite non-member.
 *
 * @return true when @p out holds the converted value, false when @p value is not representable.
 */
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

/**
 * @brief True when @p row survives the optional prior keep-mask.
 *
 * @p words is packed 1-bit-per-row (bit `row % 32` of word `row / 32`, 1 = keep) — the shared
 * cuDF-bitmask / fused-selection-wave convention. A null @p words means no prior restriction.
 */
__device__ __forceinline__ bool prior_mask_keeps(std::uint32_t const* words,
                                                 cudf::size_type row) noexcept
{
  return words == nullptr ||
         ((words[static_cast<std::size_t>(row) >> 5] >> (static_cast<std::uint32_t>(row) & 31U)) &
          1U) != 0U;
}

/**
 * @brief Invoke @p fn with a value-initialized instance of the signed integer type behind @p t.
 *
 * Covers the integer carriers compressed materialization can narrow a key column to. Returns
 * false — without invoking @p fn — for any other type; membership filters answer nullptr there
 * (a non-integer probe against an integer key set is a semantic mismatch, not a width mismatch).
 */
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
