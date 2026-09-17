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

// Shared device helpers and host-side dispatch for the membership probe kernels (IN-list, small
// IN-list, Bloom). Probe keys arrive at whatever carrier the consumer decoded, may carry a prior
// keep-mask and a validity bitmask, and are converted per element into the filter's key rep by a
// *probe adapter* selected on the host from (key domain, probe type); see
// op/dynamic_filter/dynamic_filter_key_domain.hpp for the two axes.
//
// cudf::type_dispatcher is deliberately not used for the (key rep, probe carrier) pair: the
// allowed pairs are a short explicit list and are the correctness surface, so they are spelled
// out here, and the instantiation count stays bounded (per filter kind: 2 signed reps x 4 signed
// carriers + 2 unsigned reps x 4 unsigned carriers = 16 probe kernels).

// sirius
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>

// cudf
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

// cccl
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/iterator/transform_iterator.h>

// standard library
#include <cstdint>
#include <utility>

namespace sirius::op::detail {

//===----------------------------------------------------------------------===//
// Key reps
//===----------------------------------------------------------------------===//

template <membership_key_rep R>
struct rep_type;
template <>
struct rep_type<membership_key_rep::i32> {
  using type = std::int32_t;
};
template <>
struct rep_type<membership_key_rep::i64> {
  using type = std::int64_t;
};
template <>
struct rep_type<membership_key_rep::u32> {
  using type = std::uint32_t;
};
template <>
struct rep_type<membership_key_rep::u64> {
  using type = std::uint64_t;
};
template <membership_key_rep R>
using rep_type_t = typename rep_type<R>::type;

/// Invokes @p fn with a value-initialized instance of the rep's device type.
template <class Fn>
decltype(auto) dispatch_key_rep(membership_key_rep rep, Fn&& fn)
{
  switch (rep) {
    case membership_key_rep::i32: return fn(std::int32_t{});
    case membership_key_rep::i64: return fn(std::int64_t{});
    case membership_key_rep::u32: return fn(std::uint32_t{});
    case membership_key_rep::u64: return fn(std::uint64_t{});
  }
  return fn(std::int32_t{});  // unreachable for a well-formed enum value
}

/// Value a hash set reserves as its empty slot, which therefore cannot be stored. Signed reps use
/// the minimum; unsigned reps use the maximum because 0 is a common real key.
template <class KeyT>
struct set_sentinel {
  static constexpr KeyT value = cuda::std::is_signed_v<KeyT>
                                  ? cuda::std::numeric_limits<KeyT>::min()
                                  : cuda::std::numeric_limits<KeyT>::max();
};

//===----------------------------------------------------------------------===//
// Element conversion
//===----------------------------------------------------------------------===//

/// Lossless conversion into the key domain. Widening always succeeds; narrowing only when @p value
/// is representable, and a non-representable value can never equal a stored key. Both types
/// share a signedness (the dispatchers below never mix them).
template <class KeyT, class ProbeT>
__device__ __forceinline__ bool probe_key_convert(ProbeT value, KeyT& out) noexcept
{
  static_assert(cuda::std::is_signed_v<KeyT> == cuda::std::is_signed_v<ProbeT>,
                "probe and key carriers must share a signedness");
  if constexpr (sizeof(ProbeT) <= sizeof(KeyT)) {
    out = static_cast<KeyT>(value);
    return true;
  } else {
    if constexpr (cuda::std::is_signed_v<ProbeT>) {
      if (value < static_cast<ProbeT>(cuda::std::numeric_limits<KeyT>::min())) { return false; }
    }
    if (value > static_cast<ProbeT>(cuda::std::numeric_limits<KeyT>::max())) { return false; }
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

/// Validity of the probe rows. A null probe key can never equal a build key: admission never
/// routes a null-safe (IS NOT DISTINCT FROM) comparison here and the authoritative join runs with
/// null_equality::UNEQUAL, so a null row is a definite non-member and the kernels write `false`
/// for it instead of propagating the probe's null mask onto the output. @p words is the probe
/// column's own bitmask (null = the column has no nulls), read at `offset + row` because a
/// column_view's mask pointer is not offset-adjusted.
struct probe_validity {
  cudf::bitmask_type const* words = nullptr;
  cudf::size_type offset          = 0;
  __device__ __forceinline__ bool operator()(cudf::size_type row) const noexcept
  {
    return words == nullptr || cudf::bit_is_set(words, offset + row);
  }
};

/// The validity of @p probe as the kernels read it; a column without nulls reads no mask.
inline probe_validity probe_validity_of(cudf::column_view const& probe) noexcept
{
  return probe.has_nulls() ? probe_validity{probe.null_mask(), probe.offset()} : probe_validity{};
}

//===----------------------------------------------------------------------===//
// Probe adapters: read probe[i] at its own carrier, produce a KeyT or "definite non-member"
//===----------------------------------------------------------------------===//

/// Integer carriers of the same signedness as KeyT (native ints and their narrowed carriers).
/// A new key family whose probes are not plain integers adds its own adapter with this shape.
template <class ProbeT, class KeyT>
struct integral_probe_adapter {
  using key_type = KeyT;
  ProbeT const* probe;
  __device__ __forceinline__ bool operator()(cudf::size_type i, KeyT& out) const noexcept
  {
    return probe_key_convert<KeyT>(probe[i], out);
  }
};

//===----------------------------------------------------------------------===//
// Host-side dispatch
//===----------------------------------------------------------------------===//

/// Invokes @p fn with a value-initialized instance of the integer carrier behind @p t when it
/// shares KeyT's signedness, or returns false without invoking it. Covers every carrier a key
/// column can be narrowed to; any other type is a semantic mismatch, not a width one.
template <class KeyT, class Fn>
bool dispatch_family_carrier(cudf::data_type t, Fn&& fn)
{
  if constexpr (cuda::std::is_signed_v<KeyT>) {
    switch (t.id()) {
      case cudf::type_id::INT8: fn(std::int8_t{}); return true;
      case cudf::type_id::INT16: fn(std::int16_t{}); return true;
      case cudf::type_id::INT32: fn(std::int32_t{}); return true;
      case cudf::type_id::INT64: fn(std::int64_t{}); return true;
      default: return false;
    }
  } else {
    switch (t.id()) {
      case cudf::type_id::UINT8: fn(std::uint8_t{}); return true;
      case cudf::type_id::UINT16: fn(std::uint16_t{}); return true;
      case cudf::type_id::UINT32: fn(std::uint32_t{}); return true;
      case cudf::type_id::UINT64: fn(std::uint64_t{}); return true;
      default: return false;
    }
  }
}

/// The (key domain, probe type) switch. Invokes @p fn once with the adapter that reads @p probe
/// into KeyT, or returns false (= decline) without invoking it. KeyT must be the rep the domain
/// was classified to; a rep/family disagreement is unreachable and also declines. Each key family
/// owns one arm here, mirrored on the host by membership_probe_compatible.
template <class KeyT, class Fn>
bool dispatch_probe_adapter(membership_key_domain const& domain,
                            cudf::column_view const& probe,
                            Fn&& fn)
{
  auto const integral_arm = [&]() {
    return dispatch_family_carrier<KeyT>(probe.type(), [&](auto probe_tag) {
      using probe_type = decltype(probe_tag);
      fn(integral_probe_adapter<probe_type, KeyT>{probe.data<probe_type>()});
    });
  };
  switch (domain.family) {
    case membership_key_family::signed_int:
      if constexpr (cuda::std::is_signed_v<KeyT>) { return integral_arm(); }
      return false;
    case membership_key_family::unsigned_int:
      if constexpr (cuda::std::is_unsigned_v<KeyT>) { return integral_arm(); }
      return false;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Build side
//===----------------------------------------------------------------------===//

template <class KeyT>
struct widen_to {
  template <class T>
  __host__ __device__ __forceinline__ KeyT operator()(T value) const noexcept
  {
    return static_cast<KeyT>(value);
  }
};

/// Invokes @p fn(first, last) with a device iterator range yielding the build keys as KeyT,
/// widening a narrower same-family carrier per element so no widened build copy is needed.
/// Returns false without invoking @p fn when @p keys is not a carrier KeyT can hold; classify
/// always picks a rep at least as wide as the build carrier, so that is unreachable in practice.
template <class KeyT, class Fn>
bool with_build_key_iterator(cudf::column_view const& keys, Fn&& fn)
{
  bool invoked = false;
  dispatch_family_carrier<KeyT>(keys.type(), [&](auto carrier_tag) {
    using carrier_type = decltype(carrier_tag);
    if constexpr (sizeof(carrier_type) <= sizeof(KeyT)) {
      auto const first =
        thrust::make_transform_iterator(keys.data<carrier_type>(), widen_to<KeyT>{});
      fn(first, first + keys.size());
      invoked = true;
    }
  });
  return invoked;
}

}  // namespace sirius::op::detail
