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
// out here, and the instantiation count stays bounded. Per filter kind: 2 signed reps x 4 signed
// carriers + 2 unsigned reps x 4 unsigned carriers = 16 integral probe kernels. Temporal and
// DECIMAL32/64 probes are read through their integer storage type and reuse those; the __int128
// carrier of DECIMAL128 probes adds one kernel per signed rep (+2) and the string fingerprint
// adapter adds one on the u64 rep (+1), for 19 probe kernels in total.

// sirius
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/hashing.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

// cuco
#include <cuco/hash_functions.cuh>

// rmm
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

// cccl
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/iterator/transform_iterator.h>

// standard library
#include <cstdint>
#include <memory>
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

/// Integer carriers of the same signedness as KeyT: native ints and their narrowed carriers, the
/// integer storage of temporal columns, and the unscaled storage of same-scale fixed-point probes.
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

/// Hash used for the string family on both sides: `cudf::hashing::xxhash_64` over the build
/// column (a one-column table hashes each row as XXH64(seed) over the string's UTF-8 bytes) and
/// this functor over each probe string. cudf's own device functor for that lives in a detail
/// header (`cudf/hashing/detail/xxhash_64.cuh`) that clang-cuda rejects (constexpr mismatch on
/// its specializations), so this goes straight to the `cuco::xxhash_64` it wraps and feeds it the
/// same byte view: the string's bytes, no length prefix, no terminator. The seed and byte view
/// must stay identical to the build side or the filter silently drops matches; a unit test pins
/// both against a host XXH64 reference.
constexpr std::uint64_t string_fingerprint_seed = cudf::DEFAULT_HASH_SEED;
struct string_fingerprint_hasher {
  cuco::xxhash_64<cudf::string_view> impl;
  __host__ __device__ explicit string_fingerprint_hasher(std::uint64_t seed) : impl{seed} {}
  __device__ std::uint64_t operator()(cudf::string_view const& s) const noexcept
  {
    return impl.compute_hash(reinterpret_cast<cuda::std::byte const*>(s.data()),
                             static_cast<cuda::std::size_t>(s.size_bytes()));
  }
};

/// STRING probes against a fingerprint set: hashes each probe string in-kernel, so no hashed copy
/// of the probe column is materialized. A null probe string is a definite non-member; the kernel
/// prologue (probe_validity) already writes `false` for null rows before consulting the adapter,
/// so the check here only keeps the adapter from reading an offset pair a null row need not own.
struct string_hash_adapter {
  using key_type = std::uint64_t;
  cudf::column_device_view col;
  __device__ __forceinline__ bool operator()(cudf::size_type i, std::uint64_t& out) const noexcept
  {
    if (col.nullable() && !col.is_valid_nocheck(i)) { return false; }
    out = string_fingerprint_hasher{string_fingerprint_seed}(col.element<cudf::string_view>(i));
    return true;
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

/// Invokes @p fn with a value-initialized instance of the unscaled storage integer behind the
/// fixed-point type @p t, or returns false without invoking it. Scale is the caller's check: the
/// storage integers of two scales are not comparable.
template <class Fn>
bool dispatch_decimal_carrier(cudf::data_type t, Fn&& fn)
{
  switch (t.id()) {
    case cudf::type_id::DECIMAL32: fn(std::int32_t{}); return true;
    case cudf::type_id::DECIMAL64: fn(std::int64_t{}); return true;
    case cudf::type_id::DECIMAL128: fn(__int128_t{}); return true;
    default: return false;
  }
}

/// The (key domain, probe type) switch. Invokes @p fn once with the adapter that reads @p probe
/// into KeyT, or returns false (= decline) without invoking it. KeyT must be the rep the domain
/// was classified to; a rep/family disagreement is unreachable and also declines. Each key family
/// owns one arm here, mirrored on the host by membership_probe_compatible.
///
/// @p stream orders any device-side view the adapter needs (a strings column's device view owns
/// a small allocation for its offsets child); @p fn must enqueue its kernel on the same stream so
/// the stream-ordered free of that view lands behind the kernel.
template <class KeyT, class Fn>
bool dispatch_probe_adapter(membership_key_domain const& domain,
                            cudf::column_view const& probe,
                            rmm::cuda_stream_view stream,
                            Fn&& fn)
{
  // Reads the probe at the integer carrier `probe_tag` names (its own type, the integer type a
  // temporal column stores, or the unscaled storage of a fixed-point column) and hands fn the
  // matching integral adapter.
  auto const adapt = [&](auto probe_tag) {
    using probe_type = decltype(probe_tag);
    fn(integral_probe_adapter<probe_type, KeyT>{probe.data<probe_type>()});
  };
  switch (domain.family) {
    case membership_key_family::signed_int:
      if constexpr (cuda::std::is_signed_v<KeyT>) {
        return dispatch_family_carrier<KeyT>(probe.type(), adapt);
      }
      return false;
    case membership_key_family::unsigned_int:
      if constexpr (cuda::std::is_unsigned_v<KeyT>) {
        return dispatch_family_carrier<KeyT>(probe.type(), adapt);
      }
      return false;
    // Temporal keys: the host mirror decides which probe types are comparable (same unit, or a
    // DATE storage carrier); a comparable probe is then bit-identical to an integer one.
    case membership_key_family::date_days:
    case membership_key_family::timestamp:
      if constexpr (cuda::std::is_signed_v<KeyT>) {
        if (!membership_probe_compatible(domain, probe.type())) { return false; }
        return dispatch_family_carrier<KeyT>(membership_storage_type(probe.type()), adapt);
      }
      return false;
    case membership_key_family::decimal:
      // Same scale, any fixed-point width: the unscaled storage is then a plain signed integer
      // and the integral adapter's range check makes a wider carrier exact.
      if constexpr (cuda::std::is_signed_v<KeyT>) {
        if (probe.type().scale() != domain.scale) { return false; }
        return dispatch_decimal_carrier(probe.type(), adapt);
      }
      return false;
    case membership_key_family::string_hash:
      if constexpr (cuda::std::is_same_v<KeyT, std::uint64_t>) {
        if (probe.type().id() != cudf::type_id::STRING) { return false; }
        // The device view is a host object whose child views live in a stream-ordered device
        // allocation released when it goes out of scope, after fn enqueued its kernel on stream.
        auto const device_view = cudf::column_device_view::create(probe, stream);
        fn(string_hash_adapter{*device_view});
        return true;
      }
      return false;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Build side
//===----------------------------------------------------------------------===//

template <class KeyT>
struct convert_to_rep {
  template <class T>
  __host__ __device__ __forceinline__ KeyT operator()(T value) const noexcept
  {
    return static_cast<KeyT>(value);
  }
};

/// Materializes the string family's build fingerprints: one UINT64 per build row, computed by
/// `cudf::hashing::xxhash_64` with the seed the probe adapter uses. The build side is small (it
/// is what the filter exists to summarize), so one hashed copy of it is the intended cost. The
/// column frees stream-ordered on @p stream once it goes out of scope.
[[nodiscard]] inline std::unique_ptr<cudf::column> materialize_string_fingerprints(
  cudf::column_view const& keys, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  return cudf::hashing::xxhash_64(cudf::table_view{{keys}}, string_fingerprint_seed, stream, mr);
}

/// Invokes @p fn(first, last) with a device iterator range yielding the build keys as KeyT,
/// converting a same-family carrier per element so no rep-typed build copy is needed. Integer
/// carriers widen; temporal build columns are read through their integer storage type; the one
/// narrowing pair, DECIMAL128 into the int64 rep, relies on the constructor having verified the
/// column with membership_build_fits_rep. A STRING build column against the `u64` rep is hashed
/// to fingerprints first, and @p fn must enqueue its consumer on @p stream so the fingerprints
/// outlive it (they free stream-ordered when this returns). Returns false without invoking @p fn
/// when @p keys is not a carrier of @p domain that KeyT can hold; classify always picks a rep that
/// holds the build carrier, so that is unreachable in practice.
template <class KeyT, class Fn>
bool with_build_key_iterator(membership_key_domain const& domain,
                             cudf::column_view const& keys,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr,
                             Fn&& fn)
{
  bool invoked         = false;
  auto const emit_from = [&](auto carrier_tag) {
    using carrier_type            = decltype(carrier_tag);
    constexpr bool widens_or_same = sizeof(carrier_type) <= sizeof(KeyT);
    constexpr bool verified_narrowing =
      cuda::std::is_same_v<carrier_type, __int128_t> && cuda::std::is_same_v<KeyT, std::int64_t>;
    if constexpr (widens_or_same || verified_narrowing) {
      auto const first =
        thrust::make_transform_iterator(keys.data<carrier_type>(), convert_to_rep<KeyT>{});
      fn(first, first + keys.size());
      invoked = true;
    }
  };
  switch (domain.family) {
    // Integer carriers, and the integer storage a temporal column sits on (identity for plain
    // integers).
    case membership_key_family::signed_int:
    case membership_key_family::unsigned_int:
    case membership_key_family::date_days:
    case membership_key_family::timestamp:
      dispatch_family_carrier<KeyT>(membership_storage_type(keys.type()), emit_from);
      break;
    case membership_key_family::decimal:
      if constexpr (cuda::std::is_signed_v<KeyT>) {
        if (keys.type().scale() == domain.scale) {
          dispatch_decimal_carrier(keys.type(), emit_from);
        }
      }
      break;
    case membership_key_family::string_hash:
      if constexpr (cuda::std::is_same_v<KeyT, std::uint64_t>) {
        if (keys.type().id() == cudf::type_id::STRING) {
          auto const fingerprints = materialize_string_fingerprints(keys, stream, mr);
          auto const* first       = fingerprints->view().data<std::uint64_t>();
          fn(first, first + keys.size());
          invoked = true;
        }
      }
      break;
  }
  return invoked;
}

}  // namespace sirius::op::detail
