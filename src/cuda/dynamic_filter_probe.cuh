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
// Nothing in this header decides which probe types are acceptable: `dispatch_probe_adapter` asks
// the host predicate `membership_probe_compatible`, and the per-element range rule the adapters
// apply is `sirius::value_fits` from helper/numeric_carrier_rule.hpp, the same rule the host
// narrowing predicates use. What is left here is the mapping from an accepted cudf type to the
// C++ integer the kernel reads it as.
//
// cudf::type_dispatcher is deliberately not used for the (key rep, probe carrier) pair: the
// allowed pairs are a short explicit list and are the correctness surface, so they are spelled
// out here, and the instantiation count stays bounded. Per filter kind: 2 signed reps x 4 signed
// carriers + 2 unsigned reps x 4 unsigned carriers = 16 integral probe kernels. Temporal and
// DECIMAL32/64 probes are read through their integer storage type and reuse those; the __int128
// carrier of DECIMAL128 probes adds one kernel per signed rep (+2) and the string fingerprint
// adapter adds one on the u64 rep (+1), for 19 probe kernels in total.

// sirius
#include <helper/numeric_carrier_rule.hpp>
#include <op/dynamic_filter/dynamic_filter_key_domain.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/hashing.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

// cuco
#include <cuco/hash_functions.cuh>

// rmm
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

// cccl
#include <cub/device/device_for.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <thrust/iterator/transform_iterator.h>

// cucascade
#include <cucascade/error.hpp>

// standard library
#include <cstdint>
#include <memory>
#include <utility>

namespace sirius::op::detail {

using sirius::set_sentinel;

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

//===----------------------------------------------------------------------===//
// Element conversion
//===----------------------------------------------------------------------===//

/// Lossless conversion into the key domain: `sirius::value_fits` decides, so the device converts
/// exactly the values the host narrowing predicates call representable. Widening always succeeds;
/// narrowing only when @p value is representable, and a non-representable value can never equal a
/// stored key. Both types share a signedness (the dispatchers below never mix them).
template <class KeyT, class ProbeT>
__device__ __forceinline__ bool probe_key_convert(ProbeT value, KeyT& out) noexcept
{
  static_assert(cuda::std::is_signed_v<KeyT> == cuda::std::is_signed_v<ProbeT>,
                "probe and key carriers must share a signedness");
  if (!sirius::value_fits<KeyT>(value)) { return false; }
  out = static_cast<KeyT>(value);
  return true;
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
/// of the probe column is materialized. The kernel prologue (probe_validity) writes `false` for a
/// null row before consulting this adapter, so every row passed here owns a valid string view.
struct string_hash_adapter {
  using key_type = std::uint64_t;
  cudf::column_device_view col;
  __device__ __forceinline__ bool operator()(cudf::size_type i, std::uint64_t& out) const noexcept
  {
    out = string_fingerprint_hasher{string_fingerprint_seed}(col.element<cudf::string_view>(i));
    return true;
  }
};

//===----------------------------------------------------------------------===//
// Host-side dispatch
//===----------------------------------------------------------------------===//

/// Maps an accepted probe or build type to the C++ integer the kernel reads its buffer as, and
/// invokes @p fn with a value-initialized instance of it; returns false without invoking @p fn for
/// a type of the other signedness (never reached after `membership_probe_compatible`) or a type
/// with no integer storage. Temporal types read through `sirius::integer_storage_type`;
/// fixed-point types read their unscaled storage integer (scale is the caller's check). Only the
/// carriers of KeyT's signedness are instantiated, which is what bounds the kernel count.
template <class KeyT, class Fn>
bool dispatch_storage_carrier(cudf::data_type t, Fn&& fn)
{
  if constexpr (cuda::std::is_signed_v<KeyT>) {
    switch (sirius::integer_storage_type(t).id()) {
      case cudf::type_id::INT8: fn(std::int8_t{}); return true;
      case cudf::type_id::INT16: fn(std::int16_t{}); return true;
      case cudf::type_id::INT32:
      case cudf::type_id::DECIMAL32: fn(std::int32_t{}); return true;
      case cudf::type_id::INT64:
      case cudf::type_id::DECIMAL64: fn(std::int64_t{}); return true;
      case cudf::type_id::DECIMAL128: fn(__int128_t{}); return true;
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
/// into KeyT, or returns false (= decline) without invoking it. Acceptance is decided by the host
/// predicate `membership_probe_compatible` alone; this function only picks the adapter for an
/// accepted probe. KeyT must be the rep the domain was classified to; a rep/family disagreement is
/// unreachable and also declines.
///
/// @p stream orders any device-side view the adapter needs (a strings column's device view owns
/// a small allocation for its offsets child); @p fn must enqueue its kernel on the same stream so
/// the stream-ordered free of that view lands behind the kernel.
template <class KeyT, class Fn>
bool dispatch_probe_adapter(membership_key_domain const& domain,
                            cudf::column_view const& probe,
                            ::cuda::stream_ref stream,
                            Fn&& fn)
{
  if (!membership_probe_compatible(domain, probe.type())) { return false; }
  if (domain.family == membership_key_family::string_hash) {
    if constexpr (cuda::std::is_same_v<KeyT, std::uint64_t>) {
      // The device view is a host object whose child views live in a stream-ordered device
      // allocation released when it goes out of scope, after fn enqueued its kernel on stream.
      auto const device_view = cudf::column_device_view::create(probe, stream);
      fn(string_hash_adapter{*device_view});
      return true;
    }
    return false;
  }
  // Every other family reads the probe at the integer carrier its type names (its own type, the
  // integer a temporal column stores, or the unscaled storage of a fixed-point column) and hands
  // fn the matching integral adapter; a comparable probe is then bit-identical to an integer one.
  return dispatch_storage_carrier<KeyT>(probe.type(), [&](auto probe_tag) {
    using probe_type = decltype(probe_tag);
    fn(integral_probe_adapter<probe_type, KeyT>{probe.data<probe_type>()});
  });
}

//===----------------------------------------------------------------------===//
// Probe kernel
//===----------------------------------------------------------------------===//

/// The per-row probe shared by the three filters. A row the prior keep-mask killed or whose probe
/// key is null is a definite non-member (null build slots were compacted out and the join never
/// matches nulls); so is a value the adapter cannot represent in the key domain, since every
/// stored key fits it. Only a converted key reaches @p Lookup, which is the filter-specific part:
/// a set `contains`, a Bloom `contains`, or a needle scan.
template <class Adapter, class Lookup>
struct membership_probe_functor {
  using key_type = typename Adapter::key_type;
  Adapter adapt;
  Lookup lookup;
  bool* __restrict__ out;
  std::uint32_t const* __restrict__ prior_words;  // packed 1 bit/row, or null
  probe_validity valid;

  __device__ __forceinline__ void operator()(cudf::size_type idx) const noexcept
  {
    if (!prior_mask_keeps(prior_words, idx) || !valid(idx)) {
      out[idx] = false;
      return;
    }
    key_type key;
    out[idx] = adapt(idx, key) && lookup(key);
  }
};

/// Runs one membership probe over @p probe: selects the adapter for (domain, probe type), or
/// returns null when the probe type is incompatible; otherwise allocates the BOOL8 result and
/// launches `membership_probe_functor` with @p lookup on @p stream. A pinned chunk may store the
/// key narrowed while the filter was published at the native carrier; the kernel converts per
/// element rather than materializing a widened copy. Null probe rows are written as `false`
/// in-kernel, so the mask is non-nullable by construction.
template <class KeyT, class Lookup>
[[nodiscard]] std::unique_ptr<cudf::column> run_membership_probe(
  membership_key_domain const& domain,
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  Lookup lookup)
{
  std::unique_ptr<cudf::column> out;
  auto const n          = probe.size();
  bool const dispatched = dispatch_probe_adapter<KeyT>(domain, probe, stream, [&](auto adapter) {
    out = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::BOOL8}, n, cudf::mask_state::UNALLOCATED, stream, mr);
    auto* const outp = out->mutable_view().data<bool>();
    CUCASCADE_CUDA_TRY(
      cub::DeviceFor::Bulk(n,
                           membership_probe_functor<decltype(adapter), Lookup>{
                             adapter, lookup, outp, prior_mask_words, probe_validity_of(probe)},
                           stream.get()));
  });
  if (!dispatched) { return nullptr; }
  return out;
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
  cudf::column_view const& keys, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
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
                             ::cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr,
                             Fn&& fn)
{
  if (domain.family == membership_key_family::string_hash) {
    if constexpr (cuda::std::is_same_v<KeyT, std::uint64_t>) {
      if (keys.type().id() != cudf::type_id::STRING) { return false; }
      auto const fingerprints = materialize_string_fingerprints(keys, stream, mr);
      auto const* first       = fingerprints->view().data<std::uint64_t>();
      fn(first, first + keys.size());
      return true;
    }
    return false;
  }
  // The build column is a carrier of its own domain by construction; the same acceptance rule
  // the probes use keeps a scale or family disagreement from being read as integers.
  if (!membership_probe_compatible(domain, keys.type())) { return false; }
  bool invoked = false;
  dispatch_storage_carrier<KeyT>(keys.type(), [&](auto carrier_tag) {
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
  });
  return invoked;
}

}  // namespace sirius::op::detail
