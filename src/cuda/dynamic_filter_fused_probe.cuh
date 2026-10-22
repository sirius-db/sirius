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

/**
 * @file dynamic_filter_fused_probe.cuh
 * @brief Device lookups shared by ordinary membership probes and fused mask rounds
 *
 * Descriptors borrow device storage from the filter and probe owners. Their data pointer includes
 * the column offset; validity uses the original mask and row offset. Only the tagged union member
 * is active. Host descriptors can be destroyed after launch arguments have been copied, while the
 * borrowed storage must remain alive until GPU completion.
 */

#pragma once

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policy.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_set.cuh>
#include <cuco/storage.cuh>
#include <cuda/dynamic_filter_probe.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/functional>

#include <op/dynamic_filter/dynamic_filter_device.hpp>

#include <cassert>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace sirius::op::detail {

/// @brief The number of threads per key in a fused probe.
constexpr std::size_t kCgSize = 1;
static_assert(kCgSize == 1, "cuco::static_set requires kCgSize==1 for device_for bulk iteration");

/// @brief The CuCo bucket size for the static_set used in fused probes.
constexpr std::int32_t kBucketSize = 4;

template <class KeyT>
using set_alloc = sirius::rmm_cuco_allocator<KeyT>;

template <class KeyT>
using set_type = cuco::static_set<KeyT,
                                  cuco::extent<std::size_t>,
                                  cuda::thread_scope_device,
                                  cuda::std::equal_to<KeyT>,
                                  cuco::double_hashing<kCgSize, cuco::default_hash_function<KeyT>>,
                                  set_alloc<KeyT>,
                                  cuco::storage<kBucketSize>>;

using bloom_alloc = sirius::rmm_cuco_allocator<cuda::std::byte>;

template <class KeyT>
using sirius_bloom = cuco::bloom_filter<KeyT,
                                        cuco::extent<std::size_t>,
                                        cuda::thread_scope_device,
                                        cuco::bloom_filter_policy<KeyT>,
                                        bloom_alloc>;

//===----------------------------------------------------------------------===//
// Lookup functors: set, bloom, and small IN-list
//===----------------------------------------------------------------------===//
/**
 * @brief Tests a converted key against a hash set, conservatively accepting its empty-key sentinel
 *
 * The set cannot store set_sentinel<key_type>::value, so accepting that key avoids dropping a
 * possible join match. Strings are represented by 64-bit fingerprints, so collisions can retain
 * nonmatching strings.
 *
 * @tparam SetRef Device reference to a cuco::static_set with contains enabled
 */
template <class SetRef>
struct set_lookup {
  using key_type = typename SetRef::key_type;
  SetRef set;
  __device__ __forceinline__ bool operator()(key_type key) const noexcept
  {
    return set.contains(key) || key == detail::set_sentinel<key_type>::value;
  }
};

/**
 * @brief Tests whether a converted key may be present in a Bloom filter
 *
 * @tparam FilterRef Device reference to a cuco::bloom_filter
 */
template <class FilterRef>
struct bloom_lookup {
  using key_type = typename FilterRef::key_type;
  FilterRef ref;
  __device__ __forceinline__ bool operator()(key_type key) const noexcept
  {
    return ref.contains(key);
  }
};

/**
 * @brief Tests a converted key against the small IN-list's stored keys (needles)
 *
 * Compares against all m keys without reserving a sentinel value. Integer comparisons are exact;
 * strings are represented by 64-bit fingerprints, so collisions can retain nonmatching strings.
 *
 * @tparam KeyT Integer type of the stored keys or string fingerprints
 */
template <class KeyT>
struct needle_lookup {
  using key_type = KeyT;
  KeyT const* __restrict__ needles;
  int m;
  __device__ __forceinline__ bool operator()(KeyT x) const noexcept
  {
    bool hit = false;
    for (int j = 0; j < m; ++j) {
      hit |= (x == needles[j]);
    }
    return hit;
  }
};

template <class Key>
using bloom_ref = decltype(std::declval<sirius_bloom<Key> const&>().ref());

template <class Key>
using set_ref = decltype(std::declval<set_type<Key> const&>().ref(cuco::contains));

/// @brief The kind of lookup a fused probe descriptor performs
enum class fused_probe_kind { bloom_i32, bloom_i64, set_i32, set_i64, needles_i32, needles_i64 };

/// @brief The physical storage integer a fused probe column is read as.
enum class fused_carrier { i8, i16, i32, i64 };

/// @brief Maps a storage integer to its fused_carrier; any other type fails to compile.
template <class T>
constexpr fused_carrier fused_carrier_of() noexcept
{
  if constexpr (std::is_same_v<T, std::int8_t>) {
    return fused_carrier::i8;
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return fused_carrier::i16;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return fused_carrier::i32;
  } else {
    static_assert(std::is_same_v<T, std::int64_t>, "no fused carrier for this storage type");
    return fused_carrier::i64;
  }
}

/**
 * @brief A fused probe descriptor that encapsulates the lookup logic for a specific type of probe.
 */
struct fused_probe {
  fused_probe_kind kind = fused_probe_kind::needles_i32;
  fused_carrier carrier = fused_carrier::i8;
  void const* data      = nullptr;
  probe_validity valid;
  union lookup_storage {
    bloom_lookup<bloom_ref<std::int32_t>> bloom_i32;
    bloom_lookup<bloom_ref<std::int64_t>> bloom_i64;
    set_lookup<set_ref<std::int32_t>> set_i32;
    set_lookup<set_ref<std::int64_t>> set_i64;
    needle_lookup<std::int32_t> needles_i32;
    needle_lookup<std::int64_t> needles_i64;

    lookup_storage() : needles_i32{} {}
  } lookup;
};
static_assert(std::is_trivially_copyable_v<fused_probe>);

//===----------------------------------------------------------------------===//
// Fused probe descriptor construction: bloom, set, and small IN-list
//===----------------------------------------------------------------------===//
template <class Ref>
void store_fused_lookup(fused_probe& out, bloom_lookup<Ref> lookup)
{
  if constexpr (std::is_same_v<typename Ref::key_type, std::int32_t>) {
    std::construct_at(&out.lookup.bloom_i32, lookup);
    out.kind = fused_probe_kind::bloom_i32;
  } else {
    std::construct_at(&out.lookup.bloom_i64, lookup);
    out.kind = fused_probe_kind::bloom_i64;
  }
}

template <class Ref>
void store_fused_lookup(fused_probe& out, set_lookup<Ref> lookup)
{
  if constexpr (std::is_same_v<typename Ref::key_type, std::int32_t>) {
    std::construct_at(&out.lookup.set_i32, lookup);
    out.kind = fused_probe_kind::set_i32;
  } else {
    std::construct_at(&out.lookup.set_i64, lookup);
    out.kind = fused_probe_kind::set_i64;
  }
}

template <class Key>
void store_fused_lookup(fused_probe& out, needle_lookup<Key> lookup)
{
  if constexpr (std::is_same_v<Key, std::int32_t>) {
    std::construct_at(&out.lookup.needles_i32, lookup);
    out.kind = fused_probe_kind::needles_i32;
  } else {
    std::construct_at(&out.lookup.needles_i64, lookup);
    out.kind = fused_probe_kind::needles_i64;
  }
}

template <class Lookup>
[[nodiscard]] bool describe_fused_probe(cudf::column_view const& probe,
                                        Lookup lookup,
                                        fused_probe& out)
{
  using key_type = typename Lookup::key_type;
  if constexpr (!std::is_same_v<key_type, std::int32_t> &&
                !std::is_same_v<key_type, std::int64_t>) {
    return false;
  } else {
    bool ready = false;
    dispatch_storage_carrier<key_type>(probe.type(), [&](auto carrier) {
      using carrier_type = decltype(carrier);
      if constexpr (sizeof(carrier_type) <= sizeof(std::int64_t)) {
        auto const* data = probe.data<carrier_type>();
        assert(reinterpret_cast<std::uintptr_t>(data) % alignof(carrier_type) == 0);
        store_fused_lookup(out, lookup);
        out.data    = data;
        out.valid   = probe_validity_of(probe);
        out.carrier = fused_carrier_of<carrier_type>();
        ready       = true;
      }
    });
    return ready;
  }
}

// Descriptors are built only by describe_fused_probe, whose kind and carrier come from compile-time
// mappings, so the switches below cover every value a descriptor can hold. If an unhandled value
// ever reached them, the row is kept.

template <class Carrier, class Lookup>
__device__ __forceinline__ bool fused_carrier_keeps(fused_probe const& probe,
                                                    Lookup const& lookup,
                                                    cudf::size_type row) noexcept
{
  return membership_row_keeps(
    integral_probe_adapter<Carrier, typename Lookup::key_type>{
      static_cast<Carrier const*>(probe.data)},
    lookup,
    probe.valid,
    row);
}

template <class Lookup>
__device__ __forceinline__ bool fused_lookup_keeps(fused_probe const& probe,
                                                   Lookup const& lookup,
                                                   cudf::size_type row) noexcept
{
  switch (probe.carrier) {
    case fused_carrier::i8: return fused_carrier_keeps<std::int8_t>(probe, lookup, row);
    case fused_carrier::i16: return fused_carrier_keeps<std::int16_t>(probe, lookup, row);
    case fused_carrier::i32: return fused_carrier_keeps<std::int32_t>(probe, lookup, row);
    case fused_carrier::i64: return fused_carrier_keeps<std::int64_t>(probe, lookup, row);
  }
  return true;
}

__device__ __forceinline__ bool fused_probe_keeps(fused_probe const& probe,
                                                  cudf::size_type row) noexcept
{
  switch (probe.kind) {
    case fused_probe_kind::bloom_i32: return fused_lookup_keeps(probe, probe.lookup.bloom_i32, row);
    case fused_probe_kind::bloom_i64: return fused_lookup_keeps(probe, probe.lookup.bloom_i64, row);
    case fused_probe_kind::set_i32: return fused_lookup_keeps(probe, probe.lookup.set_i32, row);
    case fused_probe_kind::set_i64: return fused_lookup_keeps(probe, probe.lookup.set_i64, row);
    case fused_probe_kind::needles_i32:
      return fused_lookup_keeps(probe, probe.lookup.needles_i32, row);
    case fused_probe_kind::needles_i64:
      return fused_lookup_keeps(probe, probe.lookup.needles_i64, row);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Filter glue
//===----------------------------------------------------------------------===//

// The Bloom, hash-set and small IN-list filters probe through these two functions. `Store` is the
// filter's private replica owner: `store.with_lookup(domain, device_id, fn)` calls `fn(lookup)`
// once with a lookup over its replica on the resolved `device_id`, and does nothing when it holds
// no usable replica there. A null `store` declines. Each path checks probe compatibility once:
// `run_membership_probe` through `dispatch_probe_adapter`, and the fused path here, since it never
// reaches that dispatch.

/**
 * @brief Creates a nonnullable BOOL8 mask column and enqueues its computation
 *
 * The returned column has one boolean value per probe row: true means keep the row as a possible
 * match, and false means reject it. Null probe keys, values outside the key representation, and
 * rows rejected by either supplied prior mask get false in the output column. Prior masks index
 * rows relative to the probe view, including when it is sliced.
 *
 * The resolved device must be current and hold the probe and prior masks. Their buffers and the
 * selected filter replica must remain alive until the submitted stream work completes.
 *
 * @tparam Store Filter replica owner exposing with_lookup(domain, device_id, fn)
 * @param store Replica owner, or nullptr to decline filtering
 * @param domain Filter key domain governing probe compatibility and conversion
 * @param probe Column of keys to test
 * @param prior_mask_words Optional device keep mask packed into 32-bit words, one bit per probe
 * row; 1 keeps the row and nullptr imposes no restriction
 * @param prior_keep Optional device array of one bool per probe row; true keeps the row and nullptr
 * imposes no restriction
 * @param device_id Device holding the replica; a negative ID selects the current device
 * @param stream Stream on which to allocate and compute the mask
 * @param mr Device memory resource for the result column
 * @return Owning pointer to the mask column with probe.size() rows, or nullptr if the store or
 * replica is unavailable or the probe is incompatible
 */
template <class Store>
[[nodiscard]] std::unique_ptr<cudf::column> compute_membership_mask(
  Store const* store,
  membership_key_domain const& domain,
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  bool const* prior_keep,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::unique_ptr<cudf::column> result;
  if (!store) { return result; }
  store->with_lookup(domain, resolve_dynamic_filter_device_id(device_id), [&](auto lookup) {
    using key_type = typename decltype(lookup)::key_type;
    result         = run_membership_probe<key_type>(
      domain, probe, prior_mask_words, stream, mr, lookup, prior_keep);
  });
  return result;
}

/**
 * @brief Describes a membership probe for a fused kernel without submitting GPU work
 *
 * Allocates no device storage. A successful descriptor borrows the probe buffers and filter
 * replica, which must remain alive until kernels using it complete. Declining leaves out unchanged
 * and does not imply that compute_membership_mask must decline.
 *
 * @tparam Store Filter replica owner exposing with_lookup(domain, device_id, fn)
 * @param store Replica owner, or nullptr to decline fusion
 * @param domain Filter key domain governing probe compatibility and conversion
 * @param probe Column of keys to test
 * @param device_id Device holding the replica; a negative ID selects the current device
 * @param[out] out Destination for a supported descriptor
 * @return True if the descriptor is ready; false if the store or replica is unavailable, the probe
 * is incompatible, or its representation is unsupported by fusion
 */
template <class Store>
[[nodiscard]] bool prepare_fused_membership_probe(Store const* store,
                                                  membership_key_domain const& domain,
                                                  cudf::column_view const& probe,
                                                  int device_id,
                                                  fused_probe& out)
{
  if (!store || !membership_probe_compatible(domain, probe.type())) { return false; }
  bool ready = false;
  store->with_lookup(domain, resolve_dynamic_filter_device_id(device_id), [&](auto lookup) {
    ready = describe_fused_probe(probe, lookup, out);
  });
  return ready;
}

}  // namespace sirius::op::detail
