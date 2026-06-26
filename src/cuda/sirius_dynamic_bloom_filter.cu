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

// GPU Bloom-filter backing for sirius_dynamic_bloom_filter. Kept in a .cu (compiled by nvcc) so the
// cuCollections device code never reaches host translation units — the class is PIMPL'd in the
// header.

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policies.cuh>
#include <cuco/hash_functions.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/cstddef>
#include <cuda/stream_ref>

#include <op/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace sirius::op {

namespace {
// ~16 bits/key → num_blocks ≈ keys/16
constexpr std::size_t kBitsPerBlock     = 256;
constexpr std::size_t kTargetBitsPerKey = 16;

std::size_t blocks_for(std::size_t num_keys)
{
  auto const bits   = std::max<std::size_t>(num_keys, 1) * kTargetBitsPerKey;
  auto const blocks = (bits + kBitsPerBlock - 1) / kBitsPerBlock;
  return std::max<std::size_t>(blocks, 1);
}

using bloom_alloc = sirius::rmm_cuco_allocator<cuda::std::byte>;

template <class KeyT>
using arrow_policy = cuco::arrow_filter_policy<KeyT>;
template <class KeyT>
using default_policy = cuco::default_filter_policy<cuco::xxhash_64<KeyT>, std::uint32_t, 8>;

template <class KeyT, class Policy>
using bloom_filter_for = cuco::
  bloom_filter<KeyT, cuco::extent<std::size_t>, cuda::thread_scope_device, Policy, bloom_alloc>;

template <class KeyT>
struct typed_bloom {
  std::unique_ptr<bloom_filter_for<KeyT, arrow_policy<KeyT>>> arrow;
  std::unique_ptr<bloom_filter_for<KeyT, default_policy<KeyT>>> standard;

  typed_bloom(std::size_t num_blocks, rmm::device_async_resource_ref mr, cuda::stream_ref stream)
  {
    if (num_blocks <= arrow_policy<KeyT>::max_filter_blocks) {
      arrow.reset(new bloom_filter_for<KeyT, arrow_policy<KeyT>>{
        cuco::extent<std::size_t>{num_blocks}, {}, {}, bloom_alloc{mr}, stream});
    } else {
      standard.reset(new bloom_filter_for<KeyT, default_policy<KeyT>>{
        cuco::extent<std::size_t>{num_blocks}, {}, {}, bloom_alloc{mr}, stream});
    }
  }

  template <class Fn>
  void with(Fn&& fn)
  {
    if (arrow) {
      fn(*arrow);
    } else {
      fn(*standard);
    }
  }
};
}  // namespace

// Exactly one typed Bloom is populated, chosen by the key width.
struct sirius_dynamic_bloom_filter::impl {
  std::unique_ptr<typed_bloom<std::int32_t>> b32;
  std::unique_ptr<typed_bloom<std::int64_t>> b64;
};

bool sirius_dynamic_bloom_filter::supports(cudf::data_type t) noexcept
{
  return t.id() == cudf::type_id::INT32 || t.id() == cudf::type_id::INT64;
}

std::size_t sirius_dynamic_bloom_filter::estimated_bytes(std::size_t num_keys) noexcept
{
  // Mirrors blocks_for(): each block is kBitsPerBlock bits = kBitsPerBlock/8 bytes.
  return blocks_for(num_keys) * (kBitsPerBlock / 8);
}

sirius_dynamic_bloom_filter::sirius_dynamic_bloom_filter(cudf::column_view const& keys,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (!supports(keys.type())) {
    throw std::invalid_argument(
      "[sirius_dynamic_bloom_filter] unsupported key type (INT32 or INT64).");
  }
  auto const n = static_cast<std::size_t>(keys.size());
  cuda::stream_ref const s{stream.value()};
  auto const num_blocks = blocks_for(n);
  _impl                 = std::make_unique<impl>();

  // Insert every build key. Build keys are non-null (FK/PK); a null slot would only add a stray
  // fingerprint — a negligible false-positive bump, never a dropped match.
  switch (keys.type().id()) {
    case cudf::type_id::INT32: {
      _impl->b32    = std::make_unique<typed_bloom<std::int32_t>>(num_blocks, mr, s);
      auto const* d = keys.data<std::int32_t>();
      _impl->b32->with([&](auto& f) { f.add(d, d + n, s); });
      break;
    }
    case cudf::type_id::INT64: {
      _impl->b64    = std::make_unique<typed_bloom<std::int64_t>>(num_blocks, mr, s);
      auto const* d = keys.data<std::int64_t>();
      _impl->b64->with([&](auto& f) { f.add(d, d + n, s); });
      break;
    }
    default: break;  // unreachable: supports() gates the type
  }
}

sirius_dynamic_bloom_filter::~sirius_dynamic_bloom_filter() = default;

std::unique_ptr<cudf::column> sirius_dynamic_bloom_filter::compute_mask(
  cudf::column_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (!supports(probe.type())) { return nullptr; }
  auto const n = probe.size();
  auto out     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cuda::stream_ref const s{stream.value()};
  auto* const outp = out->mutable_view().data<bool>();

  switch (probe.type().id()) {
    case cudf::type_id::INT32: {
      if (!_impl->b32) { return nullptr; }
      auto const* d = probe.data<std::int32_t>();
      _impl->b32->with([&](auto& f) { f.contains(d, d + n, outp, s); });
      break;
    }
    case cudf::type_id::INT64: {
      if (!_impl->b64) { return nullptr; }
      auto const* d = probe.data<std::int64_t>();
      _impl->b64->with([&](auto& f) { f.contains(d, d + n, outp, s); });
      break;
    }
    default: return nullptr;
  }

  if (probe.nullable() && probe.null_count() > 0) {
    out->set_null_mask(cudf::copy_bitmask(probe, stream, mr), probe.null_count());
  }
  return out;
}

}  // namespace sirius::op
