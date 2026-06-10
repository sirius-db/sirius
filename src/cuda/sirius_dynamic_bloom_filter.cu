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
#include <cuda/std/cstddef>
#include <cuda/stream_ref>

#include <op/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace sirius::op {

namespace {
// 256-bit blocks (8 × uint32 words, the cuco default). ~16 bits/key → num_blocks ≈ keys/16, a few
// MB even for tens of millions of keys, with a sub-percent false-positive rate. False positives are
// safe (they only let through a few extra probe rows, which the exact join then discards); false
// negatives never happen, so the filter never drops a true match.
constexpr std::size_t kBitsPerBlock     = 256;
constexpr std::size_t kTargetBitsPerKey = 16;

std::size_t blocks_for(std::size_t num_keys)
{
  auto const bits   = std::max<std::size_t>(num_keys, 1) * kTargetBitsPerKey;
  auto const blocks = (bits + kBitsPerBlock - 1) / kBitsPerBlock;
  return std::max<std::size_t>(blocks, 1);
}

// cuCollections allocator backed by RMM's stream-ordered pool — the same compute-stream memory
// resource the rest of the query uses — instead of cuco's default synchronizing
// cudaMalloc/cudaFree. cuco's allocator interface is stream-aware (allocate(n, stream)), matching
// RMM's stream-ordered resource. This both keeps the Bloom's bit array inside the RMM pool and
// avoids the implicit device sync of a raw cudaMalloc on the hot path.
template <class T>
class rmm_cuco_allocator {
 public:
  using value_type = T;
  explicit rmm_cuco_allocator(rmm::device_async_resource_ref mr) noexcept : _mr{mr} {}
  template <class U>
  rmm_cuco_allocator(rmm_cuco_allocator<U> const& other) noexcept : _mr{other.resource()}
  {
  }

  value_type* allocate(std::size_t n, cuda::stream_ref stream)
  {
    return static_cast<value_type*>(
      _mr.allocate(rmm::cuda_stream_view{stream.get()}, n * sizeof(value_type)));
  }
  void deallocate(value_type* p, std::size_t n, cuda::stream_ref stream) noexcept
  {
    _mr.deallocate(rmm::cuda_stream_view{stream.get()}, p, n * sizeof(value_type));
  }
  [[nodiscard]] rmm::device_async_resource_ref resource() const noexcept { return _mr; }

 private:
  rmm::device_async_resource_ref _mr;
};

using bloom_policy = cuco::default_filter_policy<cuco::xxhash_64<std::int64_t>, std::uint32_t, 8>;
using bloom_alloc  = rmm_cuco_allocator<cuda::std::byte>;
using bloom_type   = cuco::bloom_filter<std::int64_t,
                                        cuco::extent<std::size_t>,
                                        cuda::thread_scope_device,
                                        bloom_policy,
                                        bloom_alloc>;
}  // namespace

struct sirius_dynamic_bloom_filter::impl {
  bloom_type filter;

  impl(std::size_t num_blocks, rmm::device_async_resource_ref mr, cuda::stream_ref stream)
    : filter{cuco::extent<std::size_t>{num_blocks}, {}, {}, bloom_alloc{mr}, stream}
  {
  }
};

bool sirius_dynamic_bloom_filter::supports(cudf::data_type t) noexcept
{
  return t.id() == cudf::type_id::INT64;
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
      "[sirius_dynamic_bloom_filter] unsupported key type (INT64 only for now).");
  }
  auto const n = static_cast<std::size_t>(keys.size());
  cuda::stream_ref const s{stream.value()};
  _impl = std::make_unique<impl>(blocks_for(n), mr, s);

  // Insert every build key. The build join keys are non-null (FK/PK); a null position would only
  // add a stray fingerprint, which costs a negligible false-positive bump, never correctness.
  auto const* d = keys.data<std::int64_t>();
  _impl->filter.add(d, d + n, s);
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
  auto const* d = probe.data<std::int64_t>();
  // true where the probe key's fingerprint is present — a superset of the true matches (false
  // positives allowed). Querying garbage at null positions is harmless; their rows are dropped via
  // the carried-over null mask below.
  _impl->filter.contains(d, d + n, out->mutable_view().data<bool>(), s);

  if (probe.nullable() && probe.null_count() > 0) {
    out->set_null_mask(cudf::copy_bitmask(probe, stream, mr), probe.null_count());
  }
  return out;
}

}  // namespace sirius::op
