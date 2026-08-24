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
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policies.cuh>
#include <cuco/hash_functions.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/stream_ref>

#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_reservation.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op {

namespace {
constexpr std::size_t kBitsPerBlock     = 256;
constexpr std::size_t kTargetBitsPerKey = 16;

std::size_t blocks_for(std::size_t num_keys)
{
  auto const bits   = std::max<std::size_t>(num_keys, 1) * kTargetBitsPerKey;
  auto const blocks = cuda::ceil_div(bits, kBitsPerBlock);
  return std::max<std::size_t>(blocks, 1);
}

using bloom_alloc = sirius::rmm_cuco_allocator<cuda::std::byte>;

/**
 * @brief cuco-compatible Bloom policy using Lemire fast-range
 *
 * Arrow's policy caps filter size, while cuco's default uses costly 64-bit modulo. Construction
 * and lookup share this mapping, preserving the no-false-negative contract.
 */
template <class KeyT>
class sirius_bloom_policy {
 public:
  using hasher             = cuco::xxhash_64<KeyT>;
  using word_type          = std::uint32_t;
  using hash_argument_type = typename hasher::argument_type;
  using hash_result_type   = decltype(std::declval<hasher>()(std::declval<hash_argument_type>()));

  static constexpr std::uint32_t words_per_block = 8;

 private:
  static constexpr std::uint32_t word_bits       = cuda::std::numeric_limits<word_type>::digits;
  static constexpr std::uint32_t bit_index_width = cuda::std::bit_width(word_bits - 1);
  static constexpr word_type bit_index_mask      = (word_type{1} << bit_index_width) - 1;

  static_assert(words_per_block * bit_index_width <=
                  cuda::std::numeric_limits<hash_result_type>::digits,
                "hash is too narrow to supply one fingerprint bit per word");

 public:
  __device__ constexpr hash_result_type hash(hash_argument_type const& key) const
  {
    return hash_(key);
  }

  template <class Extent>
  [[nodiscard]] __device__ constexpr Extent block_index(hash_result_type hash,
                                                        Extent num_blocks) const
  {
    auto const wide = static_cast<__uint128_t>(static_cast<std::uint64_t>(hash)) *
                      static_cast<__uint128_t>(static_cast<std::uint64_t>(num_blocks));
    return static_cast<Extent>(static_cast<std::uint64_t>(wide >> 64));
  }

  [[nodiscard]] __device__ constexpr word_type word_pattern(hash_result_type hash,
                                                            std::uint32_t word_index) const
  {
    return word_type{1} << ((hash >> (word_index * bit_index_width)) & bit_index_mask);
  }

 private:
  hasher hash_{};
};

template <class KeyT>
using sirius_bloom = cuco::bloom_filter<KeyT,
                                        cuco::extent<std::size_t>,
                                        cuda::thread_scope_device,
                                        sirius_bloom_policy<KeyT>,
                                        bloom_alloc>;

template <class Filter>
using bloom_owner = std::unique_ptr<Filter>;

using bloom_storage =
  std::variant<bloom_owner<sirius_bloom<std::int32_t>>, bloom_owner<sirius_bloom<std::int64_t>>>;

template <class Filter>
bloom_owner<Filter> make_bloom(std::size_t num_blocks,
                               rmm::device_async_resource_ref mr,
                               cuda::stream_ref stream)
{
  return bloom_owner<Filter>(
    new Filter{cuco::extent<std::size_t>{num_blocks}, {}, {}, bloom_alloc{mr}, stream});
}

template <class Filter>
void copy_filter_storage(Filter const& source,
                         cucascade::memory::memory_space const& source_space,
                         Filter& destination,
                         rmm::cuda_device_id destination_device,
                         rmm::cuda_stream_view stream,
                         cucascade::memory::memory_space const& host_staging_space,
                         std::size_t& bytes)
{
  auto const source_blocks = source.block_extent();
  if (destination.block_extent() != source_blocks) {
    throw std::runtime_error("destination Bloom block extent changed during replication");
  }
  bytes = source_blocks * Filter::words_per_block * sizeof(typename Filter::word_type);
  detail::enqueue_replica_copy(destination.data(),
                               destination_device,
                               source.data(),
                               source_space,
                               bytes,
                               stream,
                               host_staging_space);
}

template <class Filter>
bloom_owner<Filter> build_bloom(cudf::column_view const& keys,
                                std::size_t num_blocks,
                                rmm::device_async_resource_ref mr,
                                cuda::stream_ref stream)
{
  using key_type = typename Filter::key_type;
  auto result    = make_bloom<Filter>(num_blocks, mr, stream);
  auto const* d  = keys.data<key_type>();
  auto const n   = keys.size();
  result->add_async(d, d + n, stream);
  return result;
}

template <class KeyT>
constexpr cudf::type_id key_type_id() noexcept
{
  static_assert(std::is_same_v<KeyT, std::int32_t> || std::is_same_v<KeyT, std::int64_t>);
  if constexpr (std::is_same_v<KeyT, std::int32_t>) {
    return cudf::type_id::INT32;
  } else {
    return cudf::type_id::INT64;
  }
}
}  // namespace

struct bloom_replica {
  int device_id = -1;
  bloom_storage bloom;

  template <class Filter>
  bloom_replica(int device_id, bloom_owner<Filter> owner)
    : device_id{device_id}, bloom{std::in_place_type<bloom_owner<Filter>>, std::move(owner)}
  {
  }

  ~bloom_replica() noexcept
  {
    if (device_id < 0) { return; }
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
    std::visit([](auto& owner) { owner.reset(); }, bloom);
  }

  [[nodiscard]] bool has_bloom() const noexcept
  {
    return std::visit([](auto const& owner) { return owner != nullptr; }, bloom);
  }
};

namespace {
template <class KeyT>
std::unique_ptr<bloom_replica> build_bloom_replica(int device_id,
                                                   cudf::column_view const& keys,
                                                   std::size_t num_blocks,
                                                   rmm::device_async_resource_ref mr,
                                                   cuda::stream_ref stream)
{
  return std::make_unique<bloom_replica>(
    device_id, build_bloom<sirius_bloom<KeyT>>(keys, num_blocks, mr, stream));
}
}  // namespace

struct sirius_dynamic_bloom_filter::impl {
  int source_device = -1;
  std::vector<std::unique_ptr<bloom_replica>> replicas;

  [[nodiscard]] bloom_replica const* find(int device_id) const noexcept
  {
    auto const it =
      std::find_if(replicas.begin(), replicas.end(), [device_id](auto const& replica) {
        return replica->device_id == device_id;
      });
    return it == replicas.end() ? nullptr : it->get();
  }
};

bool sirius_dynamic_bloom_filter::supports(cudf::data_type t) noexcept
{
  return t.id() == cudf::type_id::INT32 || t.id() == cudf::type_id::INT64;
}

std::size_t sirius_dynamic_bloom_filter::estimated_bytes(std::size_t num_keys) noexcept
{
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
  // Keep compacted storage alive until add_async is queued on stream.
  std::unique_ptr<cudf::table> compacted;
  cudf::column_view build_keys = keys;
  if (keys.null_count() > 0) {
    compacted  = cudf::drop_nulls(cudf::table_view{{keys}}, {0}, stream, mr);
    build_keys = compacted->view().column(0);
  }
  auto const n = build_keys.size();
  cuda::stream_ref const s{stream.value()};
  auto const num_blocks = blocks_for(n);
  _impl                 = std::make_unique<impl>();
  if (cudaGetDevice(&_impl->source_device) != cudaSuccess) {
    throw std::runtime_error("[sirius_dynamic_bloom_filter] failed to identify source device.");
  }

  std::unique_ptr<bloom_replica> source;
  switch (keys.type().id()) {
    case cudf::type_id::INT32:
      source =
        build_bloom_replica<std::int32_t>(_impl->source_device, build_keys, num_blocks, mr, s);
      break;
    case cudf::type_id::INT64:
      source =
        build_bloom_replica<std::int64_t>(_impl->source_device, build_keys, num_blocks, mr, s);
      break;
    default:
      throw std::logic_error(
        "[sirius_dynamic_bloom_filter] supported key type changed during construction.");
  }
  _impl->replicas.push_back(std::move(source));
}

sirius_dynamic_bloom_filter::~sirius_dynamic_bloom_filter() = default;

void sirius_dynamic_bloom_filter::replicate_to_devices(
  std::span<dynamic_filter_replica_space const> spaces)
{
  if (!_impl || _impl->replicas.empty()) { return; }
  auto const* source = _impl->find(_impl->source_device);
  if (!source) { return; }
  auto const source_target = std::find_if(spaces.begin(), spaces.end(), [this](auto const& target) {
    return target.get_gpu_space().get_device_id() == _impl->source_device;
  });
  if (source_target == spaces.end()) {
    SIRIUS_LOG_WARN(
      "[sirius_dynamic_bloom_filter] source GPU {} has no replica memory space; remote GPUs "
      "will skip this optional filter.",
      _impl->source_device);
    return;
  }
  auto const& source_space = source_target->get_gpu_space();

  // Keep copies and streams alive until all peer transfers are queued and synchronized.
  std::vector<std::pair<std::unique_ptr<bloom_replica>, rmm::cuda_stream_view>> pending;
  pending.reserve(spaces.size());
  _impl->replicas.reserve(_impl->replicas.size() + spaces.size());
  for (auto const& target : spaces) {
    auto const& target_space = target.get_gpu_space();
    auto const device_id     = target_space.get_device_id();
    if (device_id == _impl->source_device || _impl->find(device_id)) { continue; }
    std::size_t bytes = 0;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      auto const stream = target_space.acquire_stream();

      auto replica = std::visit(
        [&](auto const& source_bloom) {
          if (!source_bloom) {
            throw std::logic_error(
              "[sirius_dynamic_bloom_filter] source replica has no Bloom filter.");
          }
          using owner_type  = std::decay_t<decltype(source_bloom)>;
          using filter_type = typename owner_type::element_type;

          bytes = source_bloom->block_extent() * filter_type::words_per_block *
                  sizeof(typename filter_type::word_type);
          auto reservation = detail::scoped_replica_reservation::try_acquire(
            target, detail::tracked_replica_allocation_bytes(bytes), stream);
          if (!reservation) { return std::unique_ptr<bloom_replica>{}; }

          auto destination_bloom = make_bloom<filter_type>(source_bloom->block_extent(),
                                                           reservation->allocator(),
                                                           cuda::stream_ref{stream.value()});
          auto result = std::make_unique<bloom_replica>(device_id, std::move(destination_bloom));
          auto& destination = *std::get<bloom_owner<filter_type>>(result->bloom);
          copy_filter_storage(*source_bloom,
                              source_space,
                              destination,
                              rmm::cuda_device_id{device_id},
                              stream,
                              target.get_host_staging_space(),
                              bytes);
          return result;
        },
        source->bloom);
      if (!replica) {
        SIRIUS_LOG_WARN(
          "[sirius_dynamic_bloom_filter] replica GPU {} -> GPU {} skipped: destination "
          "reservation for {} bytes unavailable.",
          _impl->source_device,
          device_id,
          bytes);
        continue;
      }
      pending.emplace_back(std::move(replica), stream);
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_dynamic_bloom_filter] replica GPU {} -> GPU {} unavailable: {}. "
        "That GPU will skip this optional filter.",
        _impl->source_device,
        device_id,
        e.what());
      continue;
    }
    SIRIUS_LOG_DEBUG("[sirius_dynamic_bloom_filter] queued {}-byte replica GPU {} -> GPU {}.",
                     bytes,
                     _impl->source_device,
                     device_id);
  }

  for (auto& [replica, stream] : pending) {
    auto const device_id = replica->device_id;
    try {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      stream.synchronize();
      _impl->replicas.push_back(std::move(replica));
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_dynamic_bloom_filter] replica GPU {} -> GPU {} unavailable: {}. "
        "That GPU will skip this optional filter.",
        _impl->source_device,
        device_id,
        e.what());
    }
  }
}

bool sirius_dynamic_bloom_filter::is_available_on_device(int device_id) const noexcept
{
  return _impl && _impl->find(detail::resolve_dynamic_filter_device_id(device_id)) != nullptr;
}

std::size_t sirius_dynamic_bloom_filter::replica_count() const noexcept
{
  return _impl ? _impl->replicas.size() : 0;
}

std::unique_ptr<cudf::column> sirius_dynamic_bloom_filter::compute_mask(
  cudf::column_view const& probe,
  int device_id,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  if (!supports(probe.type())) { return nullptr; }
  auto const* replica =
    _impl ? _impl->find(detail::resolve_dynamic_filter_device_id(device_id)) : nullptr;
  if (!replica || !replica->has_bloom()) { return nullptr; }

  auto const matching_key_type = std::visit(
    [&](auto const& bloom) {
      using owner_type = std::decay_t<decltype(bloom)>;
      using key_type   = typename owner_type::element_type::key_type;
      return probe.type().id() == key_type_id<key_type>();
    },
    replica->bloom);
  if (!matching_key_type) { return nullptr; }

  auto const n = probe.size();
  auto out     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  cuda::stream_ref const s{stream.value()};
  auto* const outp = out->mutable_view().data<bool>();

  std::visit(
    [&](auto const& bloom) {
      using owner_type = std::decay_t<decltype(bloom)>;
      using key_type   = typename owner_type::element_type::key_type;
      auto const* d    = probe.data<key_type>();
      bloom->contains_async(d, d + n, outp, s);
    },
    replica->bloom);

  if (probe.nullable() && probe.null_count() > 0) {
    out->set_null_mask(cudf::copy_bitmask(probe, stream, mr), probe.null_count());
  }
  return out;
}

}  // namespace sirius::op
