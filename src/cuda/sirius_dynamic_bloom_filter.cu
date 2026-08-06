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

// cucascade
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policies.cuh>
#include <cuco/hash_functions.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/cstddef>
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
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op {

namespace {
// ~16 bits/key → num_blocks ≈ keys/16
constexpr std::size_t kBitsPerBlock                 = 256;
constexpr std::size_t kTargetBitsPerKey             = 16;
constexpr std::size_t kMaximumReductionScratchBytes = 4U * 1024U * 1024U;

std::size_t blocks_for(std::size_t num_keys)
{
  auto const bits   = std::max<std::size_t>(num_keys, 1) * kTargetBitsPerKey;
  auto const blocks = cuda::ceil_div(bits, kBitsPerBlock);
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
using arrow_bloom = bloom_filter_for<KeyT, arrow_policy<KeyT>>;

template <class KeyT>
using standard_bloom = bloom_filter_for<KeyT, default_policy<KeyT>>;

template <class Filter>
using bloom_owner = std::unique_ptr<Filter>;

// The four legal key-width/policy combinations. A live replica owns exactly one alternative.
using bloom_storage = std::variant<bloom_owner<arrow_bloom<std::int32_t>>,
                                   bloom_owner<standard_bloom<std::int32_t>>,
                                   bloom_owner<arrow_bloom<std::int64_t>>,
                                   bloom_owner<standard_bloom<std::int64_t>>>;

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

template <class Word>
__global__ void or_bloom_words(Word* destination, Word const* source, std::size_t count)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) { destination[index] |= source[index]; }
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
std::unique_ptr<bloom_replica> make_empty_bloom_replica(int device_id,
                                                        std::size_t num_blocks,
                                                        rmm::device_async_resource_ref mr,
                                                        cuda::stream_ref stream)
{
  if (num_blocks <= arrow_policy<KeyT>::max_filter_blocks) {
    return std::make_unique<bloom_replica>(device_id,
                                           make_bloom<arrow_bloom<KeyT>>(num_blocks, mr, stream));
  }
  return std::make_unique<bloom_replica>(device_id,
                                         make_bloom<standard_bloom<KeyT>>(num_blocks, mr, stream));
}

template <class KeyT>
std::unique_ptr<bloom_replica> build_bloom_replica(int device_id,
                                                   cudf::column_view const& keys,
                                                   std::size_t num_blocks,
                                                   rmm::device_async_resource_ref mr,
                                                   cuda::stream_ref stream)
{
  if (num_blocks <= arrow_policy<KeyT>::max_filter_blocks) {
    return std::make_unique<bloom_replica>(
      device_id, build_bloom<arrow_bloom<KeyT>>(keys, num_blocks, mr, stream));
  }
  return std::make_unique<bloom_replica>(
    device_id, build_bloom<standard_bloom<KeyT>>(keys, num_blocks, mr, stream));
}
}  // namespace

// Owns the complete set of ready device-local Bloom replicas.
struct sirius_dynamic_bloom_filter::impl {
  int source_device = -1;
  std::vector<std::unique_ptr<bloom_replica>> replicas;
  std::unique_ptr<rmm::device_buffer> reduction_scratch;

  ~impl()
  {
    if (source_device < 0) { return; }
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{source_device}};
    reduction_scratch.reset();
  }

  [[nodiscard]] bloom_replica* find(int device_id) noexcept
  {
    auto const it =
      std::find_if(replicas.begin(), replicas.end(), [device_id](auto const& replica) {
        return replica->device_id == device_id;
      });
    return it == replicas.end() ? nullptr : it->get();
  }

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
  // Mirrors blocks_for(): each block is kBitsPerBlock bits = kBitsPerBlock/8 bytes.
  return blocks_for(num_keys) * (kBitsPerBlock / 8);
}

sirius_dynamic_bloom_filter::sirius_dynamic_bloom_filter(cudf::column_view const& keys,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
  : sirius_dynamic_bloom_filter(keys.type(), static_cast<std::size_t>(keys.size()), stream, mr)
{
  add(keys, stream);
}

sirius_dynamic_bloom_filter::sirius_dynamic_bloom_filter(cudf::data_type key_type,
                                                         std::size_t expected_num_keys,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (!supports(key_type)) {
    throw std::invalid_argument(
      "[sirius_dynamic_bloom_filter] unsupported key type (INT32 or INT64).");
  }
  cuda::stream_ref const s{stream.value()};
  auto const num_blocks = blocks_for(expected_num_keys);
  _impl                 = std::make_unique<impl>();
  if (cudaGetDevice(&_impl->source_device) != cudaSuccess) {
    throw std::runtime_error("[sirius_dynamic_bloom_filter] failed to identify source device.");
  }

  std::unique_ptr<bloom_replica> source;
  switch (key_type.id()) {
    case cudf::type_id::INT32:
      source = make_empty_bloom_replica<std::int32_t>(_impl->source_device, num_blocks, mr, s);
      break;
    case cudf::type_id::INT64:
      source = make_empty_bloom_replica<std::int64_t>(_impl->source_device, num_blocks, mr, s);
      break;
    default:
      throw std::logic_error(
        "[sirius_dynamic_bloom_filter] supported key type changed during construction.");
  }
  _impl->replicas.push_back(std::move(source));
}

void sirius_dynamic_bloom_filter::add(cudf::column_view const& keys, rmm::cuda_stream_view stream)
{
  if (!_impl || !supports(keys.type())) {
    throw std::invalid_argument("[sirius_dynamic_bloom_filter::add] unsupported key type.");
  }
  int device_id = -1;
  if (cudaGetDevice(&device_id) != cudaSuccess || device_id != _impl->source_device) {
    throw std::logic_error("[sirius_dynamic_bloom_filter::add] source device mismatch.");
  }
  auto* source = _impl->find(_impl->source_device);
  if (source == nullptr) {
    throw std::logic_error("[sirius_dynamic_bloom_filter::add] source replica is missing.");
  }
  cuda::stream_ref const cuda_stream{stream.value()};
  std::visit(
    [&](auto& bloom) {
      using filter_type = typename std::decay_t<decltype(bloom)>::element_type;
      using key_type    = typename filter_type::key_type;
      if (keys.type().id() != key_type_id<key_type>()) {
        throw std::invalid_argument("[sirius_dynamic_bloom_filter::add] key type mismatch.");
      }
      auto const* data = keys.data<key_type>();
      bloom->add_async(data, data + keys.size(), cuda_stream);
    },
    source->bloom);
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

  // Retain all destination objects and streams until direct peer copies have been submitted to
  // every target. The completion pass then waits on transfers already running in parallel.
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

void sirius_dynamic_bloom_filter::replicate_to_devices_strict(
  std::span<dynamic_filter_replica_space const> spaces)
{
  replicate_to_devices(spaces);
  for (auto const& space : spaces) {
    if (!is_available_on_device(space.get_gpu_space().get_device_id())) {
      throw std::runtime_error(
        "[sirius_dynamic_bloom_filter] required device replica is unavailable.");
    }
  }
}

void sirius_dynamic_bloom_filter::merge_from(sirius_dynamic_bloom_filter const& source_filter,
                                             dynamic_filter_replica_space const& source_space,
                                             dynamic_filter_replica_space const& root_space,
                                             rmm::cuda_stream_view root_stream)
{
  if (!_impl || !source_filter._impl) {
    throw std::logic_error("[sirius_dynamic_bloom_filter::merge_from] missing implementation.");
  }
  auto const root_device = root_space.get_gpu_space().get_device_id();
  if (_impl->source_device != root_device ||
      source_filter._impl->source_device != source_space.get_gpu_space().get_device_id()) {
    throw std::logic_error("[sirius_dynamic_bloom_filter::merge_from] plan/device mismatch.");
  }
  auto* destination  = _impl->find(root_device);
  auto const* source = source_filter._impl->find(source_filter._impl->source_device);
  if (destination == nullptr || source == nullptr ||
      destination->bloom.index() != source->bloom.index()) {
    throw std::logic_error("[sirius_dynamic_bloom_filter::merge_from] geometry mismatch.");
  }

  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{root_device}};
  std::visit(
    [&](auto& destination_bloom) {
      using owner_type         = std::decay_t<decltype(destination_bloom)>;
      using filter_type        = typename owner_type::element_type;
      auto const& source_bloom = std::get<bloom_owner<filter_type>>(source->bloom);
      if (!destination_bloom || !source_bloom ||
          destination_bloom->block_extent() != source_bloom->block_extent()) {
        throw std::logic_error("[sirius_dynamic_bloom_filter::merge_from] geometry mismatch.");
      }
      using word_type       = typename filter_type::word_type;
      auto const word_count = destination_bloom->block_extent() * filter_type::words_per_block;
      auto const maximum_words_per_chunk =
        std::max<std::size_t>(kMaximumReductionScratchBytes / sizeof(word_type), 1);
      auto const scratch_words = std::min(word_count, maximum_words_per_chunk);
      auto const scratch_bytes = scratch_words * sizeof(word_type);
      if (!_impl->reduction_scratch || _impl->reduction_scratch->size() < scratch_bytes) {
        _impl->reduction_scratch = std::make_unique<rmm::device_buffer>(
          scratch_bytes, root_stream, root_space.get_gpu_space().get_default_allocator());
      }

      constexpr int threads = 256;
      for (std::size_t word_offset = 0; word_offset < word_count;
           word_offset += maximum_words_per_chunk) {
        auto const chunk_words = std::min(maximum_words_per_chunk, word_count - word_offset);
        auto const chunk_bytes = chunk_words * sizeof(word_type);
        detail::enqueue_replica_copy(_impl->reduction_scratch->data(),
                                     rmm::cuda_device_id{root_device},
                                     source_bloom->data() + word_offset,
                                     source_space.get_gpu_space(),
                                     chunk_bytes,
                                     root_stream,
                                     root_space.get_host_staging_space());
        auto const blocks = static_cast<int>((chunk_words + threads - 1) / threads);
        or_bloom_words<<<blocks, threads, 0, root_stream.value()>>>(
          destination_bloom->data() + word_offset,
          static_cast<word_type const*>(_impl->reduction_scratch->data()),
          chunk_words);
        auto const status = cudaPeekAtLastError();
        if (status != cudaSuccess) {
          throw std::runtime_error(
            std::string("[sirius_dynamic_bloom_filter::merge_from] OR launch failed: ") +
            cudaGetErrorString(status));
        }
      }
    },
    destination->bloom);
}

void sirius_dynamic_bloom_filter::release_reduction_scratch()
{
  if (!_impl || _impl->source_device < 0) { return; }
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{_impl->source_device}};
  _impl->reduction_scratch.reset();
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
