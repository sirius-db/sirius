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
#include <rmm/device_buffer.hpp>

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
#include <limits>
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
constexpr std::size_t kBitsPerBlock                 = 256;
constexpr std::size_t kTargetBitsPerKey             = 16;
constexpr std::size_t kBytesPerBlock                = kBitsPerBlock / 8;
constexpr std::size_t kMaximumReductionScratchBytes = 4U * 1024U * 1024U;

std::size_t blocks_for(std::size_t num_keys)
{
  static_assert(kBitsPerBlock % kTargetBitsPerKey == 0);
  constexpr auto keys_per_block = kBitsPerBlock / kTargetBitsPerKey;
  return num_keys == 0 ? 1 : 1 + (num_keys - 1) / keys_per_block;
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

// Each thread owns one destination word; callers stream-order merges on the root stream.
template <class Word>
__global__ void or_bloom_words(Word* destination, Word const* source, std::size_t count)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) { destination[index] |= source[index]; }
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
  return std::make_unique<bloom_replica>(device_id,
                                         make_bloom<sirius_bloom<KeyT>>(num_blocks, mr, stream));
}
}  // namespace

// Owns the source Bloom and every completed device-local replica.
struct sirius_dynamic_bloom_filter::impl {
  int source_device = -1;
  // Retained for null compaction in add().
  rmm::device_async_resource_ref mr;
  std::vector<std::unique_ptr<bloom_replica>> replicas;
  std::unique_ptr<rmm::device_buffer> reduction_scratch;

  explicit impl(rmm::device_async_resource_ref mr) : mr{mr} {}

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
  auto const blocks  = blocks_for(num_keys);
  auto const maximum = std::numeric_limits<std::size_t>::max();
  if (blocks > maximum / kBytesPerBlock) { return maximum; }
  auto const raw_bytes = blocks * kBytesPerBlock;
  if (raw_bytes > maximum - (rmm::CUDA_ALLOCATION_ALIGNMENT - 1)) { return maximum; }
  return detail::tracked_replica_allocation_bytes(raw_bytes);
}

// Sizing on the pre-compaction key count is conservative when nulls are dropped in add().
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
  auto const maximum    = std::numeric_limits<std::size_t>::max();
  if (num_blocks > maximum / kBytesPerBlock) {
    throw std::length_error("[sirius_dynamic_bloom_filter] requested geometry is too large.");
  }
  auto const raw_bytes = num_blocks * kBytesPerBlock;
  if (raw_bytes > maximum - (rmm::CUDA_ALLOCATION_ALIGNMENT - 1)) {
    throw std::length_error("[sirius_dynamic_bloom_filter] requested geometry is too large.");
  }
  _impl = std::make_unique<impl>(mr);
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
  // A null row's payload is garbage; inserting it would poison the filter. Keep compacted
  // storage alive until add_async is queued on stream.
  std::unique_ptr<cudf::table> compacted;
  cudf::column_view build_keys = keys;
  if (keys.null_count() > 0) {
    compacted  = cudf::drop_nulls(cudf::table_view{{keys}}, {0}, stream, _impl->mr);
    build_keys = compacted->view().column(0);
  }
  cuda::stream_ref const cuda_stream{stream.value()};
  std::visit(
    [&](auto& bloom) {
      using filter_type = typename std::decay_t<decltype(bloom)>::element_type;
      using key_type    = typename filter_type::key_type;
      if (build_keys.type().id() != key_type_id<key_type>()) {
        throw std::invalid_argument("[sirius_dynamic_bloom_filter::add] key type mismatch.");
      }
      auto const* data = build_keys.data<key_type>();
      bloom->add_async(data, data + build_keys.size(), cuda_stream);
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
