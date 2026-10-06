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

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policy.cuh>
#include <cuco/bloom_filter_ref.cuh>
#include <cuda/dynamic_filter_probe.cuh>
#include <cuda/sirius_rmm_cuco_allocator.cuh>
#include <cuda/std/cstddef>
#include <cuda/stream>
#include <cuda/utility>
#include <thrust/iterator/counting_iterator.h>

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <log/logging.hpp>
#include <op/dynamic_filter/bloom_sizing.hpp>
#include <op/dynamic_filter/detail/accumulated_bloom_builder.hpp>
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_reservation.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_transfer.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <telemetry/nvtx.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op {

namespace {
using bloom_alloc = sirius::rmm_cuco_allocator<cuda::std::byte>;

template <class KeyT>
using sirius_bloom = cuco::bloom_filter<KeyT,
                                        cuco::extent<std::size_t>,
                                        cuda::thread_scope_device,
                                        cuco::bloom_filter_policy<KeyT>,
                                        bloom_alloc>;

template <class Filter>
using bloom_owner = std::unique_ptr<Filter>;

// One alternative per membership_key_rep.
using bloom_storage = std::variant<bloom_owner<sirius_bloom<std::int32_t>>,
                                   bloom_owner<sirius_bloom<std::int64_t>>,
                                   bloom_owner<sirius_bloom<std::uint32_t>>,
                                   bloom_owner<sirius_bloom<std::uint64_t>>>;

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
                         ::cuda::stream_ref stream,
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
bloom_owner<Filter> build_bloom(membership_key_domain const& domain,
                                cudf::column_view const& keys,
                                std::size_t num_blocks,
                                rmm::device_async_resource_ref mr,
                                cuda::stream_ref stream)
{
  using key_type = typename Filter::key_type;
  auto result    = make_bloom<Filter>(num_blocks, mr, stream);
  if (keys.size() > 0) {
    // The build column may sit at a same-family carrier other than the rep; the iterator converts
    // per element instead of materializing a rep-typed copy.
    bool const added = detail::with_build_key_iterator<key_type>(
      domain, keys, stream, mr, [&](auto first, auto last) {
        result->add_async(first, last, stream);
      });
    if (!added) {
      throw std::logic_error("[sirius_dynamic_bloom_filter] build carrier does not fit its rep.");
    }
  }
  return result;
}

/// @brief The Bloom half of detail::membership_probe_functor: a converted key passes when the
/// filter reports it. An inserted key always fits the key domain, so the adapter's rejection of a
/// non-representable value preserves the no-false-negative contract.
template <class FilterRef>
struct bloom_lookup {
  using key_type = typename FilterRef::key_type;
  FilterRef ref;
  __device__ __forceinline__ bool operator()(key_type key) const noexcept
  {
    return ref.contains(key);
  }
};

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
                                                   membership_key_domain const& domain,
                                                   cudf::column_view const& keys,
                                                   std::size_t num_blocks,
                                                   rmm::device_async_resource_ref mr,
                                                   cuda::stream_ref stream)
{
  return std::make_unique<bloom_replica>(
    device_id, build_bloom<sirius_bloom<KeyT>>(domain, keys, num_blocks, mr, stream));
}
}  // namespace

namespace {

template <class Key>
using raw_bloom_ref = cuco::bloom_filter_ref<Key,
                                             cuco::extent<std::size_t>,
                                             cuda::thread_scope_device,
                                             cuco::bloom_filter_policy<Key>>;

/**
 * @brief Throws `detail::accumulation_cuda_error` for a failed CUDA call after clearing the error.
 *
 * @param status The result of the CUDA call
 * @param call The name of the CUDA call, used in the error message
 * @param kernel_launch Whether @p status comes from a kernel launch
 * @throws detail::accumulation_cuda_error if @p status is not `cudaSuccess`
 */
void check_cuda(cudaError_t status, char const* call, bool kernel_launch = false)
{
  if (status == cudaSuccess) { return; }
  (void)cudaGetLastError();
  throw detail::accumulation_cuda_error{status, kernel_launch, call};
}

/**
 * @brief Timing-free CUDA event, created on the device that is current at construction.
 */
class cuda_event final {
 public:
  cuda_event()
  {
    check_cuda(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming), "cudaEventCreate");
  }
  ~cuda_event()
  {
    if (_event != nullptr && cudaEventDestroy(_event) != cudaSuccess) { (void)cudaGetLastError(); }
  }
  cuda_event(cuda_event const&)            = delete;
  cuda_event& operator=(cuda_event const&) = delete;
  cuda_event(cuda_event&&)                 = delete;
  cuda_event& operator=(cuda_event&&)      = delete;

  /**
   * @brief Creates an event on @p device, restoring the current device afterwards.
   *
   * @param device The GPU the event belongs to
   */
  [[nodiscard]] static cuda_event on(int device)
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    return cuda_event{};  // A prvalue: initializes the caller's object without a move.
  }

  [[nodiscard]] cudaEvent_t get() const noexcept { return _event; }

 private:
  cudaEvent_t _event{};
};

/**
 * @brief One published replica of an accumulated key: the array and the partial stream that frees
 * it.
 *
 * The replica and the builder's partial share the stream, and whichever releases it last destroys
 * it, so both destructors make their own GPU current first.
 */
struct accumulated_bloom_replica {
  int device_id;
  std::size_t blocks;
  std::shared_ptr<rmm::cuda_stream>
    stream;  // Declared before bits: outlives the stream-ordered free.
  std::unique_ptr<rmm::device_buffer> bits;

  accumulated_bloom_replica(int device,
                            std::size_t num_blocks,
                            std::shared_ptr<rmm::cuda_stream> owner_stream,
                            std::unique_ptr<rmm::device_buffer> storage) noexcept
    : device_id{device},
      blocks{num_blocks},
      stream{std::move(owner_stream)},
      bits{std::move(storage)}
  {
  }

  ~accumulated_bloom_replica() noexcept
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
    bits.reset();
    stream.reset();
  }

  accumulated_bloom_replica(accumulated_bloom_replica const&)            = delete;
  accumulated_bloom_replica& operator=(accumulated_bloom_replica const&) = delete;
  accumulated_bloom_replica(accumulated_bloom_replica&&)                 = delete;
  accumulated_bloom_replica& operator=(accumulated_bloom_replica&&)      = delete;
};

template <class Key>
[[nodiscard]] raw_bloom_ref<Key> accumulated_ref(void* bits, std::size_t blocks)
{
  using block = typename raw_bloom_ref<Key>::filter_block_type;
  static_assert(sizeof(block) == bloom_bytes_per_block);
  return raw_bloom_ref<Key>{static_cast<block*>(bits), cuco::extent<std::size_t>{blocks}, {}, {}};
}

template <class Key>
void add_accumulated_keys(rmm::device_buffer& bits,
                          std::size_t blocks,
                          cudf::column_view const& keys,
                          cuda::stream_ref stream)
{
  if (keys.size() == 0) { return; }
  auto const* begin = keys.data<Key>();
  // A pending error belongs to earlier work; it must not be read below as this launch's.
  check_cuda(cudaGetLastError(), "pending CUDA error before Bloom add_if_async");
  // Null build keys match nothing under the join's null_equality::UNEQUAL, so they are not added.
  accumulated_ref<Key>(bits.data(), blocks)
    .add_if_async(begin,
                  begin + keys.size(),
                  thrust::counting_iterator<cudf::size_type>{0},
                  detail::probe_validity_of(keys),
                  stream);
  check_cuda(cudaGetLastError(), "Bloom add_if_async", true);
}

// ORs the scratch slot of every source into the root chunk. Slot i of the buffer holds source i's
// copy of the chunk.
__global__ void or_bloom_chunk(uint4* destination,
                               uint4 const* scratch,
                               std::size_t slot_stride,
                               int source_count,
                               std::size_t count)
{
  auto const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    uint4 value = destination[index];
    for (int source = 0; source < source_count; ++source) {
      uint4 const slot = scratch[static_cast<std::size_t>(source) * slot_stride + index];
      value.x |= slot.x;
      value.y |= slot.y;
      value.z |= slot.z;
      value.w |= slot.w;
    }
    destination[index] = value;
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// sirius_dynamic_bloom_filter::impl
//===----------------------------------------------------------------------===//
struct sirius_dynamic_bloom_filter::impl {
  int source_device = -1;
  std::vector<std::unique_ptr<bloom_replica>> replicas;
  std::vector<std::unique_ptr<accumulated_bloom_replica>> accumulated_replicas;

  [[nodiscard]] accumulated_bloom_replica const* find_accumulated(int device_id) const noexcept
  {
    auto const it = std::find_if(
      accumulated_replicas.begin(), accumulated_replicas.end(), [device_id](auto const& replica) {
        return replica->device_id == device_id;
      });
    return it == accumulated_replicas.end() ? nullptr : it->get();
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

//===----------------------------------------------------------------------===//
// sirius_dynamic_bloom_filter
//===----------------------------------------------------------------------===//
bool sirius_dynamic_bloom_filter::supports(cudf::data_type t) noexcept
{
  return membership_key_supported(t);
}

std::size_t sirius_dynamic_bloom_filter::estimated_bytes(std::size_t num_keys) noexcept
{
  return bloom_blocks_for(num_keys) * bloom_bytes_per_block;
}

sirius_dynamic_bloom_filter::sirius_dynamic_bloom_filter(cudf::column_view const& keys,
                                                         ::cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  // Classifies, checks the DECIMAL128 fit, compacts null build keys out, and names the source
  // device; `build.compacted` stays alive until add_async is queued on `stream`.
  auto const build = prepare_membership_build("[sirius_dynamic_bloom_filter]", keys, stream, mr);
  _domain          = build.domain;
  auto const n     = build.keys.size();
  cuda::stream_ref const s{stream.get()};
  auto const num_blocks = bloom_blocks_for(n);
  _impl                 = std::make_unique<impl>();
  _impl->source_device  = build.source_device;

  auto source = detail::dispatch_key_rep(_domain.rep, [&](auto key_tag) {
    using key_type = decltype(key_tag);
    return build_bloom_replica<key_type>(
      _impl->source_device, _domain, build.keys, num_blocks, mr, s);
  });
  _impl->replicas.push_back(std::move(source));
}

sirius_dynamic_bloom_filter::~sirius_dynamic_bloom_filter() = default;

sirius_dynamic_bloom_filter::sirius_dynamic_bloom_filter(membership_key_domain domain,
                                                         std::unique_ptr<impl> completed)
  : _domain{domain}, _impl{std::move(completed)}
{
}

std::shared_ptr<sirius_dynamic_bloom_filter> sirius_dynamic_bloom_filter::make_accumulated(
  membership_key_domain domain, std::unique_ptr<impl> completed)
{
  // The constructor is private, so std::make_shared cannot reach it; ownership is taken at once.
  return std::shared_ptr<sirius_dynamic_bloom_filter>{
    new sirius_dynamic_bloom_filter{domain, std::move(completed)}};
}

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
  std::vector<std::pair<std::unique_ptr<bloom_replica>, ::cuda::stream_ref>> pending;
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

          auto destination_bloom = make_bloom<filter_type>(
            source_bloom->block_extent(), reservation->allocator(), cuda::stream_ref{stream.get()});
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
      stream.sync();
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
  auto const resolved = detail::resolve_dynamic_filter_device_id(device_id);
  return _impl &&
         (_impl->find(resolved) != nullptr || _impl->find_accumulated(resolved) != nullptr);
}

std::size_t sirius_dynamic_bloom_filter::replica_count() const noexcept
{
  return _impl ? _impl->replicas.size() + _impl->accumulated_replicas.size() : 0;
}

std::unique_ptr<cudf::column> sirius_dynamic_bloom_filter::compute_mask(
  cudf::column_view const& probe,
  std::uint32_t const* prior_mask_words,
  int device_id,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  auto const resolved = detail::resolve_dynamic_filter_device_id(device_id);
  if (auto const* accumulated = _impl ? _impl->find_accumulated(resolved) : nullptr) {
    return detail::dispatch_key_rep(_domain.rep, [&]<class Key>(Key) {
      auto const ref = accumulated_ref<Key>(accumulated->bits->data(), accumulated->blocks);
      return detail::run_membership_probe<Key>(
        _domain, probe, prior_mask_words, stream, mr, bloom_lookup<decltype(ref)>{ref});
    });
  }
  auto const* replica = _impl ? _impl->find(resolved) : nullptr;
  if (!replica || !replica->has_bloom()) { return nullptr; }

  return std::visit(
    [&](auto const& bloom) {
      using owner_type = std::decay_t<decltype(bloom)>;
      using key_type   = typename owner_type::element_type::key_type;
      auto ref         = bloom->ref();
      return detail::run_membership_probe<key_type>(
        _domain, probe, prior_mask_words, stream, mr, bloom_lookup<decltype(ref)>{ref});
    },
    replica->bloom);
}

namespace detail {

//===----------------------------------------------------------------------===//
// accumulated_bloom_builder::impl
//===----------------------------------------------------------------------===//
struct accumulated_bloom_builder::impl {
  /**
   * @brief One GPU's private arrays and the exclusive stream that orders their use and release.
   */
  struct partial {
    cucascade::memory::memory_space* space;
    int device_id;
    std::shared_ptr<rmm::cuda_stream> stream;
    std::vector<std::unique_ptr<rmm::device_buffer>> arrays;
    std::optional<cuda_event> init;  // Recorded after the zero fill; every insert waits on it.
    std::atomic<bool> contributed{false};
    std::atomic<bool> const* leak_storage;

    partial(cucascade::memory::memory_space& owner, std::atomic<bool> const& leak)
      : space{&owner},
        device_id{owner.get_device_id()},
        stream{std::make_shared<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking)},
        leak_storage{&leak}
    {
    }

    ~partial()
    {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
      if (leak_storage->load()) {
        // A failed host join left work that may still touch these arrays: never free them.
        for (auto& array : arrays) {
          (void)array.release();
        }
      }
      init.reset();
      arrays.clear();
      stream.reset();
    }

    partial(partial const&)            = delete;
    partial& operator=(partial const&) = delete;
    partial(partial&&)                 = delete;
    partial& operator=(partial&&)      = delete;

    [[nodiscard]] ::cuda::stream_ref stream_ref() const noexcept
    {
      return ::cuda::stream_ref{stream->value()};
    }
  };

  impl(std::vector<key> active_keys, accumulated_bloom_geometry shape)
    : keys{std::move(active_keys)}, geometry{shape}
  {
  }

  [[nodiscard]] partial* find(int device_id) const noexcept
  {
    auto const found =
      std::ranges::find(partials, device_id, [](auto const& entry) { return entry->device_id; });
    return found == partials.end() ? nullptr : found->get();
  }

  std::vector<key> keys;
  accumulated_bloom_geometry geometry;
  std::atomic<bool> leak_storage{false};
  std::vector<std::unique_ptr<partial>> partials;  // Destroyed before leak_storage.
  std::vector<std::shared_ptr<sirius_dynamic_bloom_filter>> filters;

  class chunk_pipeline;
};

/**
 * @brief Enqueues the reduction and replication of one publication, then waits for it once.
 *
 * Sources are the non-root partials with at least one contribution; targets are every non-root
 * partial. Each chunk is copied from every source into one of two scratch buffers on the
 * publication stream (ingress), ORed into the root's array on the root partial's stream (merge),
 * and copied from the root into every target on the target's stream (egress). Without sources the
 * root's array is already the union and only egress runs.
 */
class accumulated_bloom_builder::impl::chunk_pipeline final {
 public:
  /**
   * @brief Creates the pipeline's events on the root GPU; enqueues nothing.
   *
   * @param owner The builder state whose arrays are published
   * @param root The reducing partial; its GPU is the current device
   * @param sources The non-root partials that received contributions; must outlive the pipeline
   * @param targets Every non-root partial; must outlive the pipeline
   * @param stream The publication stream of the root GPU
   * @param scratch One buffer per chunk parity in use (at most two), each of `sources.size()` chunk
   * slots; null when there are no sources
   */
  chunk_pipeline(impl const& owner,
                 partial& root,
                 std::span<partial* const> sources,
                 std::span<partial* const> targets,
                 ::cuda::stream_ref stream,
                 std::byte* scratch)
    : _geometry{owner.geometry},
      _root{root},
      _sources{sources},
      _targets{targets},
      _stream{stream},
      _scratch{scratch},
      _slot_bytes{_sources.size() * owner.geometry.chunk_bytes},
      _ingress_done{cuda_event::on(root.device_id), cuda_event::on(root.device_id)},
      _merged{cuda_event::on(root.device_id), cuda_event::on(root.device_id)}
  {
  }

  /**
   * @brief Orders the first reads of every partial after all of its inserts.
   *
   * Every contribution is folded into its partial's stream. The publication stream waits for each
   * source's stream before ingress; without sources, every target waits for the root's stream
   * before egress. With sources, the merge on the root's stream already follows the root's inserts.
   */
  void order_after_inserts()
  {
    for (auto* source : _sources) {
      auto const ready = cuda_event::on(source->device_id);
      record(ready, *source);
      check_cuda(cudaStreamWaitEvent(_stream.get(), ready.get(), 0),
                 "cudaStreamWaitEvent(source ready)");
    }
    if (!_sources.empty()) { return; }
    auto const root_ready = cuda_event::on(_root.device_id);
    record(root_ready, _root);
    for (auto* target : _targets) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{target->device_id}};
      check_cuda(cudaStreamWaitEvent(target->stream->value(), root_ready.get(), 0),
                 "cudaStreamWaitEvent(root ready)");
    }
  }

  /**
   * @brief Enqueues ingress, merge and egress of one chunk.
   *
   * @param key_index The key whose array the chunk belongs to
   * @param offset The chunk's byte offset in that array, a multiple of the chunk size
   */
  void enqueue_chunk(std::size_t key_index, std::size_t offset)
  {
    auto const bytes       = std::min(_geometry.chunk_bytes, _geometry.raw_bytes - offset);
    auto const buffer      = _sequence % 2;
    auto* const root_chunk = static_cast<std::byte*>(_root.arrays[key_index]->data()) + offset;
    if (!_sources.empty()) {
      enqueue_ingress(key_index, offset, bytes, buffer);
      enqueue_merge(root_chunk, bytes, buffer);
    }
    enqueue_egress(key_index, offset, bytes, buffer, root_chunk);
    ++_sequence;
  }

  /**
   * @brief Makes the publication stream wait for every replica and waits for it on the host.
   *
   * This is the publication's only host wait; filters become visible only after it.
   */
  void wait_until_replicas_ready()
  {
    for (auto* target : _targets) {
      auto const ready = cuda_event::on(target->device_id);
      record(ready, *target);
      check_cuda(cudaStreamWaitEvent(_stream.get(), ready.get(), 0),
                 "cudaStreamWaitEvent(replica ready)");
    }
    auto const root_done = cuda_event::on(_root.device_id);
    record(root_done, _root);
    check_cuda(cudaStreamWaitEvent(_stream.get(), root_done.get(), 0),
               "cudaStreamWaitEvent(root done)");
    nvtx_scoped_range wait_range{"dynfilter::accum::late_wait"};
    check_cuda(cudaStreamSynchronize(_stream.get()), "cudaStreamSynchronize(publication)");
  }

  /**
   * @brief Number of chunks enqueued so far.
   */
  [[nodiscard]] std::size_t chunks() const noexcept { return _sequence; }

 private:
  /**
   * @brief Records @p event, created on @p owner's GPU, on @p owner's stream.
   */
  static void record(cuda_event const& event, partial const& owner)
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{owner.device_id}};
    check_cuda(cudaEventRecord(event.get(), owner.stream->value()), "cudaEventRecord(partial)");
  }

  void enqueue_ingress(std::size_t key_index,
                       std::size_t offset,
                       std::size_t bytes,
                       std::size_t buffer)
  {
    if (_sequence >= 2) {
      // The merge of chunk (sequence - 2) read this buffer.
      check_cuda(cudaStreamWaitEvent(_stream.get(), _merged[buffer].get(), 0),
                 "cudaStreamWaitEvent(scratch reuse)");
    }
    auto* const slots = _scratch + buffer * _slot_bytes;
    for (std::size_t index = 0; index < _sources.size(); ++index) {
      auto const* source = _sources[index];
      check_cuda(cudaMemcpyPeerAsync(
                   slots + index * _geometry.chunk_bytes,
                   _root.device_id,
                   static_cast<std::byte const*>(source->arrays[key_index]->data()) + offset,
                   source->device_id,
                   bytes,
                   _stream.get()),
                 "cudaMemcpyPeerAsync(ingress)");
    }
    check_cuda(cudaEventRecord(_ingress_done[buffer].get(), _stream.get()),
               "cudaEventRecord(ingress)");
  }

  void enqueue_merge(std::byte* root_chunk, std::size_t bytes, std::size_t buffer)
  {
    constexpr unsigned threads = 256;
    auto const root_stream     = _root.stream->value();
    check_cuda(cudaStreamWaitEvent(root_stream, _ingress_done[buffer].get(), 0),
               "cudaStreamWaitEvent(ingress)");
    auto const words = bytes / sizeof(uint4);
    auto const blocks =
      static_cast<unsigned>(std::min<std::size_t>(cuda::ceil_div(words, threads), 4096));
    check_cuda(cudaGetLastError(), "pending CUDA error before or_bloom_chunk");
    or_bloom_chunk<<<blocks, threads, 0, root_stream>>>(
      reinterpret_cast<uint4*>(root_chunk),
      reinterpret_cast<uint4 const*>(_scratch + buffer * _slot_bytes),
      _geometry.chunk_bytes / sizeof(uint4),
      static_cast<int>(_sources.size()),
      words);
    check_cuda(cudaGetLastError(), "or_bloom_chunk", true);
    check_cuda(cudaEventRecord(_merged[buffer].get(), root_stream), "cudaEventRecord(merged)");
  }

  void enqueue_egress(std::size_t key_index,
                      std::size_t offset,
                      std::size_t bytes,
                      std::size_t buffer,
                      std::byte const* root_chunk)
  {
    for (auto* target : _targets) {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{target->device_id}};
      if (!_sources.empty()) {
        check_cuda(cudaStreamWaitEvent(target->stream->value(), _merged[buffer].get(), 0),
                   "cudaStreamWaitEvent(egress)");
      }
      check_cuda(
        cudaMemcpyPeerAsync(static_cast<std::byte*>(target->arrays[key_index]->data()) + offset,
                            target->device_id,
                            root_chunk,
                            _root.device_id,
                            bytes,
                            target->stream->value()),
        "cudaMemcpyPeerAsync(egress)");
    }
  }

  accumulated_bloom_geometry const& _geometry;
  partial& _root;
  std::span<partial* const> _sources;
  std::span<partial* const> _targets;
  ::cuda::stream_ref _stream;
  std::byte* _scratch;
  std::size_t _slot_bytes;
  std::array<cuda_event, 2> _ingress_done;  // Per scratch buffer: its ingress landed
  std::array<cuda_event, 2> _merged;        // Per scratch buffer: its merge finished
  std::size_t _sequence = 0;                // Global chunk index over (key, offset)
};

namespace {

template <class Fn>
void dispatch_key_type(cudf::data_type type, Fn&& fn)
{
  switch (type.id()) {
    case cudf::type_id::INT32: std::forward<Fn>(fn).template operator()<std::int32_t>(); break;
    case cudf::type_id::INT64: std::forward<Fn>(fn).template operator()<std::int64_t>(); break;
    default:
      throw accumulation_invariant_error(
        "[accumulated_bloom_builder] unsupported accumulated key type");
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// accumulated_bloom_builder
//===----------------------------------------------------------------------===//
accumulated_bloom_builder::accumulated_bloom_builder(std::unique_ptr<impl> state) noexcept
  : _impl{std::move(state)}
{
}

accumulated_bloom_builder::~accumulated_bloom_builder() = default;
accumulated_bloom_builder::accumulated_bloom_builder(accumulated_bloom_builder&&) noexcept =
  default;
accumulated_bloom_builder& accumulated_bloom_builder::operator=(
  accumulated_bloom_builder&&) noexcept = default;

std::optional<accumulated_bloom_builder> accumulated_bloom_builder::try_create(
  std::vector<key> keys,
  accumulated_bloom_geometry geometry,
  std::span<dynamic_filter_replica_space const> targets)
{
  for (auto const& active : keys) {
    if (!supports(active.type)) {
      throw accumulation_invariant_error(
        "[accumulated_bloom_builder] unsupported accumulated key type");
    }
  }
  auto state = std::make_unique<impl>(std::move(keys), geometry);
  auto const lease_bytes =
    state->keys.size() * tracked_replica_allocation_bytes(geometry.raw_bytes);
  for (auto const& target : targets) {
    auto& space = target.get_gpu_space();
    if (state->find(space.get_device_id()) != nullptr) { continue; }
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{space.get_device_id()}};
    auto owned = std::make_unique<impl::partial>(space, state->leak_storage);
    {
      // Detach the lease before `owned` can be destroyed, so its frees cannot credit that tracker.
      auto lease = scoped_replica_reservation::try_acquire(space, lease_bytes, owned->stream_ref());
      if (!lease) { return std::nullopt; }
      owned->arrays.reserve(state->keys.size());
      for (std::size_t index = 0; index < state->keys.size(); ++index) {
        owned->arrays.push_back(std::make_unique<rmm::device_buffer>(
          geometry.raw_bytes, rmm::cuda_stream_view{owned->stream->value()}, lease->allocator()));
        check_cuda(cudaMemsetAsync(
                     owned->arrays.back()->data(), 0, geometry.raw_bytes, owned->stream->value()),
                   "cudaMemsetAsync(accumulated Bloom)");
      }
    }
    owned->init.emplace();
    check_cuda(cudaEventRecord(owned->init->get(), owned->stream->value()),
               "cudaEventRecord(accumulated Bloom init)");
    state->partials.push_back(std::move(owned));
  }
  return accumulated_bloom_builder{std::move(state)};
}

namespace {

/**
 * @brief Throws `accumulation_invariant_error` with @p mismatch unless @p stream belongs to
 * `device`.
 */
void require_stream_on(::cuda::stream_ref stream, rmm::cuda_device_id device, char const* mismatch)
{
  int stream_device = -1;
  check_cuda(cudaStreamGetDevice(stream.get(), &stream_device), "cudaStreamGetDevice");
  if (stream_device != device.value()) { throw accumulation_invariant_error(mismatch); }
}

}  // namespace

bool accumulated_bloom_builder::enqueue_add(cudf::table_view const& input,
                                            rmm::cuda_device_id device,
                                            ::cuda::stream_ref stream)
{
  nvtx_scoped_range range{"dynfilter::accum::contribute"};
  check_cuda(cudaGetLastError(), "pending CUDA error before accumulated contribution");
  auto* target = _impl->find(device.value());
  if (target == nullptr) { return false; }
  require_stream_on(
    stream,
    device,
    "[accumulated_bloom_builder::enqueue_add] the contribution stream belongs to another GPU");
  rmm::cuda_set_device_raii guard{device};
  cuda_event done;
  std::array const joined{stream};
  try {
    join_on_failure(joined, [&] {
      check_cuda(cudaStreamWaitEvent(stream.get(), target->init->get(), 0),
                 "cudaStreamWaitEvent(init)");
      for (std::size_t index = 0; index < _impl->keys.size(); ++index) {
        auto const& active = _impl->keys[index];
        dispatch_key_type(active.type, [&]<class Key>() {
          add_accumulated_keys<Key>(
            *target->arrays[index], _impl->geometry.blocks, input.column(active.ordinal), stream);
        });
      }
      check_cuda(cudaEventRecord(done.get(), stream.get()), "cudaEventRecord(contribution)");
      // Fold the contribution into the partial stream: every later egress or free follows it.
      check_cuda(cudaStreamWaitEvent(target->stream->value(), done.get(), 0),
                 "cudaStreamWaitEvent(fold contribution)");
    });
  } catch (unjoined_gpu_work const&) {
    _impl->leak_storage.store(true);
    throw;
  }
  target->contributed.store(true, std::memory_order_relaxed);
  return true;
}

std::optional<std::span<std::shared_ptr<sirius_dynamic_bloom_filter> const>>
accumulated_bloom_builder::finish(rmm::cuda_device_id root_device,
                                  ::cuda::stream_ref stream,
                                  rmm::device_async_resource_ref scratch_mr)
{
  nvtx_scoped_range range{"dynfilter::accum::finish"};
  check_cuda(cudaGetLastError(), "pending CUDA error before accumulated publication");
  int current_device = -1;
  check_cuda(cudaGetDevice(&current_device), "cudaGetDevice(publication)");
  if (current_device != root_device.value()) {
    throw accumulation_invariant_error(
      "[accumulated_bloom_builder::finish] the root GPU is not the current device");
  }
  require_stream_on(
    stream,
    root_device,
    "[accumulated_bloom_builder::finish] the publication stream belongs to another GPU");
  auto* root = _impl->find(root_device.value());
  if (root == nullptr) { return std::nullopt; }
  std::vector<impl::partial*> sources;
  std::vector<impl::partial*> targets;
  for (auto const& entry : _impl->partials) {
    if (entry.get() == root) { continue; }
    targets.push_back(entry.get());
    if (entry->contributed.load(std::memory_order_relaxed)) { sources.push_back(entry.get()); }
  }
  auto const& geometry = _impl->geometry;
  rmm::cuda_set_device_raii root_guard{root_device};

  // Scratch is bounded by min(2, total_chunks) * contributing_nonroot_gpus * chunk_bytes. The
  // task's ignore-limit policy may admit it beyond the reservation using global capacity. Refusal
  // ends only the optional attempt; an escaping OOM would reschedule mandatory work after
  // publication was claimed.
  std::unique_ptr<rmm::device_buffer> scratch;
  if (!sources.empty()) {
    auto const total_chunks = _impl->keys.size() * geometry.chunks_per_key();
    auto const scratch_bytes =
      std::min<std::size_t>(2, total_chunks) * sources.size() * geometry.chunk_bytes;
    try {
      scratch = std::make_unique<rmm::device_buffer>(
        scratch_bytes, rmm::cuda_stream_view{stream.get()}, scratch_mr);
    } catch (rmm::out_of_memory const&) {
      SIRIUS_LOG_DEBUG("[accumulated_bloom_builder::finish] publication scratch refused.");
      return std::nullopt;
    }
  }

  std::vector<::cuda::stream_ref> joined{stream};
  for (auto const& entry : _impl->partials) {
    joined.push_back(entry->stream_ref());
  }
  std::size_t chunks = 0;
  try {
    join_on_failure(joined, [&] {
      impl::chunk_pipeline pipeline{*_impl,
                                    *root,
                                    sources,
                                    targets,
                                    stream,
                                    scratch ? static_cast<std::byte*>(scratch->data()) : nullptr};
      pipeline.order_after_inserts();
      for (std::size_t key_index = 0; key_index < _impl->keys.size(); ++key_index) {
        for (std::size_t offset = 0; offset < geometry.raw_bytes; offset += geometry.chunk_bytes) {
          pipeline.enqueue_chunk(key_index, offset);
        }
      }
      pipeline.wait_until_replicas_ready();
      chunks = pipeline.chunks();
    });
  } catch (unjoined_gpu_work const&) {
    _impl->leak_storage.store(true);
    // Task teardown detaches its tracker without freeing storage still reachable by GPU work.
    (void)scratch.release();
    throw;
  }
  scratch.reset();
  SIRIUS_LOG_DEBUG(
    "[accumulated_bloom_builder::finish] root GPU {} published {} key(s) in {} chunk(s): {} bytes "
    "in from {} source(s), {} bytes out to {} target(s).",
    root->device_id,
    _impl->keys.size(),
    chunks,
    sources.size() * _impl->keys.size() * geometry.raw_bytes,
    sources.size(),
    targets.size() * _impl->keys.size() * geometry.raw_bytes,
    targets.size());
  make_filters(root->device_id);
  return std::span<std::shared_ptr<sirius_dynamic_bloom_filter> const>{_impl->filters};
}

void accumulated_bloom_builder::make_filters(int root_device)
{
  std::vector<std::shared_ptr<sirius_dynamic_bloom_filter>> filters;
  filters.reserve(_impl->keys.size());
  for (auto const& key : _impl->keys) {
    auto const domain = classify_membership_key(key.type);
    if (!domain) {
      throw accumulation_invariant_error(
        "[accumulated_bloom_builder] accumulated key type has no key domain");
    }
    auto filter           = std::make_unique<sirius_dynamic_bloom_filter::impl>();
    filter->source_device = root_device;
    filter->accumulated_replicas.reserve(_impl->partials.size());
    for (auto const& entry : _impl->partials) {
      filter->accumulated_replicas.push_back(std::make_unique<accumulated_bloom_replica>(
        entry->device_id, _impl->geometry.blocks, entry->stream, nullptr));
    }
    filters.push_back(sirius_dynamic_bloom_filter::make_accumulated(*domain, std::move(filter)));
  }
  // Allocate every host shell before transferring leased arrays. From this point, only nonthrowing
  // ownership moves run, and the builder keeps an owner even when no channel accepts a key.
  _impl->filters = std::move(filters);
  for (std::size_t key_index = 0; key_index < _impl->keys.size(); ++key_index) {
    auto& replicas = _impl->filters[key_index]->_impl->accumulated_replicas;
    for (std::size_t gpu_index = 0; gpu_index < _impl->partials.size(); ++gpu_index) {
      replicas[gpu_index]->bits = std::move(_impl->partials[gpu_index]->arrays[key_index]);
    }
  }
}

std::optional<accumulated_bloom_builder::replica_contents>
accumulated_bloom_builder::inspect_replica(sirius_dynamic_bloom_filter const& filter,
                                           rmm::cuda_device_id device)
{
  auto const* replica = filter._impl ? filter._impl->find_accumulated(device.value()) : nullptr;
  if (replica == nullptr || !replica->bits) { return std::nullopt; }
  rmm::cuda_set_device_raii guard{device};
  auto const pending = cudaStreamQuery(replica->stream->value());
  if (pending != cudaErrorNotReady) { check_cuda(pending, "cudaStreamQuery(replica)"); }
  replica_contents contents{std::vector<std::byte>(replica->bits->size()), pending == cudaSuccess};
  check_cuda(cudaMemcpyAsync(contents.bytes.data(),
                             replica->bits->data(),
                             contents.bytes.size(),
                             cudaMemcpyDeviceToHost,
                             replica->stream->value()),
             "cudaMemcpyAsync(replica)");
  check_cuda(cudaStreamSynchronize(replica->stream->value()), "cudaStreamSynchronize(replica)");
  return contents;
}

bool accumulated_bloom_builder::releasable_here() const noexcept
{
  if (!_impl) { return true; }
  try {
    return std::ranges::none_of(_impl->partials, [](auto const& entry) {
      auto const* allocator =
        entry->space->template get_memory_resource_of<cucascade::memory::Tier::GPU>();
      return allocator != nullptr && allocator->is_stream_tracked(entry->stream_ref());
    });
  } catch (...) {
    return false;
  }
}

}  // namespace detail

}  // namespace sirius::op
