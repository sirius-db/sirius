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

#include "codegen/util/cuda_check.hpp"
#include "log/logging.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <expression_evaluator/nvrtc/string_constant_cache.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace sirius::nvrtc {

namespace {

struct event_deleter {
  void operator()(cudaEvent_t event) const noexcept { (void)cudaEventDestroy(event); }
};

/** @brief An owned CUDA event of the device that was current when it was created. */
using unique_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, event_deleter>;

[[nodiscard]] unique_event make_event()
{
  cudaEvent_t event = nullptr;
  simpatico::throw_if_cuda_error(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                 "string constant cache: event create");
  return unique_event{event};
}

/**
 * @brief Queues a copy of the host bytes @p source to @p destination on @p stream, reading
 * @p source before returning, so the host never waits for work already queued on @p stream.
 */
void copy_to_device(void* destination, std::string_view source, cudaStream_t stream)
{
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes attributes{};
  attributes.srcAccessOrder = cudaMemcpySrcAccessOrderDuringApiCall;
  attributes.flags          = cudaMemcpyFlagDefault;
  std::array<void*, 1> destinations{destination};
  std::array<std::size_t, 1> sizes{source.size()};
  std::size_t attribute_index = 0;
#if CUDART_VERSION >= 13000
  std::array<void const*, 1> sources{source.data()};
  simpatico::throw_if_cuda_error(cudaMemcpyBatchAsync(destinations.data(),
                                                      sources.data(),
                                                      sizes.data(),
                                                      1,
                                                      &attributes,
                                                      &attribute_index,
                                                      1,
                                                      stream),
                                 "string constant cache: host-to-device copy");
#else
  std::array<void*, 1> sources{const_cast<char*>(source.data())};
  std::size_t failed_index = 0;
  simpatico::throw_if_cuda_error(cudaMemcpyBatchAsync(destinations.data(),
                                                      sources.data(),
                                                      sizes.data(),
                                                      1,
                                                      &attributes,
                                                      &attribute_index,
                                                      1,
                                                      &failed_index,
                                                      stream),
                                 "string constant cache: host-to-device copy");
#endif
#else
  // Without batch copies, a copy from pageable memory may wait for `stream`, which carries only
  // this cache's uploads.
  simpatico::throw_if_cuda_error(
    cudaMemcpyAsync(destination, source.data(), source.size(), cudaMemcpyHostToDevice, stream),
    "string constant cache: host-to-device copy");
#endif
}

}  // namespace

/**
 * @brief The copies on one device, the stream that uploads and frees them, and the last work that
 * may read them.
 *
 * `stream` is declared first so that it outlives the copies, which free their memory on it.
 */
struct string_constant_cache::device_copies {
  /** @brief One pool's copy, published once by the acquisition that uploads it. */
  struct pool_copy {
    enum class state : std::uint8_t { uploading, ready, failed };

    std::atomic<state> status{state::uploading};
    /** @brief The copy's charge to its GPU memory space, or null when the device has none. */
    std::unique_ptr<cucascade::memory::reserved_arena> charge;
    std::unique_ptr<rmm::device_buffer> bytes;
    /** @brief Recorded on `stream` after the copy. */
    unique_event uploaded;
  };

  /**
   * @param charged The reservation-aware allocator of the device's GPU memory space, or null to
   * allocate from the device's current resource without a charge.
   */
  explicit device_copies(cucascade::memory::reservation_aware_resource_adaptor* charged)
    : resource(charged != nullptr ? charged->get_upstream_resource()
                                  : rmm::mr::get_current_device_resource_ref()),
      charged(charged)
  {
  }

  device_copies(device_copies const&)            = delete;
  device_copies& operator=(device_copies const&) = delete;

  /**
   * @brief Makes the stream wait for every recorded use, after which the copies free their memory
   * on it; call on this device.
   *
   * When a use could not be recorded, nothing orders the free after it, so the copies, and their
   * charges, are leaked instead.
   */
  ~device_copies()
  {
    bool ordered = !use_unrecorded.load(std::memory_order_relaxed);
    for (auto const& use : last_uses) {
      ordered = ordered && cudaStreamWaitEvent(stream.value(), use.second.get(), 0) == cudaSuccess;
    }
    if (ordered) { return; }
    sirius::log::log_noexcept([&] {
      SIRIUS_LOG_ERROR(
        "[string_constant_cache] leaking {} string-constant copies: a lease's last use could not "
        "be ordered before their release",
        copies.size());
    });
    for (auto& [pool, copy] : copies) {
      (void)copy->bytes.release();
      (void)copy->charge.release();
    }
  }

  /**
   * @brief Returns the copy of @p pool and the last-use event of @p stream, uploading the copy if
   * this is its first acquisition.
   */
  [[nodiscard]] std::pair<pool_copy const*, cudaEvent_t> acquire(std::string_view pool,
                                                                 ::cuda::stream_ref stream)
  {
    while (true) {
      std::shared_ptr<pool_copy> pending;
      cudaEvent_t last_use = nullptr;
      std::optional<decltype(copies)::iterator> to_upload;
      {
        std::lock_guard const lock(mutex);
        last_use   = find_last_use(stream);
        auto found = copies.find(pool);
        if (found == copies.end()) {
          found     = copies.emplace(std::string(pool), std::make_shared<pool_copy>()).first;
          to_upload = found;
        } else if (last_use != nullptr && found->second->status.load(std::memory_order_acquire) ==
                                            pool_copy::state::ready) {
          // A ready copy is never erased, so it outlives the lock without a reference count.
          return {found->second.get(), last_use};
        }
        pending = found->second;
      }
      if (last_use == nullptr) { last_use = add_last_use(stream); }
      if (to_upload) {
        upload(*to_upload);
      } else {
        pending->status.wait(pool_copy::state::uploading, std::memory_order_acquire);
        // A failed upload was erased; the next attempt uploads the pool afresh.
        if (pending->status.load(std::memory_order_acquire) == pool_copy::state::failed) {
          continue;
        }
      }
      return {pending.get(), last_use};
    }
  }

  rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
  /** @brief Allocates and frees the copies. */
  rmm::device_async_resource_ref resource;
  cucascade::memory::reservation_aware_resource_adaptor* charged;
  /** @brief Guards `copies` and `last_uses`; never held across a CUDA call. */
  std::mutex mutex;
  /** @brief The copies keyed by their bytes. */
  std::map<std::string, std::shared_ptr<pool_copy>, std::less<>> copies;
  /** @brief One event per stream a lease was acquired on, recorded at the stream's latest release.
   */
  std::vector<std::pair<cudaStream_t, unique_event>> last_uses;
  /** @brief Whether a release failed to record its use, which leaves no safe point to free the
   * copies. */
  std::atomic<bool> use_unrecorded{false};

 private:
  /** @brief Returns the last-use event of @p stream, or null if it has none; call under `mutex`. */
  [[nodiscard]] cudaEvent_t find_last_use(::cuda::stream_ref stream) const noexcept
  {
    auto const use =
      std::ranges::find(last_uses, stream.get(), &decltype(last_uses)::value_type::first);
    return use != last_uses.end() ? use->second.get() : nullptr;
  }

  /** @brief Creates the last-use event of @p stream unless another acquisition already did. */
  [[nodiscard]] cudaEvent_t add_last_use(::cuda::stream_ref stream)
  {
    auto event = make_event();
    std::lock_guard const lock(mutex);
    if (auto const existing = find_last_use(stream); existing != nullptr) { return existing; }
    return last_uses.emplace_back(stream.get(), std::move(event)).second.get();
  }

  /** @brief Uploads the pool @p entry names and publishes the outcome to concurrent waiters. */
  void upload(decltype(copies)::iterator entry)
  {
    auto const& pool = entry->first;
    auto& copy       = *entry->second;
    try {
      if (charged != nullptr) {
        copy.charge = charged->reserve(rmm::align_up(pool.size(), rmm::CUDA_ALLOCATION_ALIGNMENT));
        if (!copy.charge) {
          throw rmm::out_of_memory(
            "string constant cache: the GPU memory space cannot hold a string-constant copy");
        }
      }
      copy.bytes =
        std::make_unique<rmm::device_buffer>(pool.size(), ::cuda::stream_ref{stream}, resource);
      copy.uploaded = make_event();
      copy_to_device(copy.bytes->data(), pool, stream.value());
      simpatico::throw_if_cuda_error(cudaEventRecord(copy.uploaded.get(), stream.value()),
                                     "string constant cache: upload event record");
    } catch (...) {
#if CUDART_VERSION < 12080
      // The copy may still read the key from pageable memory.
      (void)cudaStreamSynchronize(stream.value());
#endif
      std::shared_ptr<pool_copy> failed;
      {
        std::lock_guard const lock(mutex);
        failed = std::move(entry->second);
        copies.erase(entry);
      }
      // The copy's memory is freed on `stream`, after anything the failed upload queued there.
      failed->status.store(pool_copy::state::failed, std::memory_order_release);
      failed->status.notify_all();
      throw;
    }
    copy.status.store(pool_copy::state::ready, std::memory_order_release);
    copy.status.notify_all();
  }
};

string_constant_cache::string_constant_cache() = default;

string_constant_cache::string_constant_cache(
  std::vector<cucascade::memory::memory_space const*> gpu_spaces)
  : _gpu_spaces(std::move(gpu_spaces))
{
}

string_constant_cache::~string_constant_cache()
{
  for (auto& [device, copies] : _devices) {
    rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{device}};
    copies.reset();
  }
}

string_constant_cache::lease string_constant_cache::acquire(std::string_view pool,
                                                            ::cuda::stream_ref stream) const
{
  if (pool.empty()) { throw internal_exception("[nvrtc] empty string constant pool"); }
  int device = 0;
  simpatico::throw_if_cuda_error(cudaGetDevice(&device), "string constant cache: cudaGetDevice");
  auto& on_device             = copies_of(device);
  auto const [copy, last_use] = on_device.acquire(pool, stream);
  simpatico::throw_if_cuda_error(cudaStreamWaitEvent(stream.get(), copy->uploaded.get(), 0),
                                 "string constant cache: upload event wait");
  return lease{on_device, static_cast<char const*>(copy->bytes->data()), last_use, stream};
}

std::size_t string_constant_cache::upload_count_for_testing() const
{
  std::lock_guard const lock(_mutex);
  std::size_t count = 0;
  for (auto const& [device, copies] : _devices) {
    std::lock_guard const device_lock(copies->mutex);
    count += copies->copies.size();
  }
  return count;
}

string_constant_cache::device_copies& string_constant_cache::copies_of(int device) const
{
  {
    std::lock_guard const lock(_mutex);
    if (auto const found = _devices.find(device); found != _devices.end()) {
      return *found->second;
    }
  }
  auto const space = std::ranges::find(
    _gpu_spaces, device, [](auto const* space) { return space->get_device_id(); });
  auto created = std::make_unique<device_copies>(
    space != _gpu_spaces.end() ? (*space)->get_memory_resource_of<cucascade::memory::Tier::GPU>()
                               : nullptr);
  std::lock_guard const lock(_mutex);
  return *_devices.try_emplace(device, std::move(created)).first->second;
}

//===----------------------------------------------------------------------===//
// lease
//===----------------------------------------------------------------------===//
string_constant_cache::lease::lease(device_copies& copies,
                                    char const* data,
                                    cudaEvent_t last_use,
                                    ::cuda::stream_ref stream)
  : _copies(&copies), _data(data), _last_use(last_use), _stream(stream)
{
}

string_constant_cache::lease::lease(lease&& other) noexcept
  : _copies(std::exchange(other._copies, nullptr)),
    _data(other._data),
    _last_use(other._last_use),
    _stream(other._stream)
{
}

string_constant_cache::lease& string_constant_cache::lease::operator=(lease&& other) noexcept
{
  if (this != &other) {
    release();
    _copies   = std::exchange(other._copies, nullptr);
    _data     = other._data;
    _last_use = other._last_use;
    _stream   = other._stream;
  }
  return *this;
}

string_constant_cache::lease::~lease() { release(); }

void string_constant_cache::lease::release() noexcept
{
  // Re-recording the stream's one event moves it past every earlier release on that stream.
  if (auto* const copies = std::exchange(_copies, nullptr); copies != nullptr) {
    if (cudaEventRecord(_last_use, _stream.get()) != cudaSuccess) {
      copies->use_unrecorded.store(true, std::memory_order_relaxed);
    }
  }
}

}  // namespace sirius::nvrtc
