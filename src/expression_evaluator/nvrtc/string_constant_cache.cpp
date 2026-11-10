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

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <expression_evaluator/nvrtc/string_constant_cache.hpp>
#include <sirius/exception.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace sirius::nvrtc {

namespace {

struct event_deleter {
  void operator()(cudaEvent_t event) const noexcept { (void)cudaEventDestroy(event); }
};

/** @brief An owned CUDA event of the device that was current when it was created. */
using unique_event = std::unique_ptr<CUevent_st, event_deleter>;

[[nodiscard]] unique_event make_event()
{
  cudaEvent_t event = nullptr;
  simpatico::throw_if_cuda_error(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                                 "string constant cache: event create");
  return unique_event{event};
}

}  // namespace

/**
 * @brief The copies on one device, the stream that uploads them, and the last work that may read
 *        them.
 *
 * `upload_stream` is declared first so that it outlives the copies, which free their memory on it.
 */
struct string_constant_cache::device_copies {
  struct pool_copy {
    rmm::device_buffer bytes;
    unique_event uploaded;  ///< Recorded on `upload_stream` after the copy.
  };

  rmm::cuda_stream upload_stream{rmm::cuda_stream::flags::non_blocking};
  /// The copies keyed by their bytes; a key is also the source of its copy's upload.
  std::map<std::string, pool_copy, std::less<>> copies;
  /// One event per stream a lease was released on, recorded at its latest release.
  std::vector<std::pair<cudaStream_t, unique_event>> last_uses;
  /// Whether a release failed to record its use, which leaves only a device-wide wait safe.
  bool use_unrecorded{false};

  /** @brief Uploads @p pool from memory allocated by @p mr and returns its copy. */
  pool_copy const& upload(std::string_view pool, rmm::device_async_resource_ref mr)
  {
    auto const stream = upload_stream.value();
    auto const copy   = copies.try_emplace(std::string(pool)).first;
    auto& fresh       = copy->second;
    try {
      fresh.bytes    = rmm::device_buffer{pool.size(), upload_stream.view(), mr};
      fresh.uploaded = make_event();
      // The key is the source: it stays in place until the destructor has waited for the upload.
      // Copying from pageable memory may block until `upload_stream` drains, and it carries only
      // this cache's uploads.
      simpatico::throw_if_cuda_error(
        cudaMemcpyAsync(
          fresh.bytes.data(), copy->first.data(), pool.size(), cudaMemcpyHostToDevice, stream),
        "string constant cache: host-to-device copy");
      simpatico::throw_if_cuda_error(cudaEventRecord(fresh.uploaded.get(), stream),
                                     "string constant cache: upload event record");
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      copies.erase(copy);
      throw;
    }
    return fresh;
  }

  /** @brief Records that the work queued on @p stream so far may read the copies. */
  void record_use(::cuda::stream_ref stream) noexcept
  {
    try {
      auto use =
        std::ranges::find(last_uses, stream.get(), &decltype(last_uses)::value_type::first);
      if (use == last_uses.end()) {
        last_uses.emplace_back(stream.get(), make_event());
        use = std::prev(last_uses.end());
      }
      if (cudaEventRecord(use->second.get(), stream.get()) != cudaSuccess) {
        use_unrecorded = true;
      }
    } catch (...) {
      use_unrecorded = true;
    }
  }

  /** @brief Waits on the host for the uploads and for every recorded use; call on this device. */
  void wait_for_readers() const noexcept
  {
    if (use_unrecorded) {
      (void)cudaDeviceSynchronize();
      return;
    }
    (void)cudaStreamSynchronize(upload_stream.value());
    for (auto const& use : last_uses) {
      (void)cudaEventSynchronize(use.second.get());
    }
  }
};

string_constant_cache::string_constant_cache() = default;

string_constant_cache::~string_constant_cache()
{
  for (auto& [device, copies] : _devices) {
    rmm::cuda_set_device_raii const on_device{rmm::cuda_device_id{device}};
    copies->wait_for_readers();
    copies.reset();
  }
}

string_constant_cache::lease string_constant_cache::acquire(std::string_view pool,
                                                            ::cuda::stream_ref stream,
                                                            rmm::device_async_resource_ref mr) const
{
  if (pool.empty()) { throw internal_exception("[nvrtc] empty string constant pool"); }
  int device = 0;
  simpatico::throw_if_cuda_error(cudaGetDevice(&device), "string constant cache: cudaGetDevice");
  std::lock_guard const lock(_mutex);
  auto& on_device = _devices[device];
  if (!on_device) { on_device = std::make_unique<device_copies>(); }
  auto const found = on_device->copies.find(pool);
  auto const& copy = found != on_device->copies.end() ? found->second : on_device->upload(pool, mr);
  simpatico::throw_if_cuda_error(cudaStreamWaitEvent(stream.get(), copy.uploaded.get(), 0),
                                 "string constant cache: upload event wait");
  return lease{*this, static_cast<char const*>(copy.bytes.data()), device, stream};
}

std::size_t string_constant_cache::upload_count_for_testing() const
{
  std::lock_guard const lock(_mutex);
  std::size_t count = 0;
  for (auto const& [device, copies] : _devices) {
    count += copies->copies.size();
  }
  return count;
}

void string_constant_cache::record_use(int device, ::cuda::stream_ref stream) const noexcept
{
  std::lock_guard const lock(_mutex);
  if (auto const on_device = _devices.find(device); on_device != _devices.end()) {
    on_device->second->record_use(stream);
  }
}

//===----------------------------------------------------------------------===//
// lease
//===----------------------------------------------------------------------===//
string_constant_cache::lease::lease(string_constant_cache const& cache,
                                    char const* data,
                                    int device,
                                    ::cuda::stream_ref stream)
  : _cache(&cache), _data(data), _device(device), _stream(stream)
{
}

string_constant_cache::lease::lease(lease&& other) noexcept
  : _cache(std::exchange(other._cache, nullptr)),
    _data(other._data),
    _device(other._device),
    _stream(other._stream)
{
}

string_constant_cache::lease& string_constant_cache::lease::operator=(lease&& other) noexcept
{
  if (this != &other) {
    release();
    _cache  = std::exchange(other._cache, nullptr);
    _data   = other._data;
    _device = other._device;
    _stream = other._stream;
  }
  return *this;
}

string_constant_cache::lease::~lease() { release(); }

void string_constant_cache::lease::release() noexcept
{
  if (_cache != nullptr) { std::exchange(_cache, nullptr)->record_use(_device, _stream); }
}

}  // namespace sirius::nvrtc
