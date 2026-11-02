// SPDX-License-Identifier: Apache-2.0
#include "util/host_observation.hpp"

#include "codegen/util/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace simpatico {

namespace {

// Pinned storage one thread reuses for its observation copies. The read and the write path each
// keep their own thread_local slab (a device-to-host readback never shares storage with a pending
// host-to-device write). Thread-exit destruction may run after the CUDA context is gone, so the
// destructor ignores the cudaFreeHost result, the convention stream_pool::shutdown follows.
class pinned_staging_slab {
 public:
  pinned_staging_slab() = default;
  ~pinned_staging_slab()
  {
    if (event_) {
      if (pending_) (void)cudaEventSynchronize(event_);
      (void)cudaEventDestroy(event_);
    }
    if (data_) (void)cudaFreeHost(data_);
  }
  pinned_staging_slab(pinned_staging_slab const&)            = delete;
  pinned_staging_slab& operator=(pinned_staging_slab const&) = delete;

  // Storage of at least `bytes`, or nullptr above the cap. If a previous host-to-device copy from
  // this slab is still pending, waits for its recorded event before returning the buffer.
  void* reserve(std::size_t bytes)
  {
    if (bytes > pinned_staging_cap_bytes) return nullptr;
    if (pending_) {
      throw_if_cuda_error(cudaEventSynchronize(event_),
                          "host observation: pinned staging event wait");
      pending_ = false;
    }
    if (data_ && bytes <= capacity_) return data_;
    // Allocate before releasing so a failed growth leaves the current slab usable.
    auto const grown =
      std::min(pinned_staging_cap_bytes, std::max({bytes, 2 * capacity_, initial_bytes}));
    void* fresh = nullptr;
    throw_if_cuda_error(cudaHostAlloc(&fresh, grown, cudaHostAllocPortable),
                        "host observation: pinned staging allocation");
    if (data_) (void)cudaFreeHost(data_);
    data_     = fresh;
    capacity_ = grown;
    return data_;
  }

  // Record that an asynchronous copy reading this slab was queued on `stream`.
  void mark_pending(::cuda::stream_ref stream)
  {
    if (!event_) {
      throw_if_cuda_error(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
                          "host observation: pinned staging event create");
    }
    throw_if_cuda_error(cudaEventRecord(event_, stream.get()),
                        "host observation: pinned staging event record");
    pending_ = true;
  }

 private:
  static constexpr std::size_t initial_bytes = std::size_t{64} << 10;
  void* data_                                = nullptr;
  std::size_t capacity_                      = 0;
  cudaEvent_t event_                         = nullptr;
  bool pending_                              = false;
};

}  // namespace

void read_device_bytes_completed(void* destination,
                                 void const* source,
                                 std::size_t bytes,
                                 ::cuda::stream_ref stream)
{
  device_read const read{destination, source, bytes};
  read_device_bytes_completed(std::span{&read, 1}, stream);
}

void read_device_bytes_completed(std::span<device_read const> reads, ::cuda::stream_ref stream)
{
  // Each read starts at an 8-byte boundary of the staging storage.
  auto const staged_offset = [](std::size_t offset) { return (offset + 7) & ~std::size_t{7}; };
  std::size_t staged_bytes = 0;
  for (auto const& read : reads) {
    staged_bytes = staged_offset(staged_bytes) + read.bytes;
  }
  try {
    thread_local pinned_staging_slab slab;
    auto* const staging = static_cast<std::byte*>(slab.reserve(staged_bytes));
    std::size_t offset  = 0;
    for (auto const& read : reads) {
      offset = staged_offset(offset);
      throw_if_cuda_error(cudaMemcpyAsync(staging ? staging + offset : read.destination,
                                          read.source,
                                          read.bytes,
                                          cudaMemcpyDeviceToHost,
                                          stream.get()),
                          "host observation: device-to-host copy");
      offset += read.bytes;
    }
    throw_if_cuda_error(cudaStreamSynchronize(stream.get()), "host observation: stream wait");
    if (staging) {
      offset = 0;
      for (auto const& read : reads) {
        offset = staged_offset(offset);
        std::memcpy(read.destination, staging + offset, read.bytes);
        offset += read.bytes;
      }
    }
  } catch (...) {
    // A copy may still be queued into a destination or the staging storage.
    (void)cudaStreamSynchronize(stream.get());
    throw;
  }
}

void write_device_bytes_async(void* destination,
                              void const* source,
                              std::size_t bytes,
                              ::cuda::stream_ref stream)
{
  if (bytes == 0) return;
  thread_local pinned_staging_slab slab;
  void* const staging = slab.reserve(bytes);
  if (staging) {
    std::memcpy(staging, source, bytes);
    throw_if_cuda_error(
      cudaMemcpyAsync(destination, staging, bytes, cudaMemcpyHostToDevice, stream.get()),
      "host observation: host-to-device copy");
    slab.mark_pending(stream);
  } else {
    throw_if_cuda_error(
      cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream.get()),
      "host observation: unbuffered host-to-device copy");
    throw_if_cuda_error(cudaStreamSynchronize(stream.get()),
                        "host observation: unbuffered host-to-device wait");
  }
}

}  // namespace simpatico
