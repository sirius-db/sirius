// SPDX-License-Identifier: Apache-2.0
#include "util/host_observation.hpp"

#include "codegen/util/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

namespace simpatico {

namespace {

// Pinned storage one thread reuses as the destination of its device-to-host observation copies.
// Thread-exit destruction may run after the CUDA context is gone, so the destructor ignores the
// cudaFreeHost result, the convention stream_pool::shutdown follows.
class pinned_staging_slab {
 public:
  pinned_staging_slab() = default;
  ~pinned_staging_slab()
  {
    if (data_) (void)cudaFreeHost(data_);
  }
  pinned_staging_slab(pinned_staging_slab const&)            = delete;
  pinned_staging_slab& operator=(pinned_staging_slab const&) = delete;

  // Storage of at least `bytes`, or nullptr above the cap. It is valid until the next call, which
  // does not wait for pending copies.
  void* reserve(std::size_t bytes)
  {
    if (bytes > pinned_staging_cap_bytes) return nullptr;
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

 private:
  static constexpr std::size_t initial_bytes = std::size_t{64} << 10;
  void* data_                                = nullptr;
  std::size_t capacity_                      = 0;
};

}  // namespace

void read_device_bytes_completed(void* destination,
                                 void const* source,
                                 std::size_t bytes,
                                 ::cuda::stream_ref stream)
{
  try {
    thread_local pinned_staging_slab slab;
    void* const staging = slab.reserve(bytes);
    throw_if_cuda_error(
      cudaMemcpyAsync(
        staging ? staging : destination, source, bytes, cudaMemcpyDeviceToHost, stream.get()),
      "host observation: device-to-host copy");
    throw_if_cuda_error(cudaStreamSynchronize(stream.get()), "host observation: stream wait");
    if (staging) std::memcpy(destination, staging, bytes);
  } catch (...) {
    // A copy may still be queued into the destination or the staging storage.
    (void)cudaStreamSynchronize(stream.get());
    throw;
  }
}

}  // namespace simpatico
