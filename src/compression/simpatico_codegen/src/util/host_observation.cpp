// SPDX-License-Identifier: Apache-2.0
#include "util/host_observation.hpp"

#include "codegen/util/cuda_check.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

namespace simpatico {

pinned_staging_slab::~pinned_staging_slab()
{
  if (data_) (void)cudaFreeHost(data_);
}

void* pinned_staging_slab::reserve(std::size_t bytes)
{
  if (bytes > cap_bytes) return nullptr;
  if (data_ && bytes <= capacity_) return data_;
  // Allocate before releasing so a failed growth leaves the current slab usable.
  auto const grown = std::min(cap_bytes, std::max({bytes, 2 * capacity_, initial_bytes}));
  void* fresh      = nullptr;
  throw_if_cuda_error(cudaHostAlloc(&fresh, grown, cudaHostAllocPortable),
                      "host observation: pinned staging allocation");
  if (data_) (void)cudaFreeHost(data_);
  data_     = fresh;
  capacity_ = grown;
  return data_;
}

pinned_staging_slab& thread_pinned_staging()
{
  thread_local pinned_staging_slab slab;
  return slab;
}

void read_device_bytes_completed(void* destination,
                                 void const* source,
                                 std::size_t bytes,
                                 ::cuda::stream_ref stream)
{
  try {
    void* const staging = thread_pinned_staging().reserve(bytes);
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
