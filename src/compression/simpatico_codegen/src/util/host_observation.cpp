// SPDX-License-Identifier: Apache-2.0
#include "util/host_observation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace simpatico {
namespace {

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}  // namespace

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
  check_cuda(cudaHostAlloc(&fresh, grown, cudaHostAllocPortable));
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
                                 rmm::cuda_stream_view stream)
{
  void* const staging = thread_pinned_staging().reserve(bytes);
  check_cuda(cudaMemcpyAsync(
    staging ? staging : destination, source, bytes, cudaMemcpyDeviceToHost, stream.value()));
  check_cuda(cudaStreamSynchronize(stream.value()));
  if (staging) std::memcpy(destination, staging, bytes);
}

}  // namespace simpatico
