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

// Linked only into sirius_cuda_fault_test, together with --wrap for these calls.
#include "utils/cuda_status_fault.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace {
thread_local int armed_operation       = 0;
thread_local cudaError_t armed_status  = cudaSuccess;
thread_local std::size_t armed_skip    = 0;
thread_local cudaStream_t armed_stream = nullptr;
thread_local std::optional<sirius::test::cuda_transfer_observation> observed_transfers;
thread_local cucascade::memory::fixed_size_host_memory_resource const* observed_resource = nullptr;
thread_local std::size_t initial_host_bytes                                              = 0;

cudaError_t take_status(int operation, cudaStream_t stream = nullptr)
{
  if (armed_operation != operation || (armed_stream != nullptr && armed_stream != stream)) {
    return cudaSuccess;
  }
  if (armed_skip != 0) {
    --armed_skip;
    return cudaSuccess;
  }
  armed_operation = 0;
  return armed_status;
}

struct pool_access_fault {
  cudaMemPool_t pool;
  int accessor;
  std::size_t skip;
  cudaError_t status;
};
std::mutex pool_access_mutex;
std::optional<pool_access_fault> armed_pool_access;
bool pool_access_consumed = false;
}  // namespace

namespace sirius::test {
scoped_cuda_status_fault::scoped_cuda_status_fault(operation op,
                                                   cudaError_t status,
                                                   std::size_t skip,
                                                   cudaStream_t matching_stream)
{
  if (armed_operation != 0) { throw std::logic_error("a CUDA status fault is already armed"); }
  armed_status    = status;
  armed_skip      = skip;
  armed_stream    = matching_stream;
  armed_operation = op;
}
scoped_cuda_status_fault::~scoped_cuda_status_fault()
{
  armed_operation = 0;
  armed_stream    = nullptr;
}
bool scoped_cuda_status_fault::consumed() { return armed_operation == 0; }

scoped_cuda_transfer_observer::scoped_cuda_transfer_observer(
  cucascade::memory::fixed_size_host_memory_resource const& staging_resource)
{
  if (observed_transfers) { throw std::logic_error("a CUDA transfer observer is already active"); }
  observed_transfers.emplace();
  observed_resource  = &staging_resource;
  initial_host_bytes = staging_resource.get_total_allocated_bytes();
}

scoped_cuda_transfer_observer::~scoped_cuda_transfer_observer()
{
  observed_resource = nullptr;
  observed_transfers.reset();
}

cuda_transfer_observation scoped_cuda_transfer_observer::snapshot() const noexcept
{
  return *observed_transfers;
}

scoped_pool_access_fault::scoped_pool_access_fault(cudaMemPool_t pool,
                                                   int accessor,
                                                   std::size_t skip,
                                                   cudaError_t status)
{
  std::scoped_lock lock(pool_access_mutex);
  REQUIRE_FALSE(armed_pool_access.has_value());
  armed_pool_access    = pool_access_fault{pool, accessor, skip, status};
  pool_access_consumed = false;
}
scoped_pool_access_fault::~scoped_pool_access_fault()
{
  std::scoped_lock lock(pool_access_mutex);
  armed_pool_access.reset();
}
bool scoped_pool_access_fault::consumed() const
{
  std::scoped_lock lock(pool_access_mutex);
  return pool_access_consumed;
}
}  // namespace sirius::test

extern "C" cudaError_t __real_cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int);
extern "C" cudaError_t __real_cudaDeviceSynchronize();
extern "C" cudaError_t __real_cudaMemPoolGetAccess(cudaMemAccessFlags*,
                                                   cudaMemPool_t,
                                                   cudaMemLocation*);

extern "C" cudaError_t __wrap_cudaStreamWaitEvent(cudaStream_t stream,
                                                  cudaEvent_t event,
                                                  unsigned int flags)
{
  if (auto status = take_status(sirius::test::scoped_cuda_status_fault::wait_event, stream);
      status != cudaSuccess)
    return status;
  return __real_cudaStreamWaitEvent(stream, event, flags);
}

extern "C" cudaError_t __wrap_cudaDeviceSynchronize()
{
  if (auto status = take_status(sirius::test::scoped_cuda_status_fault::synchronize_device);
      status != cudaSuccess)
    return status;
  return __real_cudaDeviceSynchronize();
}

extern "C" cudaError_t __wrap_cudaMemPoolGetAccess(cudaMemAccessFlags* flags,
                                                   cudaMemPool_t pool,
                                                   cudaMemLocation* location)
{
  {
    std::scoped_lock lock(pool_access_mutex);
    auto& fault = armed_pool_access;
    if (fault && pool == fault->pool && location && location->id == fault->accessor) {
      if (fault->skip == 0) {
        auto const status = fault->status;
        fault.reset();
        pool_access_consumed = true;
        return status;
      }
      --fault->skip;
    }
  }
  return __real_cudaMemPoolGetAccess(flags, pool, location);
}

extern "C" cudaError_t __real_cudaMemcpyAsync(
  void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);

extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination,
                                              void const* source,
                                              std::size_t bytes,
                                              cudaMemcpyKind kind,
                                              cudaStream_t stream)
{
  int const operation =
    kind == cudaMemcpyDeviceToHost   ? sirius::test::scoped_cuda_status_fault::copy_device_to_host
    : kind == cudaMemcpyHostToDevice ? sirius::test::scoped_cuda_status_fault::copy_host_to_device
                                     : 0;
  if (operation != 0) {
    if (auto const status = take_status(operation, stream); status != cudaSuccess) {
      return status;
    }
  }
  return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream)
{
  // Completion faults report an error only after the real work has retired. Enqueue faults above
  // exercise cleanup joining accepted work.
  if (observed_transfers) {
    ++observed_transfers->synchronizations;
    if (observed_resource->get_total_allocated_bytes() > initial_host_bytes) {
      ++observed_transfers->synchronizations_with_staging;
    }
  }
  auto const real_status = __real_cudaStreamSynchronize(stream);
  if (real_status != cudaSuccess) { return real_status; }
  return take_status(sirius::test::scoped_cuda_status_fault::synchronize_stream, stream);
}

#if CUDART_VERSION >= 12080
#if CUDART_VERSION >= 13000
extern "C" cudaError_t __real_cudaMemcpyBatchAsync(void* const*,
                                                   void const* const*,
                                                   std::size_t const*,
                                                   std::size_t,
                                                   cudaMemcpyAttributes*,
                                                   std::size_t*,
                                                   std::size_t,
                                                   cudaStream_t);

extern "C" cudaError_t __wrap_cudaMemcpyBatchAsync(void* const* destinations,
                                                   void const* const* sources,
                                                   std::size_t const* sizes,
                                                   std::size_t count,
                                                   cudaMemcpyAttributes* attributes,
                                                   std::size_t* attribute_indices,
                                                   std::size_t attribute_count,
                                                   cudaStream_t stream)
#else
extern "C" cudaError_t __real_cudaMemcpyBatchAsync(void**,
                                                   void**,
                                                   std::size_t*,
                                                   std::size_t,
                                                   cudaMemcpyAttributes*,
                                                   std::size_t*,
                                                   std::size_t,
                                                   std::size_t*,
                                                   cudaStream_t);

extern "C" cudaError_t __wrap_cudaMemcpyBatchAsync(void** destinations,
                                                   void** sources,
                                                   std::size_t* sizes,
                                                   std::size_t count,
                                                   cudaMemcpyAttributes* attributes,
                                                   std::size_t* attribute_indices,
                                                   std::size_t attribute_count,
                                                   std::size_t* failed_index,
                                                   cudaStream_t stream)
#endif
{
  if (observed_transfers) { ++observed_transfers->batch_submissions; }
  auto const injected_status =
    take_status(sirius::test::scoped_cuda_status_fault::copy_batch, stream);
  if (injected_status != cudaSuccess) {
    // Accept a real prefix before reporting the later failure, so cleanup must join submitted work.
    if (count < 2 || attribute_count != 1 || attribute_indices[0] != 0) {
      return cudaErrorInvalidValue;
    }
    auto const status = __real_cudaMemcpyBatchAsync(destinations,
                                                    sources,
                                                    sizes,
                                                    1,
                                                    attributes,
                                                    attribute_indices,
                                                    attribute_count,
#if CUDART_VERSION < 13000
                                                    failed_index,
#endif
                                                    stream);
    if (status != cudaSuccess) { return status; }
    if (observed_transfers) { ++observed_transfers->partial_batch_submissions; }
    return injected_status;
  }
  return __real_cudaMemcpyBatchAsync(destinations,
                                     sources,
                                     sizes,
                                     count,
                                     attributes,
                                     attribute_indices,
                                     attribute_count,
#if CUDART_VERSION < 13000
                                     failed_index,
#endif
                                     stream);
}
#endif
