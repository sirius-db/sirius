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

// Linked only into sirius_cuda_fault_test, together with the matching --wrap options.
#include "utils/cuda_status_fault.hpp"

#include "cuda/device_copy_batch.hpp"

#include <rmm/detail/error.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/stream_pool.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
using status_fault = sirius::test::scoped_cuda_status_fault;

std::atomic<cudaError_t> acquisition_status{cudaSuccess};

struct submitted_stream {
  cudaStream_t stream = nullptr;
  bool joined         = false;
};

struct thread_fault {
  status_fault::operation operation;
  cudaError_t status;
  std::size_t skip;
  cudaStream_t matching_stream;
  std::chrono::milliseconds delay;
  bool consumed     = false;
  bool invalid_join = false;
  std::vector<submitted_stream> streams;
};
thread_local std::optional<thread_fault> armed_fault;
thread_local std::optional<sirius::test::cuda_transfer_observation> observed_transfers;
thread_local cucascade::memory::fixed_size_host_memory_resource const* observed_resource = nullptr;
thread_local std::size_t initial_host_bytes                                              = 0;

// A thread other than the arming one sees no fault, which reads as consumed.
thread_fault const* calling_thread_fault() noexcept
{
  return armed_fault ? &*armed_fault : nullptr;
}

cudaError_t take_status(int operation, cudaStream_t stream = nullptr)
{
  if (!armed_fault || armed_fault->consumed || armed_fault->operation != operation ||
      (armed_fault->matching_stream != nullptr && armed_fault->matching_stream != stream)) {
    return cudaSuccess;
  }
  if (armed_fault->skip != 0) {
    --armed_fault->skip;
    return cudaSuccess;
  }
  armed_fault->consumed = true;
  if (armed_fault->delay.count() != 0 && stream != nullptr) {
    auto duration     = std::make_unique<std::chrono::milliseconds>(armed_fault->delay);
    auto const status = cudaLaunchHostFunc(
      stream,
      [](void* value) {
        std::unique_ptr<std::chrono::milliseconds> retained{
          static_cast<std::chrono::milliseconds*>(value)};
        std::this_thread::sleep_for(*retained);
      },
      duration.get());
    if (status != cudaSuccess) { return status; }
    (void)duration.release();
  }
  return armed_fault->status;
}

// Marks @p stream as holding work that no later synchronization has yet joined.
void record_submission(cudaStream_t stream)
{
  if (!armed_fault) { return; }
  auto& streams = armed_fault->streams;
  if (auto const found = std::ranges::find(streams, stream, &submitted_stream::stream);
      found != streams.end()) {
    found->joined = false;
    return;
  }
  streams.push_back({stream, false});
}

void record_join(cudaStream_t stream, cudaError_t status)
{
  if (!armed_fault) { return; }
  armed_fault->invalid_join |= status == cudaErrorInvalidResourceHandle;
  if (status != cudaSuccess) { return; }
  for (auto& submitted : armed_fault->streams) {
    if (submitted.stream == stream) { submitted.joined = true; }
  }
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
                                                   cudaStream_t matching_stream,
                                                   std::chrono::milliseconds delay)
  : operation_(op)
{
  REQUIRE((status != cudaSuccess || delay.count() != 0));
  if (op == acquire_stream) {
    REQUIRE(skip == 0);
    REQUIRE(matching_stream == nullptr);
    REQUIRE(delay.count() == 0);
    auto expected = cudaSuccess;
    REQUIRE(acquisition_status.compare_exchange_strong(expected, status));
    return;
  }
  if (armed_fault) { throw std::logic_error("a CUDA status fault is already armed"); }
  armed_fault.emplace(thread_fault{.operation       = op,
                                   .status          = status,
                                   .skip            = skip,
                                   .matching_stream = matching_stream,
                                   .delay           = delay});
}
scoped_cuda_status_fault::~scoped_cuda_status_fault()
{
  if (operation_ == acquire_stream) {
    acquisition_status.store(cudaSuccess);
  } else {
    armed_fault.reset();
  }
}
bool scoped_cuda_status_fault::consumed() const noexcept
{
  if (operation_ == acquire_stream) { return acquisition_status.load() == cudaSuccess; }
  auto const* fault = calling_thread_fault();
  return fault == nullptr || fault->consumed;
}
std::size_t scoped_cuda_status_fault::observed_stream_count() const noexcept
{
  auto const* fault = operation_ == acquire_stream ? nullptr : calling_thread_fault();
  return fault == nullptr ? 0 : fault->streams.size();
}
bool scoped_cuda_status_fault::all_submitted_streams_joined() const noexcept
{
  auto const* fault = operation_ == acquire_stream ? nullptr : calling_thread_fault();
  return fault == nullptr || std::ranges::all_of(fault->streams, &submitted_stream::joined);
}
bool scoped_cuda_status_fault::invalid_join() const noexcept
{
  auto const* fault = operation_ == acquire_stream ? nullptr : calling_thread_fault();
  return fault != nullptr && fault->invalid_join;
}

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
extern "C" cudaError_t __real_cudaMemcpyPeerAsync(
  void*, int, void const*, int, std::size_t, cudaStream_t);
extern "C" cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);

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
  // An unjoined failure skips the real wait, so cleanup must still join the stream's work.
  if (auto const injected =
        take_status(sirius::test::scoped_cuda_status_fault::synchronize_stream_unjoined, stream);
      injected != cudaSuccess) {
    record_submission(stream);
    return injected;
  }
  auto const real_status = __real_cudaStreamSynchronize(stream);
  record_join(stream, real_status);
  if (real_status != cudaSuccess) { return real_status; }
  return take_status(sirius::test::scoped_cuda_status_fault::synchronize_stream, stream);
}

extern "C" cudaError_t __wrap_cudaMemcpyPeerAsync(void* dst,
                                                  int dst_device,
                                                  void const* src,
                                                  int src_device,
                                                  std::size_t bytes,
                                                  cudaStream_t stream)
{
  auto const injected = take_status(sirius::test::scoped_cuda_status_fault::copy_peer, stream);
  record_submission(stream);
  if (injected != cudaSuccess) { return injected; }
  return __real_cudaMemcpyPeerAsync(dst, dst_device, src, src_device, bytes, stream);
}

extern "C" cudaError_t __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  auto const injected = take_status(sirius::test::scoped_cuda_status_fault::record_event, stream);
  record_submission(stream);
  if (injected != cudaSuccess) { return injected; }
  return __real_cudaEventRecord(event, stream);
}

extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)
{
  auto const injected = take_status(sirius::test::scoped_cuda_status_fault::create_event);
  if (injected != cudaSuccess) { return injected; }
  return __real_cudaEventCreateWithFlags(event, flags);
}

// Wrap the Sirius call boundary: libcudf's internal runtime calls cannot be link-wrapped here.
extern "C" cudaError_t __real__ZNK6sirius4cuda17device_copy_batch7enqueueEN4cuda3__410stream_refE(
  sirius::cuda::device_copy_batch const*, ::cuda::stream_ref);
extern "C" cudaError_t __wrap__ZNK6sirius4cuda17device_copy_batch7enqueueEN4cuda3__410stream_refE(
  sirius::cuda::device_copy_batch const* batch, ::cuda::stream_ref stream)
{
  auto const injected =
    take_status(sirius::test::scoped_cuda_status_fault::enqueue_copy_batch, stream.get());
  record_submission(stream.get());
  if (injected != cudaSuccess) { return injected; }
  return __real__ZNK6sirius4cuda17device_copy_batch7enqueueEN4cuda3__410stream_refE(batch, stream);
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

// Wrap the out-of-line cuCascade acquisition boundary. RMM's constructor lives
// in a shared library, so wrapping CUDA creation in this executable would not
// intercept its calls. The wrapper produces the same typed RMM exception as
// GROW failure, and never changes production code or the real CUDA context.
using stream_pool = cucascade::memory::exclusive_stream_pool;
extern "C" cucascade::memory::borrowed_stream
real_stream_acquire(stream_pool*, stream_pool::stream_acquire_policy) asm(
  "__real__ZN9cucascade6memory21exclusive_stream_pool14acquire_streamENS1_21stream_acquire_"
  "policyE");
extern "C" cucascade::memory::borrowed_stream
wrap_stream_acquire(stream_pool*, stream_pool::stream_acquire_policy) asm(
  "__wrap__ZN9cucascade6memory21exclusive_stream_pool14acquire_streamENS1_21stream_acquire_"
  "policyE");
extern "C" cucascade::memory::borrowed_stream wrap_stream_acquire(
  stream_pool* pool, stream_pool::stream_acquire_policy policy)
{
  auto status = acquisition_status.exchange(cudaSuccess);
  RMM_CUDA_TRY(status);
  return real_stream_acquire(pool, policy);
}
