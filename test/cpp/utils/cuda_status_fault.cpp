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

#include <cstddef>
#include <mutex>
#include <optional>
#include <utility>

namespace {
thread_local int armed_operation      = 0;
thread_local cudaError_t armed_status = cudaSuccess;

cudaError_t take_status(int operation)
{
  if (armed_operation != operation) return cudaSuccess;
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
scoped_cuda_status_fault::scoped_cuda_status_fault(operation op, cudaError_t status)
{
  REQUIRE(armed_operation == 0);
  armed_status    = status;
  armed_operation = op;
}
scoped_cuda_status_fault::~scoped_cuda_status_fault() { armed_operation = 0; }
bool scoped_cuda_status_fault::consumed() { return armed_operation == 0; }

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
  if (auto status = take_status(sirius::test::scoped_cuda_status_fault::wait_event);
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
