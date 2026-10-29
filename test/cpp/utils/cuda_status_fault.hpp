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

#pragma once

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <cstddef>

namespace cucascade::memory {
class fixed_size_host_memory_resource;
}

namespace sirius::test {
// One-shot CUDA failure, armed only after test setup. Readiness faults target
// the calling thread; acquisition faults target the next worker acquisition.
// These wrappers are linked into the isolated test executable, never production.
class scoped_cuda_status_fault {
 public:
  // acquire_stream targets the next pool acquisition on any thread, allowing
  // a prefetch worker to observe a typed failure without poisoning the device.
  enum operation {
    wait_event          = 1,
    synchronize_device  = 2,
    acquire_stream      = 3,
    copy_device_to_host = 4,
    copy_host_to_device = 5,
    synchronize_stream  = 6,
    copy_batch          = 7
  };
  scoped_cuda_status_fault(operation op,
                           cudaError_t status,
                           std::size_t skip             = 0,
                           cudaStream_t matching_stream = nullptr);
  ~scoped_cuda_status_fault();
  scoped_cuda_status_fault(scoped_cuda_status_fault const&)            = delete;
  scoped_cuda_status_fault& operator=(scoped_cuda_status_fault const&) = delete;
  bool consumed();

 private:
  operation operation_;
};

struct cuda_transfer_observation {
  std::size_t synchronizations              = 0;
  std::size_t batch_submissions             = 0;
  std::size_t partial_batch_submissions     = 0;
  std::size_t synchronizations_with_staging = 0;
};

// Observe only this thread's calls between setup and readback/teardown. HOST bytes above the
// resource's initial allocation count show that staging is still retained when each join begins.
class scoped_cuda_transfer_observer {
 public:
  explicit scoped_cuda_transfer_observer(
    cucascade::memory::fixed_size_host_memory_resource const& staging_resource);
  ~scoped_cuda_transfer_observer();
  scoped_cuda_transfer_observer(scoped_cuda_transfer_observer const&)            = delete;
  scoped_cuda_transfer_observer& operator=(scoped_cuda_transfer_observer const&) = delete;
  [[nodiscard]] cuda_transfer_observation snapshot() const noexcept;
};

// One-shot cudaMemPoolGetAccess failure for one pool and accessor, from any thread. The first
// @p skip matching queries succeed.
class scoped_pool_access_fault {
 public:
  scoped_pool_access_fault(cudaMemPool_t pool, int accessor, std::size_t skip, cudaError_t status);
  ~scoped_pool_access_fault();
  scoped_pool_access_fault(scoped_pool_access_fault const&)            = delete;
  scoped_pool_access_fault& operator=(scoped_pool_access_fault const&) = delete;
  bool consumed() const;
};
}  // namespace sirius::test
