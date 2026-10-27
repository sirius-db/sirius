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

namespace sirius::test {
// One-shot CUDA failure, armed only after test setup. Readiness faults target
// the calling thread; acquisition faults target the next worker acquisition.
// These wrappers are linked into the isolated test executable, never production.
class scoped_cuda_status_fault {
 public:
  // acquire_stream targets the next pool acquisition on any thread, allowing
  // a prefetch worker to observe a typed failure without poisoning the device.
  enum operation { wait_event = 1, synchronize_device = 2, acquire_stream = 3 };
  scoped_cuda_status_fault(operation op, cudaError_t status);
  ~scoped_cuda_status_fault();
  scoped_cuda_status_fault(scoped_cuda_status_fault const&)            = delete;
  scoped_cuda_status_fault& operator=(scoped_cuda_status_fault const&) = delete;
  bool consumed();

 private:
  operation operation_;
};
}  // namespace sirius::test
