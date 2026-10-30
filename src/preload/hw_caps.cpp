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

#include "preload/hw_caps.hpp"

#include <cuda_runtime.h>

#include <cucascade/memory/memory_space.hpp>

namespace sirius::preload {

namespace {

/// Read one CUDA runtime device attribute; 0 when the running driver cannot answer. Takes the
/// device explicitly, so no context is created and the calling thread's current device is left
/// untouched.
int query_device_attribute(cudaDeviceAttr attribute, int device) noexcept
{
  int value = 0;
  if (cudaDeviceGetAttribute(&value, attribute, device) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error the failed query left behind
    return 0;
  }
  return value;
}

}  // namespace

bool gpu_has_c2c(memory::topology_index const& topology, int device) noexcept
{
  (void)topology;  // C2C is read straight from the CUDA driver attributes of the device ordinal.
  if (device < 0) { return false; }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device >= count) {
    cudaGetLastError();
    return false;
  }
  return c2c_from_attributes(
    query_device_attribute(cudaDevAttrHostNativeAtomicSupported, device),
    query_device_attribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
}

bool disk_uses_gds(cucascade::memory::memory_space const& disk) noexcept
{
  // The pinned cuCascade has no GDS / io-context query: every DISK lane is treated as non-GDS.
  (void)disk;
  return false;
}

}  // namespace sirius::preload
