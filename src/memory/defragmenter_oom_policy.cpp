
/*
 * Copyright 2025, Sirius Contributors.
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

#include "memory/defragmenter_oom_policy.hpp"

#include <rmm/error.hpp>

#include <cuda_runtime.h>

namespace sirius {
namespace memory {

namespace {

/**
 * @brief Returns true if the default CUDA memory pool for the current device appears
 * fragmented, meaning enough memory is reserved but not in use to satisfy @p bytes.
 *
 * Fragmentation is detected by comparing `cudaMemPoolAttrReservedMemCurrent`
 * (total bytes held by the pool from the driver) against
 * `cudaMemPoolAttrUsedMemCurrent` (bytes actively in use by live allocations).
 * If the gap between the two is at least @p bytes the pool holds enough free,
 * fragmented blocks that a trim may consolidate into a single contiguous region.
 */
bool is_pool_fragmented(std::size_t bytes)
{
  int device{};
  if (cudaGetDevice(&device) != cudaSuccess) { return false; }

  cudaMemPool_t pool{};
  if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) { return false; }

  std::uint64_t reserved{};
  std::uint64_t used{};

  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) != cudaSuccess) {
    return false;
  }
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) != cudaSuccess) {
    return false;
  }

  // There is at least `bytes` worth of reserved-but-unused (fragmented) memory.
  return reserved > used && (reserved - used) >= bytes;
}

/**
 * @brief Trims the default CUDA memory pool for the current device, releasing all
 * reserved-but-unused memory back to the driver.
 */
void trim_pool()
{
  int device{};
  if (cudaGetDevice(&device) != cudaSuccess) { return; }

  cudaMemPool_t pool{};
  if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) { return; }

  // Keep zero bytes — release all free blocks to the driver so the driver can
  // reassemble them into larger contiguous regions for the retry.
  cudaMemPoolTrimTo(pool, /*minBytesToKeep=*/0);
}

}  // namespace

std::string defragmenter_oom_policy::get_policy_name() const noexcept { return "defragmenter"; }

void* defragmenter_oom_policy::do_handle_oom(std::size_t bytes,
                                             rmm::cuda_stream_view stream,
                                             std::exception_ptr eptr,
                                             RetryFunc retry_function)
{
  // Only attempt defragmentation when the pool holds enough free-but-fragmented
  // memory to satisfy the request. If the GPU simply doesn't have enough memory,
  // trimming won't help and we rethrow immediately.
  if (!is_pool_fragmented(bytes)) { std::rethrow_exception(eptr); }

  trim_pool();

  try {
    return retry_function(bytes, stream);
  } catch (...) {
    // Retry failed — surface the original allocation failure to the caller.
    std::rethrow_exception(eptr);
  }
}

}  // namespace memory
}  // namespace sirius
