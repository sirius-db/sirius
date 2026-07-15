
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

#include "cuda_runtime_api.h"
#include "log/logging.hpp"

#include <cuda_runtime.h>

#include <cucascade/memory/error.hpp>

namespace sirius {
namespace memory {

namespace {

/**
 * @brief Returns true if pool appears fragmented for an allocation of bytes.
 *
 * Fragmentation is detected by comparing `cudaMemPoolAttrReservedMemCurrent`
 * (total bytes held by the pool from the driver) against
 * `cudaMemPoolAttrUsedMemCurrent` (bytes actively in use by live allocations).
 * If the gap between the two is at least bytes the pool holds enough free,
 * fragmented blocks that a trim may consolidate into a single contiguous region.
 */
bool is_pool_fragmented(cudaMemPool_t pool, std::size_t bytes)
{
  if (!pool) return false;

  std::uint64_t reserved{};
  std::uint64_t used{};

  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) != cudaSuccess) {
    return false;
  }
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) != cudaSuccess) {
    return false;
  }

  // There is at least `10 X bytes` worth of reserved-but-unused (fragmented) memory.
  return reserved > used && (reserved - used) >= 10 * bytes;
}

}  // namespace

std::string defragmenter_oom_policy::get_policy_name() const noexcept { return "defragmenter"; }

void* defragmenter_oom_policy::do_handle_oom(std::size_t bytes,
                                             rmm::cuda_stream_view stream,
                                             std::exception_ptr eptr,
                                             RetryFunc retry_function)
{
  // Only cucascade_out_of_memory carries a pool handle we can inspect and trim.
  // Any other exception type is rethrown immediately.
  cucascade::memory::cucascade_out_of_memory* oom_ex{};
  try {
    std::rethrow_exception(eptr);
  } catch (cucascade::memory::cucascade_out_of_memory& ex) {
    oom_ex = &ex;
  } catch (...) {
    std::rethrow_exception(eptr);
  }

  if (oom_ex->error_kind != cucascade::memory::MemoryError::ALLOCATION_FAILED) {
    // The requested allocation size exceeds the pool's maximum allocation size, so no amount
    // of trimming will help. Surface the error to the caller immediately.
    std::rethrow_exception(eptr);
  }

  // If the pool doesn't look fragmented, trimming won't help — bail out.
  if (!oom_ex->pool_handle || !is_pool_fragmented(oom_ex->pool_handle, bytes)) {
    std::rethrow_exception(eptr);
  }

  // Release all free, fragmented blocks back to the driver so it can reassemble
  // them into larger contiguous regions for the retry.
  cudaMemPoolTrimTo(oom_ex->pool_handle, /*minBytesToKeep=*/0);
  if (auto const err = cudaDeviceSynchronize(); err != cudaSuccess) {
    SIRIUS_LOG_ERROR("defragmenter_oom_policy: cudaDeviceSynchronize failed after trim: {}",
                     cudaGetErrorString(err));
    std::rethrow_exception(eptr);
  }

  try {
    return retry_function(bytes, stream);
  } catch (...) {
    // Retry failed — surface the original allocation failure to the caller.
    std::rethrow_exception(eptr);
  }
}

}  // namespace memory
}  // namespace sirius
