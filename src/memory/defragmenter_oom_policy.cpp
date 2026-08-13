
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

#include <cstdlib>

namespace sirius {
namespace memory {

namespace {

/// Multiple of the failed allocation that must be sitting reserved-but-unused
/// before a trim is judged worthwhile.
///
/// A hard-coded 10 is a high bar: a 380 MB allocation needs 3.8 GB of
/// free-but-reserved memory before the pool counts as fragmented, so under real
/// memory pressure the policy rethrows without ever trimming. Tunable via
/// SIRIUS_OOM_TRIM_FACTOR so the threshold can be swept without a rebuild;
/// 0 disables the check and always attempts the trim.
double oom_trim_factor()
{
  static const double factor = [] {
    const char* env = std::getenv("SIRIUS_OOM_TRIM_FACTOR");
    if (env == nullptr) { return 10.0; }
    char* end      = nullptr;
    const double v = std::strtod(env, &end);
    if (end == env || v < 0.0) { return 10.0; }
    return v;
  }();
  return factor;
}

/// Pool occupancy at the moment of an OOM, for the diagnostic below.
struct pool_state {
  bool ok{false};
  std::uint64_t reserved{};
  std::uint64_t used{};
  [[nodiscard]] std::uint64_t free_reserved() const { return reserved > used ? reserved - used : 0; }
};

pool_state read_pool_state(cudaMemPool_t pool)
{
  pool_state s;
  if (!pool) { return s; }
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &s.reserved) !=
      cudaSuccess) {
    return s;
  }
  if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &s.used) != cudaSuccess) {
    return s;
  }
  s.ok = true;
  return s;
}

/**
 * @brief Returns true if pool appears fragmented for an allocation of bytes.
 *
 * Fragmentation is detected by comparing `cudaMemPoolAttrReservedMemCurrent`
 * (total bytes held by the pool from the driver) against
 * `cudaMemPoolAttrUsedMemCurrent` (bytes actively in use by live allocations).
 * If the gap between the two is at least `factor x bytes` the pool holds enough
 * free, fragmented blocks that a trim may consolidate into a contiguous region.
 */
bool is_pool_fragmented(const pool_state& s, std::size_t bytes)
{
  if (!s.ok) { return false; }
  const double factor = oom_trim_factor();
  if (factor <= 0.0) { return true; }  // check disabled: always try the trim
  return s.free_reserved() >= static_cast<std::uint64_t>(factor * static_cast<double>(bytes));
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
    SIRIUS_LOG_DEBUG("[oom_defrag] no trim: error_kind={} (not ALLOCATION_FAILED), bytes={}",
                     static_cast<int>(oom_ex->error_kind),
                     bytes);
    std::rethrow_exception(eptr);
  }

  if (!oom_ex->pool_handle) {
    SIRIUS_LOG_DEBUG("[oom_defrag] no trim: allocation carries no pool handle, bytes={}", bytes);
    std::rethrow_exception(eptr);
  }

  // Read once and reuse, so the decision and the diagnostic describe the same
  // observation rather than two reads either side of a racing allocation.
  const auto state = read_pool_state(oom_ex->pool_handle);

  // If the pool doesn't look fragmented, trimming won't help — bail out.
  if (!is_pool_fragmented(state, bytes)) {
    SIRIUS_LOG_DEBUG(
      "[oom_defrag] no trim: pool not fragmented enough for {}B - reserved={}B used={}B "
      "free_reserved={}B, need >= {}x ({}B){}",
      bytes,
      state.reserved,
      state.used,
      state.free_reserved(),
      oom_trim_factor(),
      static_cast<std::uint64_t>(oom_trim_factor() * static_cast<double>(bytes)),
      state.ok ? "" : " [pool attributes unreadable]");
    std::rethrow_exception(eptr);
  }

  // Release all free, fragmented blocks back to the driver so it can reassemble
  // them into larger contiguous regions for the retry.
  SIRIUS_LOG_DEBUG("[oom_defrag] trimming for {}B: reserved={}B used={}B free_reserved={}B",
                   bytes,
                   state.reserved,
                   state.used,
                   state.free_reserved());
  cudaMemPoolTrimTo(oom_ex->pool_handle, /*minBytesToKeep=*/0);
  cudaDeviceSynchronize();  // Ensure that the trim operation is complete before retrying.

  const auto after = read_pool_state(oom_ex->pool_handle);
  try {
    void* p = retry_function(bytes, stream);
    SIRIUS_LOG_DEBUG("[oom_defrag] trim SUCCEEDED for {}B: reserved {}B -> {}B (released {}B)",
                     bytes,
                     state.reserved,
                     after.reserved,
                     state.reserved > after.reserved ? state.reserved - after.reserved : 0);
    return p;
  } catch (...) {
    // Retry failed — surface the original allocation failure to the caller.
    SIRIUS_LOG_DEBUG("[oom_defrag] trim FAILED to satisfy {}B: reserved {}B -> {}B",
                     bytes,
                     state.reserved,
                     after.reserved);
    std::rethrow_exception(eptr);
  }
}

}  // namespace memory
}  // namespace sirius
