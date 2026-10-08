
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

#pragma once

#include <cuda/stream>

#include <cucascade/memory/oom_handling_policy.hpp>

#include <cstddef>
#include <cstdint>

namespace sirius {
namespace memory {

namespace detail {

/// Physical allocation failures since the last trim after which the free-space
/// rule is relaxed from `factor x request` to `1 x request`.
inline constexpr std::uint64_t kRelaxTrimAfterFailures = 8;

/**
 * @brief Decide whether a trim is worth attempting for a failed allocation.
 *
 * The normal rule wants `factor x bytes` of reserved-but-unused memory in the pool
 * (factor 10 by default; <= 0 always trims). That bar is high on purpose: a trim
 * costs a device-wide sync, and with little free memory it rarely helps.
 *
 * It is also a bar a physically over-committed device never clears: on SF3000
 * q3/q13 there were 1,313 and 809 real `cudaErrorMemoryAllocation`s and not one
 * trim, because the pool never held 10x the request free. Once the failures repeat
 * (@p failures_since_trim >= kRelaxTrimAfterFailures) the rule drops to "the pool
 * holds at least the request free": below that a trim cannot possibly produce a
 * block big enough, at or above it the free memory exists and only its shape is
 * wrong, which is what a trim fixes. The process-wide rate limit on trims still
 * applies, so the relaxed rule costs at most one sync per interval.
 */
[[nodiscard]] bool should_trim(std::uint64_t free_reserved,
                               std::size_t bytes,
                               double factor,
                               std::uint64_t failures_since_trim) noexcept;

}  // namespace detail

/**
 * @brief OOM policy that attempts to recover from fragmentation before giving up.
 *
 * When an allocation fails, this policy checks whether the CUDA memory pool is
 * fragmented (i.e., reserved memory significantly exceeds used memory). If so,
 * it trims the pool to release fragmented free blocks back to the driver, then
 * retries the allocation. If the retry also fails, the original exception is
 * rethrown.
 */
struct defragmenter_oom_policy final : public cucascade::memory::oom_handling_policy {
  std::string get_policy_name() const noexcept override;

 protected:
  void* do_handle_oom(std::size_t bytes,
                      ::cuda::stream_ref stream,
                      std::exception_ptr eptr,
                      RetryFunc retry_function) override;
};

}  // namespace memory
}  // namespace sirius
