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

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace sirius::memory {

/** @brief Size initial host pools to 10% of NUMA RAM, capped by the host budget.
 *
 * Round down to whole pools, using the same block alignment as the host allocator.
 * A budget smaller than one pool produces zero initial pools.
 */
[[nodiscard]] inline std::size_t default_initial_host_pools(std::size_t numa_bytes,
                                                            std::size_t capacity_bytes,
                                                            std::size_t block_size,
                                                            std::size_t pool_size)
{
  constexpr auto alignment = alignof(std::max_align_t);
  constexpr auto max_size  = std::numeric_limits<std::size_t>::max();
  if (block_size == 0 || pool_size == 0 || block_size > max_size - (alignment - 1)) {
    throw std::runtime_error("memory.host: invalid block_size or pool_size");
  }
  auto const aligned_block_size = (block_size + alignment - 1) / alignment * alignment;
  if (pool_size > max_size / aligned_block_size) {
    throw std::runtime_error("memory.host: pool size in bytes overflows");
  }
  return std::min(numa_bytes / 10, capacity_bytes) / (aligned_block_size * pool_size);
}

}  // namespace sirius::memory
