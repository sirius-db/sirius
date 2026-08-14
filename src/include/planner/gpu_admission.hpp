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

#include <cstddef>
#include <vector>

namespace sirius::planner {

/**
 * @brief Narrow an active-GPU list to the first @p cap entries.
 *
 * @p cap is `topology.gpus_per_query`; 0, negative (rejected at config load), or a value
 * at or above the list size all mean "no cap", so asking for more GPUs than exist yields
 * every GPU rather than failing. Taking a prefix of the sorted list is stable and
 * reproducible but not NUMA-aware — a future allocator should prefer GPUs sharing a NUMA
 * domain over ids 0..k-1.
 */
inline std::vector<int> apply_gpu_cap(std::vector<int> gpu_ids, int cap)
{
  if (cap > 0 && static_cast<std::size_t>(cap) < gpu_ids.size()) {
    gpu_ids.resize(static_cast<std::size_t>(cap));
  }
  return gpu_ids;
}

}  // namespace sirius::planner
