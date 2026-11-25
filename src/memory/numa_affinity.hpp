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
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace sirius::memory {

/// Independent round-robin walks for host NUMA domains. A shared counter aliases
/// alternating domains onto the same GPU within each domain instead of rotating.
class numa_affinity {
 public:
  std::optional<int> next_gpu(int numa_node, std::span<int const> gpus)
  {
    if (gpus.empty()) { return std::nullopt; }
    std::lock_guard lock(_mutex);
    auto& cursor = _cursors[numa_node];
    return gpus[cursor++ % gpus.size()];
  }

 private:
  std::mutex _mutex;
  std::unordered_map<int, std::uint64_t> _cursors;
};

/// The GPUs a host-data task may be routed to when its query runs on `admitted`: the members of
/// `local` (the GPUs on the data's NUMA node) that were admitted, in `local` order. Walking every
/// local GPU instead sends some tasks to a GPU the query does not run on; each of those then has
/// to be moved, and lands on a GPU that already took its own share. When no local GPU was
/// admitted, every admitted GPU is equally far from the data, so the walk covers them all. An
/// empty `admitted` means the query runs on every GPU and `local` is returned whole.
[[nodiscard]] inline std::vector<int> admitted_local_gpus(std::span<int const> local,
                                                          std::span<int const> admitted)
{
  if (admitted.empty()) { return {local.begin(), local.end()}; }
  std::vector<int> gpus;
  gpus.reserve(local.size());
  for (int const gpu : local) {
    if (std::find(admitted.begin(), admitted.end(), gpu) != admitted.end()) { gpus.push_back(gpu); }
  }
  if (gpus.empty()) { gpus.assign(admitted.begin(), admitted.end()); }
  return gpus;
}

}  // namespace sirius::memory
