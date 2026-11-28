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
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace sirius::memory {

/// Independent round-robin walks for host NUMA domains, one per (domain, GPU list) pair. A
/// counter shared across domains aliases alternating domains onto the same GPU within each
/// domain instead of rotating. A counter shared across lists of different lengths does the same
/// to queries admitted to different GPU subsets: with {0,1} and {0,1,2} alternating on one
/// domain, the first list would only ever see even counter values and return GPU 0.
class numa_affinity {
 public:
  std::optional<int> next_gpu(int numa_node, std::span<int const> gpus)
  {
    if (gpus.empty()) { return std::nullopt; }
    walk_key key{numa_node, std::vector<int>(gpus.begin(), gpus.end())};
    std::lock_guard lock(_mutex);
    auto& cursor = _cursors.try_emplace(std::move(key), 0).first->second;
    return gpus[cursor++ % gpus.size()];
  }

 private:
  using walk_key = std::pair<int, std::vector<int>>;

  std::mutex _mutex;
  /// One entry per distinct list seen for a domain. Admission keeps a prefix of the sorted GPU
  /// list, so a domain sees at most as many lists as it has GPUs.
  std::map<walk_key, std::uint64_t> _cursors;
};

/// The GPUs a host-data task may be routed to when its query runs on `admitted`: the members of
/// `local` (the GPUs on the data's NUMA node) that were admitted, in `local` order. Walking every
/// local GPU instead sends some tasks to a GPU the query does not run on; each of those then has
/// to be moved, and lands on a GPU that already took its own share. The result is empty when no
/// local GPU was admitted: locality then has no GPU to name, and the task is placed like any
/// other task without a preference. An empty `admitted` means no admitted set was recorded for
/// the query, and `local` is returned whole.
[[nodiscard]] inline std::vector<int> admitted_local_gpus(std::span<int const> local,
                                                          std::span<int const> admitted)
{
  if (admitted.empty()) { return {local.begin(), local.end()}; }
  std::vector<int> gpus;
  gpus.reserve(local.size());
  for (int const gpu : local) {
    if (std::find(admitted.begin(), admitted.end(), gpu) != admitted.end()) { gpus.push_back(gpu); }
  }
  return gpus;
}

}  // namespace sirius::memory
