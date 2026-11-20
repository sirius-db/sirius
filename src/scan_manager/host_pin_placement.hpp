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
#include <atomic>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

/// Owned by the scan manager, so successive host pins share one placement cursor.
/// GPU pins deliberately keep their per-call cursor for column-merge compatibility.
class host_pin_placement {
 public:
  /// Build a stable order from the resolved (GPU id, destination host-space id) pairs.
  /// Visit one GPU per host space in each round, skipping exhausted groups. Sorting
  /// host spaces and their GPUs makes the result independent of container iteration order.
  static std::vector<int> interleaved_devices(std::span<std::pair<int, int> const> gpu_hosts)
  {
    std::map<int, std::vector<int>> groups;
    for (auto const& [gpu, host] : gpu_hosts) {
      groups[host].push_back(gpu);
    }
    std::size_t rounds = 0;
    for (auto& [host, gpus] : groups) {
      std::sort(gpus.begin(), gpus.end());
      rounds = std::max(rounds, gpus.size());
    }
    std::vector<int> order;
    order.reserve(gpu_hosts.size());
    for (std::size_t round = 0; round < rounds; ++round) {
      for (auto const& [host, gpus] : groups) {
        if (round < gpus.size()) { order.push_back(gpus[round]); }
      }
    }
    return order;
  }

  /// Advance once for each materialized host-pin chunk, across calls and table names.
  /// Failed pins may consume positions; unpinning does not rewind the cursor.
  int next_gpu(std::span<int const> order)
  {
    if (order.empty()) { throw std::invalid_argument("host pin placement requires a GPU"); }
    auto const index = _cursor.fetch_add(1, std::memory_order_relaxed) % order.size();
    return order[index];
  }

 private:
  std::atomic<std::uint64_t> _cursor{0};
};

}  // namespace sirius::scan_manager
