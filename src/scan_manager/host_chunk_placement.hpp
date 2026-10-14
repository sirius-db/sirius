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

#include "memory/topology_index.hpp"

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Decides which GPU a host-resident cached scan batch is uploaded to and run on.
 *
 * The decision is made once, when the scan manager queues the split, and stamped on it as the
 * producer device preference: the task creator honours the stamp when it creates the scan task,
 * and the memory prefetcher converts the batch into that GPU's space when it uploads it early.
 * Deciding in two places -- the task creator's own NUMA round-robin at task creation and an
 * independent round-robin in the prefetcher -- lets a batch be uploaded to one GPU while its
 * task is sent to another.
 *
 * Policy, the same as the task creator's locality rule: round-robin over the admitted GPUs on
 * the batch's NUMA node (a host space's device id is its NUMA node, the key topology_index
 * groups GPUs under, including -1 for "unknown"), or over every admitted GPU when none is
 * local. One cursor per node, shared by every scan of the query, so small tables do not all
 * start on a node's first GPU.
 */
class host_chunk_placement {
 public:
  /// @param admitted_gpu_ids  The query's admitted GPUs, in admission order.
  /// @param topology          GPU<->NUMA map; null means no locality, round-robin over all.
  host_chunk_placement(std::vector<int> admitted_gpu_ids,
                       std::shared_ptr<const sirius::memory::topology_index> topology)
    : _admitted(std::move(admitted_gpu_ids)), _topology(std::move(topology))
  {
  }

  [[nodiscard]] std::span<const int> admitted_gpu_ids() const noexcept { return _admitted; }

  /// Devices a batch resident in @p source may go to, most preferred first; empty when nothing
  /// is admitted. Advances the node's round-robin cursor.
  [[nodiscard]] std::vector<int> candidates(const cucascade::memory::memory_space* source)
  {
    std::vector<int> out;
    int group = k_any_group;
    if (_topology && source && source->get_tier() == cucascade::memory::Tier::HOST) {
      auto const local = _topology->gpus_of(source->get_device_id());
      for (int const id : _admitted) {
        if (std::ranges::find(local, id) != local.end()) { out.push_back(id); }
      }
      if (!out.empty()) { group = source->get_device_id(); }
    }
    if (out.empty()) { out = _admitted; }
    if (out.size() > 1) {
      std::size_t start = 0;
      {
        std::lock_guard lk{_mutex};
        start = _cursors[group]++ % out.size();
      }
      std::rotate(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(start), out.end());
    }
    return out;
  }

  /// The device for a batch resident in @p source, or -1 when nothing is admitted.
  [[nodiscard]] int choose(const cucascade::memory::memory_space* source)
  {
    auto const out = candidates(source);
    return out.empty() ? -1 : out.front();
  }

 private:
  /// Cursor key for batches with no NUMA-local admitted GPU; distinct from every node id,
  /// including -1.
  static constexpr int k_any_group = std::numeric_limits<int>::min();

  std::vector<int> _admitted;
  std::shared_ptr<const sirius::memory::topology_index> _topology;
  std::mutex _mutex;
  std::unordered_map<int, std::size_t> _cursors;
};

}  // namespace sirius::scan_manager
