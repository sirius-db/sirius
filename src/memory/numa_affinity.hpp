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

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>

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

}  // namespace sirius::memory
