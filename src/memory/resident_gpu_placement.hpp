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
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace sirius::memory {

/// Picks the GPU for a task whose input is already in GPU memory, when the query may be admitted
/// to only some of the GPUs. Data on an admitted GPU stays there. Data on an excluded GPU takes
/// the next GPU of a round-robin walk over the admitted set, so the moved tasks spread evenly.
/// The walk is not narrowed to the excluded GPU's NUMA node: that would put the excluded GPU's
/// whole share on the one admitted GPU beside it. An empty `admitted` admits every GPU.
class resident_gpu_placement {
 public:
  /// For data on one GPU, such as a GPU-tier pinned chunk.
  [[nodiscard]] int gpu_for_home(int home, std::span<int const> admitted)
  {
    if (admitted.empty() || is_admitted(home, admitted)) { return home; }
    return next_admitted(admitted);
  }

  /// For data spread over GPUs, given as (GPU, bytes) pairs: the admitted GPU with the most
  /// bytes, the first one in range order on a tie. An empty range gives no GPU.
  template <typename BytesPerGpu>
  [[nodiscard]] std::optional<int> gpu_for_bytes(BytesPerGpu const& bytes_per_gpu,
                                                 std::span<int const> admitted)
  {
    std::optional<int> best;
    std::size_t best_bytes = 0;
    bool any_data          = false;
    for (auto const& [gpu, bytes] : bytes_per_gpu) {
      any_data = true;
      if (!admitted.empty() && !is_admitted(gpu, admitted)) { continue; }
      if (!best.has_value() || bytes > best_bytes) {
        best       = gpu;
        best_bytes = bytes;
      }
    }
    if (best.has_value() || !any_data) { return best; }
    return next_admitted(admitted);
  }

 private:
  [[nodiscard]] static bool is_admitted(int gpu, std::span<int const> admitted)
  {
    return std::find(admitted.begin(), admitted.end(), gpu) != admitted.end();
  }

  /// `admitted` is not empty.
  [[nodiscard]] int next_admitted(std::span<int const> admitted)
  {
    std::vector<int> key(admitted.begin(), admitted.end());
    std::lock_guard lock(_mutex);
    auto& cursor = _cursors.try_emplace(std::move(key), 0).first->second;
    return admitted[cursor++ % admitted.size()];
  }

  std::mutex _mutex;
  /// One cursor per admitted set, so queries on different sets do not disturb each other.
  std::map<std::vector<int>, std::uint64_t> _cursors;
};

}  // namespace sirius::memory
