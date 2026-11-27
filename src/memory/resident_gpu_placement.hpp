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

/// Picks the GPU that runs a task whose input is already in GPU memory, for a query that may be
/// admitted to only some of the GPUs.
///
/// Data on an admitted GPU stays there. Data on a GPU the query was not admitted to has to move,
/// and each such task takes the next GPU of a round-robin walk over the admitted set. The walk
/// has one cursor per admitted set and nothing else advances it, so the tasks moved off the
/// excluded GPUs spread evenly over the admitted ones. Leaving the move to the task creator's
/// clamp does not give that: its counter is shared with every unpreferred task of every query,
/// so the moved tasks can land on the same GPU again and again.
///
/// The walk covers the whole admitted set, not only the GPUs on the excluded GPU's NUMA node.
/// The data crosses to another GPU either way, and keeping to the node would put an excluded
/// GPU's whole share on the one admitted GPU next to it.
///
/// An empty `admitted` means no admitted set was recorded for the query, and every GPU counts as
/// admitted.
class resident_gpu_placement {
 public:
  /// The GPU for data that sits on `home` alone, such as a GPU-tier pinned chunk: `home` when the
  /// query was admitted to it, otherwise the next GPU of the walk over `admitted`.
  [[nodiscard]] int gpu_for_home(int home, std::span<int const> admitted)
  {
    if (admitted.empty() || is_admitted(home, admitted)) { return home; }
    return next_admitted(admitted);
  }

  /// The GPU for data spread over several GPUs. `bytes_per_gpu` is a range of (GPU, bytes held
  /// there) pairs. The result is the admitted GPU that holds the most bytes, the first such GPU
  /// in range order on a tie. When no admitted GPU holds any of the data, it is the next GPU of
  /// the walk over `admitted`. An empty range gives no GPU.
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
  /// One entry per distinct admitted set seen. Admission keeps a prefix of the sorted GPU list,
  /// so there are at most as many sets as GPUs.
  std::map<std::vector<int>, std::uint64_t> _cursors;
};

}  // namespace sirius::memory
