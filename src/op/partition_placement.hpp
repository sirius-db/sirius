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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius {
namespace op {

/// Which GPU each partition of one exchange runs on.
///
/// The partition-sizing consumer (hash join, grouped-aggregate merge, ...) chooses it alongside the
/// partition count in `get_partition_strategy`; PARTITION distributes it to every operator that
/// emits that exchange's partitioned data, and each emitter stamps the partition's device onto its
/// `partitioned_operator_data`, which the task creator honors like any other producer preference.
/// Every task of a partition therefore lands on one GPU, which cuco hash tables require.
///
/// The devices must be a subset of the query's admitted GPUs. That set is fixed for the life of a
/// query: running an exchange on fewer GPUs is expressed as a placement over a subset
/// (`select_gpu_subset`), never by changing the admitted set under a placement whose hash tables
/// already live on its devices.
class partition_placement {
 public:
  /// Marks a partition with no device preference (the task creator places it by data locality).
  static constexpr int unpinned_device = -1;

  /// `num_partitions` partitions, none pinned. Used where no GPU list is known (engine-free tests).
  static partition_placement unpinned(std::size_t num_partitions)
  {
    if (num_partitions == 0) {
      throw std::invalid_argument("partition_placement: num_partitions must be at least 1");
    }
    return partition_placement(std::vector<int>(num_partitions, unpinned_device));
  }

  /// Partition p on `gpu_ids[p % gpu_ids.size()]`. An empty `gpu_ids` yields `unpinned`.
  static partition_placement round_robin(std::size_t num_partitions,
                                         std::vector<int> const& gpu_ids)
  {
    if (gpu_ids.empty()) { return unpinned(num_partitions); }
    if (num_partitions == 0) {
      throw std::invalid_argument("partition_placement: num_partitions must be at least 1");
    }
    std::vector<int> devices(num_partitions);
    for (std::size_t p = 0; p < num_partitions; ++p) {
      devices[p] = gpu_ids[p % gpu_ids.size()];
    }
    return partition_placement(std::move(devices));
  }

  /// One partition per GPU: partition i on `gpu_ids[i]`. An empty `gpu_ids` yields `unpinned(1)`.
  static partition_placement one_per_device(std::vector<int> const& gpu_ids)
  {
    if (gpu_ids.empty()) { return unpinned(1); }
    return partition_placement(gpu_ids);
  }

  [[nodiscard]] std::size_t num_partitions() const noexcept { return _device_per_partition.size(); }

  /// The GPU `partition` is pinned to, or nullopt when it is unpinned.
  /// @throws std::out_of_range if `partition >= num_partitions()`.
  [[nodiscard]] std::optional<int> device_for(std::size_t partition) const
  {
    if (partition >= _device_per_partition.size()) {
      throw std::out_of_range("partition_placement: partition " + std::to_string(partition) +
                              " out of range (" + std::to_string(_device_per_partition.size()) +
                              " partitions)");
    }
    int const device = _device_per_partition[partition];
    if (device == unpinned_device) { return std::nullopt; }
    return device;
  }

  /// The lowest-index partition pinned to `device_id`, or nullopt when none is.
  /// Multiple partitions may share a GPU; this lookup returns only the first match.
  /// Broadcast joins use it to route probe batches because their placement has one partition
  /// per GPU. It is not a general inverse of device_for().
  [[nodiscard]] std::optional<std::size_t> first_partition_for_device(int device_id) const noexcept
  {
    for (std::size_t p = 0; p < _device_per_partition.size(); ++p) {
      if (_device_per_partition[p] != unpinned_device && _device_per_partition[p] == device_id) {
        return p;
      }
    }
    return std::nullopt;
  }

  /// The distinct pinned GPUs, in order of first use.
  [[nodiscard]] std::vector<int> devices() const
  {
    std::vector<int> result;
    for (int const device : _device_per_partition) {
      if (device == unpinned_device) { continue; }
      if (std::find(result.begin(), result.end(), device) == result.end()) {
        result.push_back(device);
      }
    }
    return result;
  }

  [[nodiscard]] bool any_pinned() const noexcept
  {
    return std::any_of(_device_per_partition.begin(), _device_per_partition.end(), [](int device) {
      return device != unpinned_device;
    });
  }

  /// "[0->1, 1->3, 2->?]", for logs.
  [[nodiscard]] std::string to_string() const
  {
    std::string result = "[";
    for (std::size_t p = 0; p < _device_per_partition.size(); ++p) {
      if (p > 0) { result += ", "; }
      result += std::to_string(p) + "->";
      int const device = _device_per_partition[p];
      result += device == unpinned_device ? std::string{"?"} : std::to_string(device);
    }
    result += "]";
    return result;
  }

  bool operator==(partition_placement const&) const = default;

 private:
  explicit partition_placement(std::vector<int> device_per_partition)
    : _device_per_partition(std::move(device_per_partition))
  {
  }

  /// One device id per partition; `unpinned_device` when the partition has no preference.
  std::vector<int> _device_per_partition;
};

/// Choose `k` of `gpu_ids`: rotate the list by `seed % gpu_ids.size()` and take the first `k`
/// (`k` is clamped to the list size). An empty list yields an empty result.
///
/// A single-partition BUILD_PROBE join spreads across GPUs with
/// `round_robin(1, select_gpu_subset(ids, 1, operator_id))`, so several small joins in one query do
/// not all land on the first GPU.
[[nodiscard]] inline std::vector<int> select_gpu_subset(std::vector<int> const& gpu_ids,
                                                        std::size_t k,
                                                        std::size_t seed)
{
  if (gpu_ids.empty()) { return {}; }
  std::size_t const n     = gpu_ids.size();
  std::size_t const count = std::min(k, n);
  std::size_t const start = seed % n;
  std::vector<int> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    result.push_back(gpu_ids[(start + i) % n]);
  }
  return result;
}

}  // namespace op
}  // namespace sirius
