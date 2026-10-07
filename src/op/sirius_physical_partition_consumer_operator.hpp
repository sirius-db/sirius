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

#include "op/partition_placement.hpp"
#include "op/sirius_physical_operator.hpp"
#include "sirius_config.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius {
namespace op {

/// What the upstream PARTITION operator measures/knows and forwards to its downstream consumer so
/// the consumer can decide how many partitions to produce (and whether to broadcast).
struct partition_sizing_input {
  uint64_t total_bytes;  ///< Bytes waiting on the sizing partition's input port.
  bool is_build_side;    ///< The sizing partition drives the build side (only the build side can
                         ///< drive broadcast / build-probe).
  bool build_foldable;   ///< A downstream build-side CONCAT can concat_all to a single batch.
  /// Bytes waiting on both sibling partitions' input ports (equal to `total_bytes` when the
  /// partition has no sibling). A consumer whose task holds both join inputs at once sizes from
  /// this rather than `total_bytes`; one whose task holds only the sizing side uses `total_bytes`.
  uint64_t combined_total_bytes;
};

/// The partitioning decision returned by a consumer's get_partition_strategy. `num_partitions` is
/// applied to the partition operator(s); `broadcast`/`build_probe` are reported back so the
/// partition can configure its own wiring (e.g. enabling build-side concat_all). `placement` says
/// which GPU each partition runs on; the partition hands it to every operator that emits this
/// exchange's partitioned data.
struct partition_strategy {
  /// @throws std::invalid_argument if `num_partitions < 1` or the placement covers a different
  ///         number of partitions.
  partition_strategy(int num_partitions_p,
                     bool broadcast_p,
                     bool build_probe_p,
                     partition_placement placement_p)
    : num_partitions(num_partitions_p),
      broadcast(broadcast_p),
      build_probe(build_probe_p),
      placement(std::move(placement_p))
  {
    if (num_partitions < 1) {
      throw std::invalid_argument("partition_strategy: num_partitions (" +
                                  std::to_string(num_partitions) + ") must be at least 1");
    }
    if (placement.num_partitions() != static_cast<std::size_t>(num_partitions)) {
      throw std::invalid_argument(
        "partition_strategy: placement covers " + std::to_string(placement.num_partitions()) +
        " partitions but num_partitions is " + std::to_string(num_partitions));
    }
  }

  int num_partitions;
  bool broadcast;
  bool build_probe;
  partition_placement placement;
};

/// Multi-GPU partition floor derived purely from the GPU count: below the small-table threshold a
/// consumer stays on one partition (single GPU); at/above it we force one partition per GPU so
/// every GPU sees work.
[[nodiscard]] constexpr int partition_min_num_partitions(int num_gpus)
{
  return num_gpus > 1 ? num_gpus : 1;
}

/// Small-table byte threshold derived from the GPU count (see PARTITION_SMALL_TABLE_BYTES_PER_GPU).
[[nodiscard]] constexpr uint64_t partition_small_table_bytes(int num_gpus)
{
  return static_cast<uint64_t>(num_gpus) * config::PARTITION_SMALL_TABLE_BYTES_PER_GPU;
}

/// The "natural" partition count for `total_bytes`: ceil(total_bytes / hash_partition_bytes),
/// floored to `partition_min_num_partitions(num_gpus)` once the input clears the small-table
/// threshold. Pure counterpart of the old sirius_physical_partition::determine_num_partitions
/// arithmetic.
[[nodiscard]] inline int natural_num_partitions(uint64_t total_bytes,
                                                uint64_t hash_partition_bytes,
                                                int num_gpus)
{
  if (hash_partition_bytes == 0) {
    throw std::invalid_argument("hash_partition_bytes must be greater than zero");
  }
  int num_partitions =
    static_cast<int>(std::max(uint64_t{1},
                              total_bytes / hash_partition_bytes +
                                static_cast<uint64_t>(total_bytes % hash_partition_bytes != 0)));
  int const min_parts = partition_min_num_partitions(num_gpus);
  if (min_parts > 1 && total_bytes >= partition_small_table_bytes(num_gpus)) {
    num_partitions = std::max(num_partitions, min_parts);
  }
  return num_partitions;
}

//! sirius_physical_partition_consumer_operator is an interface for operators
//! that can consume partitioned data batches
class sirius_physical_partition_consumer_operator : public sirius_physical_operator {
 public:
  sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType type,
                                              duckdb::vector<sirius::logical_type> types,
                                              std::size_t estimated_cardinality)
    : sirius_physical_operator(type, std::move(types), estimated_cardinality)
  {
  }

  virtual ~sirius_physical_partition_consumer_operator();

  //! Push a data batch to a specific port with partition information
  //! @param port_id The port identifier
  //! @param batch The data batch to push
  //! @param partition_idx The partition index
  virtual void push_data_batch_partitioned(std::string_view port_id,
                                           std::shared_ptr<::cucascade::data_batch> batch,
                                           std::size_t partition_idx);

  /// @brief Decide how the upstream PARTITION operator should partition its input for this
  /// consumer, given what the partition measured (`in`). Implementations own the count/broadcast
  /// decision, update any of their own execution state (e.g. hash-join BUILD_PROBE mode), and
  /// pre-size their own input repositories under their own lock. The base default throws — only
  /// consumers that actually drive a partition count (HASH_JOIN, NESTED_LOOP_JOIN, MERGE_GROUP_BY)
  /// override it; batch-only consumers such as CONCAT never appear as a partition's downstream
  /// sizing consumer.
  virtual partition_strategy get_partition_strategy(const partition_sizing_input& in);

  /// @brief Inform the consumer which GPUs the query was admitted to (sorted, deduped). Set at
  /// convert time; a placement chosen in get_partition_strategy must stay within this set. Its
  /// size drives the multi-GPU partition floor and (for joins) BUILD_PROBE / broadcast admission.
  /// Empty only for operators built without an engine (unit tests).
  void set_active_gpu_ids(std::vector<int> active_gpu_ids)
  {
    _active_gpu_ids = std::move(active_gpu_ids);
  }

  [[nodiscard]] std::vector<int> const& active_gpu_ids() const noexcept { return _active_gpu_ids; }

  /// Number of GPUs the query runs on; 1 when no GPU list was set.
  [[nodiscard]] int num_gpus() const noexcept
  {
    return std::max(1, static_cast<int>(_active_gpu_ids.size()));
  }

  /// @brief Install the placement of the partitioned data this operator receives. Called by the
  /// upstream PARTITION once the partition count is decided. Setting an equal placement again is a
  /// no-op (a consumer fed by two sibling partitions is reached once per sibling).
  /// @throws sirius::internal_exception if a different placement was already installed.
  void set_placement(std::shared_ptr<const partition_placement> placement);

  /// The installed placement. PARTITION must install it before partitioned data is emitted;
  /// standalone operator tests must install it explicitly with set_placement().
  /// @throws sirius::internal_exception if no placement has been installed.
  [[nodiscard]] std::shared_ptr<const partition_placement> placement() const;

 protected:
  //! Target size (bytes) per hash partition — the natural-count divisor. Set from operator_params
  //! at construction (a config option, not a runtime setter).
  uint64_t _hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES;

 private:
  //! GPUs the query was admitted to (set at convert time via set_active_gpu_ids).
  std::vector<int> _active_gpu_ids;

  //! Placement of the partitioned data this operator receives (see set_placement). Written once by
  //! the upstream PARTITION and read by task-creation threads without the operator lock.
  std::atomic<std::shared_ptr<const partition_placement>> _placement;
};

}  // namespace op
}  // namespace sirius
