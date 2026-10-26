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

#include "op/partition_placement.hpp"
#include "sirius_config.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sirius::op {

/// Consumer policy inputs. Semantic join capabilities stay here rather than in the row router
/// or the batch receiver interface. The explicit input binding supplies these measurements.
struct partition_sizing_input {
  uint64_t total_bytes;  ///< Bytes waiting on the sizing partition's input port.
  bool is_build_side;    ///< The sizing partition drives the build side (only the build side can
                         ///< drive broadcast / build-probe).
  bool build_foldable;   ///< A downstream build-side CONCAT can concat_all to a single batch.
  /// Bytes waiting on both sibling partitions' input ports (equal to `total_bytes` when the
  /// partition has no sibling). A consumer whose task holds both join inputs at once sizes from
  /// this rather than `total_bytes`; one whose task holds only the sizing side uses `total_bytes`.
  uint64_t combined_total_bytes;
  /// Rows waiting on the sizing partition's input port (projected alongside `total_bytes` when
  /// the bytes are a projection). Best effort: batches whose representation does not record a
  /// row count are counted at the bytes-per-row of the batches that do.
  uint64_t total_rows = 0;
};

/// Consumer policy decision. Consumer-specific input bindings translate this into physical
/// layouts and configure the consumer/CONCAT before the shared exchange permits input claims.
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
  auto const wanted = std::max(uint64_t{1},
                               total_bytes / hash_partition_bytes +
                                 static_cast<uint64_t>(total_bytes % hash_partition_bytes != 0));
  if (wanted > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("partition count exceeds the hash kernel limit");
  }
  int num_partitions  = static_cast<int>(wanted);
  int const min_parts = partition_min_num_partitions(num_gpus);
  if (min_parts > 1 && total_bytes >= partition_small_table_bytes(num_gpus)) {
    num_partitions = std::max(num_partitions, min_parts);
  }
  return num_partitions;
}

}  // namespace sirius::op
