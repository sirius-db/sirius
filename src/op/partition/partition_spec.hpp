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

#include <cudf/types.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op {

/// How a partitioned sink routes rows across its output streams. Destination nodes are the
/// wrapper's routing table — the sink stays oblivious.
enum class partition_mode {
  hash,       ///< GPU-hash-partition by key_columns; key_columns must be non-empty.
  broadcast,  ///< Replicate every batch to all N outputs; key_columns must be empty.
  single,     ///< Forward every batch to destination zero.
  local,      ///< Forward to the first destination placed on the resident device.
};

struct partition_spec {
  partition_mode mode = partition_mode::hash;

  /// Hashed to pick a destination. Must be non-empty for hash mode; must be empty for broadcast.
  std::vector<int> key_columns;

  /// Per-key cast before hashing (e.g. INT32 vs INT64). Empty = hash as-is. Ignored in broadcast.
  std::vector<cudf::data_type> key_cast_types;

  partition_spec() = default;

  /// Hash-mode shorthand: partition_spec({col}, {cast}) without naming the mode.
  partition_spec(std::vector<int> keys, std::vector<cudf::data_type> casts = {})
    : key_columns(std::move(keys)), key_cast_types(std::move(casts))
  {
  }
};

/// Resolved routing, immutable before any task claims input. Key selection lives in partition_spec.
struct partition_layout {
  partition_mode mode;
  std::shared_ptr<partition_placement const> placement;

  partition_layout(partition_mode mode, partition_placement placement)
    : partition_layout(mode, std::make_shared<partition_placement const>(std::move(placement)))
  {
  }
  partition_layout(partition_mode mode, std::shared_ptr<partition_placement const> placement)
    : mode(mode), placement(std::move(placement))
  {
    if (!this->placement) { throw std::invalid_argument("partition layout requires placement"); }
  }
};

}  // namespace sirius::op
