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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius {
namespace op {

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

  /// @brief Inform the consumer which GPUs the query was admitted to (sorted, deduped). Set at
  /// convert time; consumer policy must choose placement within this set. Its
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

  /// @brief Install the placement of the partitioned data this operator receives before
  /// publication. Setting an equal placement again is a
  /// no-op (a consumer fed by two sibling partitions is reached once per sibling).
  /// @throws sirius::internal_exception if a different placement was already installed.
  void set_placement(std::shared_ptr<const partition_placement> placement);

  /// The installed placement. PARTITION must install it before partitioned data is emitted;
  /// standalone operator tests must install it explicitly with set_placement().
  /// @throws sirius::internal_exception if no placement has been installed.
  [[nodiscard]] std::shared_ptr<const partition_placement> placement() const;

 private:
  //! GPUs the query was admitted to (set at convert time via set_active_gpu_ids).
  std::vector<int> _active_gpu_ids;

  //! Placement of the partitioned data this operator receives (see set_placement). Written once by
  //! the upstream PARTITION and read by task-creation threads without the operator lock.
  std::atomic<std::shared_ptr<const partition_placement>> _placement;
};

}  // namespace op
}  // namespace sirius
