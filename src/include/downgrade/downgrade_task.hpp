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

#include "memory/sirius_memory_reservation_manager.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/data_batch.hpp>

#include <memory>
#include <optional>

namespace sirius {
namespace parallel {

/**
 * @brief A plain task struct representing a unit of work in a memory downgrade operation.
 *
 * Encapsulates a single data_batch and the reservation manager needed to perform
 * a GPU-to-HOST memory tier migration. No polymorphism or task hierarchy.
 */
struct downgrade_task {
  std::shared_ptr<cucascade::data_batch> batch;
  sirius::memory::sirius_memory_reservation_manager& res_mgr;

  /// Preferred HOST memory_space device_id (NUMA node) for the downgrade target.
  /// When set, the GPU->HOST reservation request uses
  /// cucascade::memory::any_memory_space_in_tier_with_preference{Tier::HOST, *preferred_numa_node}
  /// so the batch prefers a NUMA-local host memory_space, with cross-NUMA fallback ordering
  /// provided by cucascade's strategy. When unset (nullopt), the dispatch uses the unpreferred
  /// any_memory_space_in_tier{Tier::HOST} strategy (original single-GPU default behavior).
  std::optional<int> preferred_numa_node;

  /**
   * @brief Executes the memory downgrade operation for this task.
   *
   * Moves the batch's data from GPU tier to HOST tier using the reservation manager
   * for HOST memory allocation.
   *
   * @param stream CUDA stream used for device memory operations
   * @return true if the batch was successfully downgraded, false if skipped
   */
  bool execute(rmm::cuda_stream_view stream);
};

}  // namespace parallel
}  // namespace sirius
