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

#include <cucascade/data/data_batch.hpp>

#include <memory>
#include <rmm/cuda_stream_view.hpp>

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

  /**
   * @brief Executes the memory downgrade operation for this task.
   *
   * Moves the batch's data from GPU tier to HOST tier using the reservation manager
   * for HOST memory allocation.
   *
   * @param stream CUDA stream used for device memory operations
   */
  void execute(rmm::cuda_stream_view stream);
};

}  // namespace parallel
}  // namespace sirius
