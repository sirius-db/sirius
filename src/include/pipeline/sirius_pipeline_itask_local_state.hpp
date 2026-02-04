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

#include "parallel/task.hpp"

#include <cucascade/memory/memory_reservation.hpp>

#include <memory>

namespace sirius {
namespace pipeline {

/**
 * @brief Interface for pipeline task local states that manage memory reservations.
 *
 * This class extends itask_local_state to provide memory reservation management
 * capabilities for pipeline tasks. It serves as a common base for both GPU pipeline
 * tasks and DuckDB scan tasks that need to manage memory reservations.
 */
// WSM TODO: consider merging this with itask_local_state
class sirius_pipeline_itask_local_state : public parallel::itask_local_state {
 public:
  /**
   * @brief Destructor for proper cleanup of derived classes.
   */
  ~sirius_pipeline_itask_local_state() override = default;

  /**
   * @brief Release and return the memory reservation held by this task.
   *
   * This method transfers ownership of the reservation to the caller.
   * After calling this method, the task no longer holds a reservation.
   *
   * @return std::unique_ptr<cucascade::memory::reservation> The released reservation,
   *         or nullptr if no reservation was held.
   */
  std::unique_ptr<cucascade::memory::reservation> release_reservation()
  {
    return std::move(_reservation);
  }

  /**
   * @brief Set a memory reservation for this task.
   *
   * This method transfers ownership of the provided reservation to the task.
   * Any previously held reservation will be released.
   *
   * @param res The memory reservation to set (ownership transferred to the task)
   */
  void set_reservation(std::unique_ptr<cucascade::memory::reservation> res)
  {
    _reservation = std::move(res);
  }

 protected:
  /**
   * @brief Protected default constructor.
   *
   * This constructor is protected to ensure the class can only be instantiated
   * through derived classes.
   */
  sirius_pipeline_itask_local_state() = default;

  std::unique_ptr<cucascade::memory::reservation>
    _reservation;  ///< Memory reservation for GPU resources
};

}  // namespace pipeline
}  // namespace sirius
