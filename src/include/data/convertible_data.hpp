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

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <memory>
#include <vector>

// Forward declarations — avoid pulling in heavy headers
namespace cucascade {
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade

namespace sirius {
namespace memory {
class sirius_memory_reservation_manager;
}  // namespace memory
}  // namespace sirius

namespace sirius {

/**
 * @brief Abstract interface for a single unit of data that can be converted between memory tiers.
 *
 * Generalizes the save-prev_state / lock-for-in_transit / convert / restore pattern used by
 * downgrade_task::execute() and batch_lock_utils::lock_or_prepare_batch(). Concrete
 * implementations wrap either a data_batch (Phase 6) or a gpu_pipeline_task (Phase 7).
 *
 * Implementations are responsible for:
 * - Acquiring any necessary locks (e.g., in_transit lock on data_batch)
 * - Performing the actual data conversion via the converter registry
 * - Restoring the original state on failure (exception safety)
 * - Releasing locks on success
 */
class convertible_data {
 public:
  virtual ~convertible_data() = default;

  /**
   * @brief Convert this data unit to reside in one of the target memory spaces.
   *
   * The implementation chooses the first available target space for which a reservation
   * can be obtained. On success the underlying data representation changes to the new
   * tier. On failure or exception the data retains its original representation and state.
   *
   * @param target_spaces  Candidate memory spaces to convert into (tried in order).
   * @param stream         CUDA stream for asynchronous memory operations.
   * @param res_mgr        Reservation manager for acquiring memory in the target space.
   * @return true if the conversion succeeded, false if no target space was available.
   * @throws std::exception on unrecoverable conversion errors (state is still restored).
   */
  virtual bool convert(const std::vector<cucascade::memory::memory_space*>& target_spaces,
                       rmm::cuda_stream_view stream,
                       sirius::memory::sirius_memory_reservation_manager& res_mgr) = 0;

  /**
   * @brief Get the size in bytes of this data unit in the specified memory space.
   *
   * Returns 0 if the data does not currently reside in the given space.
   *
   * @param space The memory space to query.
   * @return The size in bytes, or 0 if not in that space.
   */
  virtual std::size_t bytes_in_space(cucascade::memory::memory_space* space) const = 0;
};

/**
 * @brief Abstract interface for discovering convertible data units in a container.
 *
 * Providers iterate over a backing container (data_repository for Phase 6,
 * inspectable_mpsc for Phase 7) and return wrapped convertible_data instances
 * matching a target memory space filter.
 */
class convertible_data_provider {
 public:
  virtual ~convertible_data_provider() = default;

  /**
   * @brief Get the next convertible data unit matching the given memory space.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction: true = front-to-back, false = back-to-front.
   * @return A convertible_data instance, or nullptr if no matching data found.
   */
  virtual std::unique_ptr<convertible_data> get_next_convertible(
    cucascade::memory::memory_space* space, bool front_to_back) = 0;

  /**
   * @brief Get all convertible data units matching the given memory space.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction: true = front-to-back, false = back-to-front.
   * @return A vector of convertible_data instances (may be empty).
   */
  virtual std::vector<std::unique_ptr<convertible_data>> get_all_convertible(
    cucascade::memory::memory_space* space, bool front_to_back) = 0;

  /**
   * @brief Get the total byte size of all data in the given memory space.
   *
   * Sums bytes_in_space() across all data units the provider can see.
   *
   * @param space The memory space to query.
   * @return Total size in bytes.
   */
  virtual std::size_t get_bytes_in_space(cucascade::memory::memory_space* space) const = 0;
};

}  // namespace sirius
