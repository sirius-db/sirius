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
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

// Forward declarations -- avoid pulling in heavy headers
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
 * Generalizes the RAII mutable lock / convert / auto-release pattern. Concrete
 * implementations wrap either a data_batch or a gpu_pipeline_task.
 *
 * Implementations are responsible for:
 * - Acquiring mutable_data_batch via to_mutable() (blocking) or try_to_mutable() (non-blocking)
 * - Performing the actual data conversion via convert_to on the mutable accessor
 * - RAII automatic state restoration on all exit paths (success, failure, exception)
 */
class convertible_data {
 public:
  virtual ~convertible_data() = default;

  /**
   * @brief Convert this data unit to reside in one of the target memory spaces.
   *
   * The implementation chooses the first available target space for which a reservation
   * can be obtained. On success the underlying data representation changes to the new
   * tier. On all exit paths the mutable_data_batch RAII destructor automatically
   * restores the batch to idle state.
   *
   * @param target_spaces  Candidate memory spaces to convert into (tried in order).
   * @param stream         CUDA stream for asynchronous memory operations.
   * @param res_mgr        Reservation manager for acquiring memory in the target space.
   * @param blocking       When true, uses to_mutable() (blocks until exclusive lock acquired).
   *                       When false, uses try_to_mutable() (returns nullopt immediately if
   *                       the lock is unavailable).
   * @return A vector of bytes converted per target space index on success, or nullopt if
   *         no conversion occurred.
   * @throws std::exception on unrecoverable conversion errors (RAII handles state restore).
   */
  virtual std::optional<std::vector<std::size_t>> convert(
    const std::vector<const cucascade::memory::memory_space*>& target_spaces,
    rmm::cuda_stream_view stream,
    sirius::memory::sirius_memory_reservation_manager& res_mgr,
    bool blocking) = 0;

  /**
   * @brief Get the size in bytes of this data unit in the specified memory space.
   *
   * Returns 0 if the data does not currently reside in the given space.
   *
   * @param space The memory space to query.
   * @return The size in bytes, or 0 if not in that space.
   */
  virtual std::size_t bytes_in_space(cucascade::memory::memory_space* space) const = 0;

  /**
   * @brief Stable id of the underlying data unit, when it has one.
   *
   * Lets a caller trace an individual unit across tiers. Units that do not map to a single batch
   * (a pipeline task wrapping several) return nullopt.
   */
  virtual std::optional<std::uint64_t> data_id() const { return std::nullopt; }
};

/**
 * @brief Abstract interface for discovering convertible data units in a container.
 *
 * Providers iterate over a backing container (data_repository or inspectable_mpsc) and return
 * wrapped convertible_data instances matching a target memory space filter.
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
   * @param space             The memory space to filter by.
   * @param front_to_back     Iteration direction: true = front-to-back, false = back-to-front.
   * @param ignore_subscribed When true (default), skip batches that have been subscribed to by a
   * task.
   * @return A vector of convertible_data instances (may be empty).
   */
  virtual std::vector<std::unique_ptr<convertible_data>> get_all_convertible(
    cucascade::memory::memory_space* space, bool front_to_back, bool ignore_subscribed = true) = 0;
};

}  // namespace sirius
