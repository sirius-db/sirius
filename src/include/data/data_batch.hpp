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
#include <variant>
#include <memory>
#include <stdexcept>
#include <cudf/table/table.hpp>

#include "helper/helper.hpp"
#include "data/common.hpp"

namespace sirius {
namespace memory {
class memory_space;
}
}  // namespace sirius

namespace sirius {

class data_batch_view;          // Forward declarationc
class data_repository_manager;  // Forward declaration

/**
 * @brief A data batch represents a collection of data that can be moved between different memory
 * tiers.
 *
 * data_batch is the core unit of data management in the Sirius system. It wraps an
 * idata_representation and provides reference counting functionality to track how many views are
 * currently accessing the data. This enables safe memory management and efficient data movement
 * between GPU memory, host memory, and storage tiers.
 *
 * Key characteristics:
 * - Move-only semantics (no copy constructor/assignment)
 * - Reference counting for safe shared access via data_batch_view
 * - Delegated tier management to underlying idata_representation
 * - Unique batch ID for tracking and debugging purposes
 *
 * @note This class is not thread-safe for construction/destruction, but the reference counting
 *       operations are protected by an internal mutex and thread-safe.
 */
class data_batch {
 public:
  /**
   * @brief Construct a new data_batch with the given ID and data representation.
   *
   * @param batch_id Unique identifier for this batch (obtained from data_repository_manager)
   * @param data Ownership of the data representation is transferred to this batch
   */
  data_batch(uint64_t batch_id,
             data_repository_manager& data_repo_mgr,
             sirius::unique_ptr<idata_representation> data);
  data_batch(uint64_t batch_id,
             data_repository_manager& data_repo_mgr,
             sirius::unique_ptr<idata_representation> data,
             sirius::memory::memory_space& memory_space);

  /**
   * @brief Move constructor - transfers ownership of the batch and its data.
   *
   * Moves the batch_id and data from the other batch, then resets the other batch's
   * batch_id to 0 and data pointer to nullptr.
   *
   * @param other The batch to move from (will have batch_id set to 0 and data set to nullptr)
   * @throws std::runtime_error if the source batch has active views (view_count != 0)
   * @throws std::runtime_error if the source batch has active pins (pin_count != 0)
   */
  data_batch(data_batch&& other);

  /**
   * @brief Move assignment operator - transfers ownership of the batch and its data.
   *
   * Performs self-assignment check, then moves the batch_id and data from the other batch.
   * Resets the other batch's batch_id to 0 and data pointer to nullptr.
   *
   * @param other The batch to move from (will have batch_id set to 0 and data set to nullptr)
   * @return data_batch& Reference to this batch
   * @throws std::runtime_error if the source batch has active views (view_count != 0)
   * @throws std::runtime_error if the source batch has active pins (pin_count != 0)
   */
  data_batch& operator=(data_batch&& other);

  /**
   * @brief Get the current memory tier where this batch's data resides.
   *
   * @return Tier The memory tier (GPU, HOST, or STORAGE)
   */
  memory::Tier get_current_tier() const;

  /**
   * @brief Get the unique identifier for this data batch.
   *
   * @return uint64_t The batch ID assigned during construction
   */
  uint64_t get_batch_id() const;

  /**
   * @brief Decrement the view reference count (mutex-protected).
   *
   * Called when a data_batch_view is destroyed. Returns the count before decrement.
   */
  size_t decrement_view_ref_count();

  /**
   * @brief Increment the view reference count (mutex-protected).
   */
  void increment_view_ref_count();

  /**
   * @brief Thread-safe method to increment pin count with tier validation.
   *
   * Called when a batch is pinned in memory to prevent eviction. This method
   * validates that the data is in GPU tier before incrementing the pin count.
   * Uses a mutex lock for thread-safe tier checking and atomic operations.
   * Pin count == 0 means that the data_batch can be downgraded from GPU memory.
   *
   * @throws std::runtime_error if data is not currently in GPU tier
   */
  void increment_pin_ref_count();

  /**
   * @brief Thread-safe method to decrement pin count with tier validation.
   *
   * Called when a pin is released. This method validates that the data is in GPU tier
   * before decrementing the pin count. When the pin count reaches zero, the batch can be
   * considered for eviction or tier movement according to memory management policies.
   * Uses a mutex lock for thread-safe tier checking and atomic operations.
   *
   * @throws std::runtime_error if data is not currently in GPU tier
   */
  void decrement_pin_ref_count();

  /**
   * @brief Create a data_batch_view referencing this data_batch.
   *
   * Casts the underlying data representation to gpu_table_representation and creates
   * a data_batch_view from its CUDF table view. The data_batch_view constructor will
   * handle incrementing the reference count.
   *
   * @return sirius::unique_ptr<data_batch_view> A unique pointer to the new data_batch_view
   * @note Assumes data is already in GPU tier as gpu_table_representation
   */
  sirius::unique_ptr<data_batch_view> create_view();

  /**
   * @brief Get the current view reference count (mutex-protected).
   */
  size_t get_view_count() const;

  /**
   * @brief Get the current pin count (mutex-protected).
   */
  size_t get_pin_count() const;

  /**
   * @brief Get the underlying data representation.
   *
   * Returns a pointer to the idata_representation that holds the actual data.
   * This allows access to tier-specific operations and data access methods.
   *
   * @return idata_representation* Pointer to the data representation (non-owning)
   */
  idata_representation* get_data() const;

  /**
   * @brief Get the data batch holder.
   *
   * Returns a pointer to the data batch holder.
   *
   * @return data_repository_manager* Pointer to the data repository manager
   */
  data_repository_manager* get_data_repository_manager() const;

  /**
   * @brief Get the memory_space where this batch currently resides.
   */
  sirius::memory::memory_space* get_memory_space() const;

  /**
   * @brief Replace the underlying data representation.
   *        Requires no active views or pins.
   */
  void set_data(sirius::unique_ptr<idata_representation> data);

  /**
   * @brief Convert the underlying representation to the target memory_space.
   *        Requires no active views or pins.
   */
  void convert_to_memory_space(const sirius::memory::memory_space* target_memory_space,
                               rmm::cuda_stream_view stream);

  bool try_to_lock_for_downgrade()
  {
    std::lock_guard<sirius::mutex> lock(_mutex);
    if (_pin_count == 0 && !_downgrade_locked) {
      _downgrade_locked = true;
      return true;
    }
    return false;
  }

 private:
  mutable sirius::mutex
    _mutex;            ///< Mutex for thread-safe access to tier checking and reference counting
  uint64_t _batch_id;  ///< Unique identifier for this data batch
  sirius::unique_ptr<idata_representation> _data;  ///< Pointer to the actual data representation
  size_t _view_count = 0;                          ///< Reference count for tracking views
  size_t _pin_count  = 0;  ///< Reference count for tracking pins to prevent eviction
  data_repository_manager* _data_repo_mgr;      ///< Pointer to the data repository manager
  sirius::memory::memory_space* _memory_space;  ///< Memory space where the data resides
  bool _downgrade_locked = false;               ///< Whether the batch is locked for downgrade
};

}  // namespace sirius