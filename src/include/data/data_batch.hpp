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
#include "memory/memory_reservation.hpp"

namespace sirius {

class DataBatchView; // Forward declaration

using sirius::memory::Tier;

/**
 * @brief A data batch represents a collection of data that can be moved between different memory tiers.
 * 
 * DataBatch is the core unit of data management in the Sirius system. It wraps an IDataRepresentation
 * and provides reference counting functionality to track how many views are currently accessing the data.
 * This enables safe memory management and efficient data movement between GPU memory, host memory, 
 * and storage tiers.
 * 
 * Key characteristics:
 * - Move-only semantics (no copy constructor/assignment)
 * - Reference counting for safe shared access via DataBatchView
 * - Delegated tier management to underlying IDataRepresentation
 * - Unique batch ID for tracking and debugging purposes
 * 
 * @note This class is not thread-safe for construction/destruction, but the reference counting
 *       operations are atomic and thread-safe.
 */
class DataBatch {
public:
    /**
     * @brief Construct a new DataBatch with the given ID and data representation.
     * 
     * @param batch_id Unique identifier for this batch (obtained from DataRepositoryManager)
     * @param data Ownership of the data representation is transferred to this batch
     */
    DataBatch(uint64_t batch_id, sirius::unique_ptr<IDataRepresentation> data);
    
    /**
     * @brief Move constructor - transfers ownership of the batch and its data.
     * 
     * Moves the batch_id and data from the other batch, then resets the other batch's
     * batch_id to 0 and data pointer to nullptr.
     * 
     * @param other The batch to move from (will have batch_id set to 0 and data set to nullptr)
     */
    DataBatch(DataBatch&& other) noexcept;

    /**
     * @brief Move assignment operator - transfers ownership of the batch and its data.
     * 
     * Performs self-assignment check, then moves the batch_id and data from the other batch.
     * Resets the other batch's batch_id to 0 and data pointer to nullptr.
     * 
     * @param other The batch to move from (will have batch_id set to 0 and data set to nullptr)
     * @return DataBatch& Reference to this batch
     */
    DataBatch& operator=(DataBatch&& other) noexcept;

    /**
     * @brief Get the current memory tier where this batch's data resides.
     * 
     * @return Tier The memory tier (GPU, HOST, or STORAGE)
     */
    Tier GetCurrentTier() const;

    /**
     * @brief Get the unique identifier for this data batch.
     * 
     * @return uint64_t The batch ID assigned during construction
     */
    uint64_t GetBatchId() const;

    /**
     * @brief Atomically decrement the reference count.
     * 
     * Called when a DataBatchView is destroyed. When the reference count reaches zero,
     * this DataBatch object deletes itself.
     * Uses relaxed memory ordering as reference counting doesn't require synchronization.
     * View count == 0 means that the DataBatch can be destroyed.
     * 
     * @warning This method will delete the DataBatch object when ref count reaches zero.
     */
    void DecrementViewRefCount();

    /**
     * @brief Atomically increment the reference count.
     * 
     * Called when a new DataBatchView is created. Uses relaxed memory ordering
     * for performance as reference counting doesn't require synchronization.
     */
    void IncrementViewRefCount();

    /**
     * @brief Thread-safe method to increment pin count with tier validation.
     * 
     * Called when a batch is pinned in memory to prevent eviction. This method
     * validates that the data is in GPU tier before incrementing the pin count.
     * Uses a mutex lock for thread-safe tier checking and atomic operations.
     * Pin count == 0 means that the DataBatch can be downgraded from GPU memory.
     * 
     * @throws std::runtime_error if data is not currently in GPU tier
     */
    void IncrementPinRefCount();

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
    void DecrementPinRefCount();

    /**
     * @brief Create a DataBatchView referencing this DataBatch.
     * 
     * Casts the underlying data representation to GPUTableRepresentation and creates
     * a DataBatchView from its CUDF table view. The DataBatchView constructor will
     * handle incrementing the reference count.
     * 
     * @return sirius::unique_ptr<DataBatchView> A unique pointer to the new DataBatchView
     * @note Assumes data is already in GPU tier as GPUTableRepresentation
     */
    sirius::unique_ptr<DataBatchView> CreateView();

    /**
     * @brief Get the current view reference count.
     * 
     * Returns the number of DataBatchViews currently referencing this batch.
     * Uses relaxed memory ordering for the atomic load.
     * 
     * @return size_t The current view reference count
     * @note Thread-safe atomic operation with relaxed memory ordering
     */
    size_t GetViewCount() const;

    /**
     * @brief Get the current pin count.
     * 
     * Returns the number of active pins on this batch. A non-zero pin count
     * indicates the batch should not be evicted or moved between memory tiers.
     * Uses relaxed memory ordering for the atomic load.
     * 
     * @return size_t The current pin count
     * @note Thread-safe atomic operation with relaxed memory ordering
     */
    size_t GetPinCount() const;

    /**
     * @brief Get the underlying data representation.
     * 
     * Returns a pointer to the IDataRepresentation that holds the actual data.
     * This allows access to tier-specific operations and data access methods.
     * 
     * @return sirius::unique_ptr<IDataRepresentation> Pointer to the data representation
     */
    sirius::unique_ptr<IDataRepresentation> GetData() const;

private:
    mutable sirius::mutex mutex_;                         ///< Mutex for thread-safe access to tier checking and reference counting
    uint64_t batch_id_;                                   ///< Unique identifier for this data batch
    sirius::unique_ptr<IDataRepresentation> data_;       ///< Pointer to the actual data representation
    sirius::atomic<size_t> view_count_ = 0;               ///< Reference count for tracking views
    sirius::atomic<size_t> pin_count_ = 0;                ///< Reference count for tracking pins to prevent eviction
};

} // namespace sirius