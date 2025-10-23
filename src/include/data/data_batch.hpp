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
#include <cudf/table/table.hpp>

#include "helper/helper.hpp"
#include "data/common.hpp"
#include "memory/memory_reservation.hpp"

namespace sirius {

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
    DataBatch(uint64_t batch_id, sirius::unique_ptr<IDataRepresentation> data) 
        : batch_id_(batch_id), data_(std::move(data)) {}
    
    /**
     * @brief Move constructor - transfers ownership of the batch and its data.
     * 
     * @param other The batch to move from (will be left in a valid but unspecified state)
     */
    DataBatch(DataBatch&& other) noexcept
        : batch_id_(other.batch_id_), data_(std::move(other.data_)) {
        other.batch_id_ = 0;
        other.data_ = nullptr;
    }

    /**
     * @brief Move assignment operator - transfers ownership of the batch and its data.
     * 
     * @param other The batch to move from (will be left in a valid but unspecified state)
     * @return DataBatch& Reference to this batch
     */
    DataBatch& operator=(DataBatch&& other) noexcept {
        if (this != &other) {
            batch_id_ = other.batch_id_;
            data_ = std::move(other.data_);
            other.batch_id_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the current memory tier where this batch's data resides.
     * 
     * @return Tier The memory tier (GPU, HOST, or STORAGE)
     */
    Tier GetCurrentTier() const {
        return data_->getCurrentTier();
    }

    /**
     * @brief Get the unique identifier for this data batch.
     * 
     * @return uint64_t The batch ID assigned during construction
     */
    uint64_t GetBatchId() const {
        return batch_id_;
    }

    /**
     * @brief Atomically increment the reference count.
     * 
     * Called when a new DataBatchView is created that references this batch.
     * Uses relaxed memory ordering as reference counting doesn't require synchronization
     * with other memory operations.
     */
    void IncrementRefCount() {
        ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Atomically decrement the reference count.
     * 
     * Called when a DataBatchView is destroyed. When the reference count reaches zero,
     * this batch can be safely evicted or moved between memory tiers.
     * Uses relaxed memory ordering as reference counting doesn't require synchronization
     * with other memory operations.
     */
    void DecrementRefCount() {
        ref_count_.fetch_sub(1, std::memory_order_relaxed);
    }

    /**
     * @brief Create a DataBatchView referencing this DataBatch.
     * 
     * Increments the reference count to account for the new view.
     * 
     * @return sirius::unique_ptr<DataBatchView> A unique pointer to the new DataBatchView
     */
    sirius::unique_ptr<DataBatchView> CreateDataBatchView() {
        return sirius::make_unique<DataBatchView>(this, cols);
    }

private:
    uint64_t batch_id_;                                   ///< Unique identifier for this data batch
    sirius::unique_ptr<IDataRepresentation> data_;       ///< Pointer to the actual data representation
    sirius::atomic<size_t> ref_count_ = 0;               ///< Reference count for tracking usage
};

} // namespace sirius