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
 * @brief A container representing input and output data units for pipeline operations.
 * 
 * The DataBatch in Sirius represents a batch/chunk/row group of data that is processed
 * by or output from a task within a pipeline, following the "morsel-driven" execution
 * model used by many modern database systems. The underlying data can be stored in
 * different memory tiers and formats, with ownership managed by the underlying
 * IDataRepresentation rather than the DataBatch itself.
 * 
 * This design allows for efficient data movement between different memory hierarchies
 * (GPU memory, CPU memory, disk) while maintaining a consistent interface for
 * pipeline operations.
 */
class DataBatch {
public:
    /**
     * @brief Constructs a new DataBatch object
     * 
     * @param batch_id Unique identifier for this data batch
     * @param data The actual data representation associated with this batch
     */
    DataBatch(uint64_t batch_id, sirius::unique_ptr<IDataRepresentation> data) 
        : batch_id_(batch_id), data_(std::move(data)) {}
    
    /**
     * @brief Move constructor for DataBatch
     * 
     * @param other The DataBatch object to move from
     */
    DataBatch(DataBatch&& other) noexcept
        : batch_id_(other.batch_id_), data_(std::move(other.data_)) {
        other.batch_id_ = 0;
        other.data_ = nullptr;
    }

    /**
     * @brief Move assignment operator for DataBatch
     * 
     * @param other The DataBatch object to move from
     * @return DataBatch& Reference to this object
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
     * @brief Gets the memory tier where this data batch currently resides
     * 
     * @return Tier The memory tier (e.g., GPU, CPU, disk) where the data is stored
     */
    Tier getCurrentTier() const {
        return data_->getCurrentTier();
    }

    /**
     * @brief Gets this DataBatch's unique identifier
     * 
     * @return uint64_t The unique identifier associated with this DataBatch
     */
    uint64_t getBatchId() const {
        return batch_id_;
    }

private:
    uint64_t batch_id_;                                   ///< Unique identifier for this data batch
    sirius::unique_ptr<IDataRepresentation> data_;       ///< Pointer to the actual data representation
};

} // namespace sirius