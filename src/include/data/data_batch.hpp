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

class DataBatch {
public:
    DataBatch(uint64_t batch_id, sirius::unique_ptr<IDataRepresentation> data) 
        : batch_id_(batch_id), data_(std::move(data)) {}
    
    DataBatch(DataBatch&& other) noexcept
        : batch_id_(other.batch_id_), data_(std::move(other.data_)) {
        other.batch_id_ = 0;
        other.data_ = nullptr;
    }

    DataBatch& operator=(DataBatch&& other) noexcept {
        if (this != &other) {
            batch_id_ = other.batch_id_;
            data_ = std::move(other.data_);
            other.batch_id_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    Tier getCurrentTier() const {
        return data_->getCurrentTier();
    }

    uint64_t getBatchId() const {
        return batch_id_;
    }

    void increment_refcount() {
        ref_count_.fetch_add(1, std::memory_order_relaxed);
    }

    void decrement_refcount() {
        ref_count_.fetch_sub(1, std::memory_order_relaxed);
    }

private:
    uint64_t batch_id_;                                   ///< Unique identifier for this data batch
    sirius::unique_ptr<IDataRepresentation> data_;       ///< Pointer to the actual data representation
    sirius::atomic<size_t> ref_count_ = 0;               ///< Reference count for tracking usage
};

} // namespace sirius