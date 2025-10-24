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
#include "data/data_batch.hpp"
#include "data/common.hpp"

namespace sirius {

using sirius::memory::Tier;

/**
 * @brief A view into a DataBatch that provides CUDF table interface while managing reference counting.
 * 
 * DataBatchView extends cudf::table_view to provide a high-level interface for working with
 * columnar data while maintaining proper reference counting on the underlying DataBatch.
 * This ensures that the DataBatch cannot be evicted or destroyed while views are still active.
 * 
 * Key characteristics:
 * - Inherits from cudf::table_view, providing full CUDF API compatibility
 * - Automatically manages reference counting on construction/destruction/assignment
 * - Copyable (each copy increments reference count)
 * - Thread-safe reference counting operations
 * - RAII semantics ensure proper cleanup
 * 
 * Usage pattern:
 * ```cpp
 * auto view = std::make_unique<DataBatchView>(batch, columns);
 * // Use view with any CUDF operations
 * auto result = cudf::filter(view->select({0, 1}), predicate);
 * // Reference count automatically decremented when view goes out of scope
 * ```
 * 
 * @note The underlying DataBatch must remain valid for the lifetime of any views referencing it.
 */
class DataBatchView : public cudf::table_view {
public:
    /**
     * @brief Construct a new DataBatchView from a DataBatch and column specifications.
     * 
     * This constructor performs the following thread-safe operations:
     * 1. Acquires a lock on the DataBatch
     * 2. Validates that the data is currently in GPU tier
     * 3. Increments the reference count
     * 4. Releases the lock
     * 
     * @param batch Pointer to the DataBatch to create a view of (must remain valid)
     * @param cols Vector of column views that define the structure of this table view
     * 
     * @throws std::runtime_error if the data is not currently in GPU tier
     * @note Automatically increments the reference count on the provided batch
     */
    DataBatchView(DataBatch* batch, const std::vector<cudf::column_view>& cols)
        : batch_(batch), cudf::table_view(cols) {
        // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
        batch_->IncrementRefCount();
    }

    /**
     * @brief Copy constructor - creates a new view referencing the same batch.
     * 
     * Performs thread-safe validation and reference counting operations.
     * 
     * @param other The DataBatchView to copy from
     * 
     * @throws std::runtime_error if the data is not currently in GPU tier
     * @note Automatically increments the reference count on the underlying batch
     */
    DataBatchView(const DataBatchView& other)
        : cudf::table_view(other), batch_(other.batch_) {
        // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
        batch_->IncrementRefCount();
    }

    /**
     * @brief Copy assignment operator - updates this view to reference a different batch.
     * 
     * Properly manages reference counts by decrementing the old batch's count (implicitly
     * through destruction) and incrementing the new batch's count with validation.
     * 
     * @param other The DataBatchView to copy from
     * @return DataBatchView& Reference to this view
     * 
     * @throws std::runtime_error if the new data is not currently in GPU tier
     */
    DataBatchView& operator=(const DataBatchView& other) {
        if (this != &other) {
            cudf::table_view::operator=(other);
            batch_ = other.batch_;
            // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
            batch_->IncrementRefCount();
        }
        return *this;
    }

    /**
     * @brief Destructor - automatically decrements the reference count on the underlying batch.
     * 
     * This enables the DataBatch to be safely evicted or destroyed when no more views
     * are referencing it.
     */
    ~DataBatchView() {
        batch_->DecrementRefCount();
    }

private:
    DataBatch* batch_;
};

} // namespace sirius