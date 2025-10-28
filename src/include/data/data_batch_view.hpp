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

/**
 * @brief A view into a DataBatch that manages reference counting and provides CUDF table access.
 * 
 * DataBatchView maintains a pointer to a DataBatch and manages reference counting to ensure
 * the batch cannot be destroyed while views are still active. It provides pin/unpin functionality
 * to prevent eviction and allows access to CUDF table_view when pinned.
 * 
 * Key characteristics:
 * - Manages reference counting on the underlying DataBatch
 * - Copyable (each copy increments reference count)
 * - Thread-safe reference counting through atomic operations in DataBatch
 * - Pin/unpin functionality prevents batch eviction from GPU memory
 * - Provides CUDF table_view access when pinned
 * 
 * Usage pattern:
 * ```cpp
 * auto view = batch->CreateView();
 * view->Pin();  // Prevents eviction
 * auto cudf_view = view->GetCudfTableView();
 * // Use cudf_view with CUDF operations
 * view->Unpin();  // Allows eviction again
 * ```
 * 
 * @note The underlying DataBatch must remain valid for the lifetime of any views referencing it.
 */
class DataBatchView {
public:
    /**
     * @brief Construct a new DataBatchView from a DataBatch.
     * 
     * Stores the batch pointer and increments the view reference count on the batch.
     * Uses atomic operations for thread-safe reference counting.
     * 
     * @param batch Pointer to the DataBatch to create a view of (must remain valid)
     * @note Automatically increments the reference count on the provided batch
     */
    DataBatchView(DataBatch* batch);

    /**
     * @brief Copy constructor - creates a new view referencing the same batch.
     * 
     * Copies the batch pointer from the other view and increments the reference count.
     * Uses atomic operations for thread-safe reference counting.
     * 
     * @param other The DataBatchView to copy from
     * @note Automatically increments the reference count on the underlying batch
     */
    DataBatchView(const DataBatchView& other);

    /**
     * @brief Copy assignment operator - updates this view to reference a different batch.
     * 
     * Performs self-assignment check, then copies the batch pointer and increments
     * the new batch's reference count.
     * 
     * @param other The DataBatchView to copy from
     * @return DataBatchView& Reference to this view
     * 
     * @warning Currently does not decrement the old batch's reference count before assignment.
     *          This may result in reference count leaks.
     * @warning Implementation calls cudf::table_view::operator= but this class doesn't inherit from it.
     */
    DataBatchView& operator=(const DataBatchView& other);

    /**
     * @brief Destructor - automatically unpins if pinned and decrements the view reference count on the underlying batch.
     * 
     * If the view is pinned, calls Unpin() to decrement the pin count and clear the pinned flag.
     * Decrements the view reference count on the underlying batch.
     */
    ~DataBatchView();

    /**
     * @brief Pin the DataBatchView to prevent batch eviction from GPU memory.
     * 
     * Validates that the data is in GPU tier and not already pinned, then sets the
     * pinned flag and increments the pin reference count on the underlying batch.
     * 
     * @throws std::runtime_error if data is not in GPU tier
     * @throws std::runtime_error if the view is already pinned
     * @note Future implementation will support automatic tier movement to GPU
     */
    void Pin();

    /**
     * @brief Unpin the DataBatchView to allow batch eviction.
     * 
     * Validates that the view is currently pinned, then clears the pinned flag
     * and decrements the pin reference count on the underlying batch.
     * 
     * @throws std::runtime_error if the view is not currently pinned
     */
    void Unpin();

    /**
     * @brief Get a CUDF table_view for performing CUDF operations.
     * 
     * Returns a CUDF table_view by casting the underlying data to GPUTableRepresentation
     * and extracting the table view. Requires the view to be pinned before calling.
     * 
     * @return cudf::table_view A CUDF table view for this batch's data
     * @throws std::runtime_error if the view is not currently pinned
     */
    cudf::table_view GetCudfTableView() const;

private:
    DataBatch* batch_;  ///< Pointer to the underlying DataBatch being viewed
    bool pinned_ = false;  ///< Whether the batch is pinned
};

} // namespace sirius