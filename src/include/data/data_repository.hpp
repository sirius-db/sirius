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
#include "data_batch.hpp"
#include "helper/helper.hpp"

namespace sirius {

/**
 * @brief Abstract interface for managing collections of data_batch_view objects within a pipeline.
 * 
 * idata_repository defines the contract for storing, retrieving, and managing data batches
 * within a specific pipeline. Different implementations can provide various storage strategies,
 * such as:
 * - FIFO (First In, First Out) repositories for streaming data
 * - LRU (Least Recently Used) repositories for caching scenarios
 * - Priority-based repositories for workload-aware scheduling
 * 
 * The repository is responsible for:
 * - Managing the lifecycle of data_batch_view objects
 * - Implementing eviction policies when memory pressure occurs
 * - Providing downgrade candidates for memory tier management
 * - Thread-safe access to shared data structures
 * 
 * @note Implementations must be thread-safe as multiple threads may access
 *       the repository concurrently during query execution.
 */
class idata_repository {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes.
     */
    virtual ~idata_repository() = default;

    /**
     * @brief Add a new data batch to this repository.
     * 
     * The repository takes ownership of the data_batch_view and will manage its lifecycle
     * according to the implementation's storage policy.
     * 
     * @param batch_view Unique pointer to the data_batch_view to add (ownership transferred)
     * 
     * @note Thread-safe operation protected by internal mutex
     */
    virtual void add_new_data_batch_view(sirius::unique_ptr<data_batch_view> batch_view);

    /**
     * @brief Remove and return a data batch from this repository according to eviction policy.
     * 
     * The specific data batch returned depends on the implementation's eviction strategy:
     * - FIFO: Returns the oldest batch
     * - LRU: Returns the least recently used batch
     * - Priority: Returns the lowest priority batch
     * 
     * @return sirius::unique_ptr<data_batch_view> The evicted data batch, or nullptr if empty
     * 
     * @note Thread-safe operation protected by internal mutex
     */
    virtual sirius::unique_ptr<data_batch_view> pull_data_batch_view();

protected:
    sirius::mutex _mutex;                                      ///< Mutex for thread-safe access to repository operations
    sirius::vector<sirius::unique_ptr<data_batch_view>> _data_batches;  ///< Map of pipeline source to data_batch_view
};

} // namespace sirius