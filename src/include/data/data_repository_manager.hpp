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

#include <unordered_map>

#include "data_batch.hpp"
#include "helper/helper.hpp"
#include "data/data_repository.hpp"

namespace sirius {

/**
 * @brief Central manager for coordinating data repositories across multiple pipelines.
 * 
 * DataRepositoryManager serves as the top-level coordinator for data management in the
 * Sirius system. It maintains a collection of IDataRepository instances, each associated
 * with a specific pipeline, and provides centralized services for:
 * 
 * - Repository lifecycle management (creation, access, cleanup)
 * - Cross-pipeline data batch coordination
 * - Unique batch ID generation
 * - Global eviction and memory management policies
 * 
 * Architecture:
 * ```
 * DataRepositoryManager
 * ├── Pipeline 1 → IDataRepository (FIFO/LRU/Priority)
 * ├── Pipeline 2 → IDataRepository (FIFO/LRU/Priority)  
 * └── Pipeline N → IDataRepository (FIFO/LRU/Priority)
 * ```
 * 
 * The manager abstracts the complexity of multi-pipeline data management and provides
 * a unified interface for higher-level components like the GPU executor and memory manager.
 * 
 * @note All operations are thread-safe and can be called concurrently from multiple
 *       pipeline execution threads.
 */
class DataRepositoryManager {
public:
    /**
     * @brief Default constructor - initializes empty repository manager.
     */
    DataRepositoryManager() = default;

    /**
     * @brief Register a new data repository for the specified pipeline.
     * 
     * Associates a data repository implementation with a pipeline ID. Each pipeline
     * can have exactly one repository, and attempting to add a repository for an
     * existing pipeline will replace the previous one.
     * 
     * @param pipeline_id Unique identifier for the pipeline
     * @param repository Unique pointer to the repository implementation (ownership transferred)
     * 
     * @note Thread-safe operation
     */
    void AddNewRepository(size_t pipeline_id, sirius::unique_ptr<IDataRepository> repository);

    /**
     * @brief Add a new DataBatch to the holder.
     * 
     * This method stores the actual DataBatch object in the manager's holder.
     * DataBatchViews reference these batches.
     * 
     * @param data_batch The DataBatch to add (ownership transferred)
     * 
     * @note Thread-safe operation
     */
    void AddNewDataBatch(sirius::unique_ptr<DataBatch> data_batch, sirius::vector<size_t> pipeline_ids);

    /**
     * @brief Get direct access to a pipeline's repository for advanced operations.
     * 
     * Provides direct access to the underlying repository implementation, allowing
     * for repository-specific operations that aren't covered by the common interface.
     * 
     * @param pipeline_id ID of the pipeline whose repository to access
     * @return sirius::unique_ptr<IDataRepository>& Reference to the repository
     * 
     * @throws std::out_of_range If no repository exists for the specified pipeline
     * @note Thread-safe for read access, but modifications should use the repository's own thread safety
     */
    sirius::unique_ptr<IDataRepository>& GetRepository(size_t pipeline_id);

    /**
     * @brief Generate a globally unique data batch identifier.
     * 
     * Returns a monotonically increasing ID that's unique across all pipelines
     * and repositories managed by this instance. Used to ensure data batches
     * can be uniquely identified for debugging, tracking, and cross-reference purposes.
     * 
     * @return uint64_t A unique batch ID
     * 
     * @note Thread-safe atomic operation with no contention
     */
    uint64_t GetNextDataBatchId();

private:
    mutex mutex_;                                      ///< Mutex for thread-safe access to holder
    sirius::atomic<uint64_t> next_data_batch_id_ = 0;  ///< Atomic counter for generating unique data batch identifiers
    sirius::unordered_map<size_t, sirius::unique_ptr<IDataRepository>> repositories; ///< Map of pipeline ID to IDataRepository
    sirius::vector<sirius::unique_ptr<DataBatch>> holder;  ///< Map to hold the actual DataBatch
};

}