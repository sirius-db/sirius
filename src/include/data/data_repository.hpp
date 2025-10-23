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
#include "data/data_repository_level.hpp"

namespace sirius {

/**
 * @brief A hierarchical container for DataBatches produced and consumed by system tasks.
 * 
 * The DataRepository serves as an intermediary storage system for DataBatches that are
 * transitioning between tasks in the execution pipeline. Each task outputs its results
 * to the DataRepository, and subsequent tasks retrieve them when ready for processing.
 * 
 * The repository is organized as a leveled container where each level corresponds to
 * the output of a specific pipeline in the query execution plan. When determining
 * which DataBatches to downgrade to lower memory tiers, the DataRepository considers
 * the pipeline's position in the overall query DAG and delegates to individual levels
 * to determine the optimal downgrade candidates within each level.
 */
class DataRepository {
public:
    /**
     * @brief Default constructor for the DataRepository
     */
    DataRepository() = default;

    /** 
     * @brief Adds a new level to the DataRepository for a specific pipeline
     * 
     * This method transfers ownership of the level to the DataRepository. The level
     * should not be used by any other component after this call.
     * 
     * @param pipeline_id The unique identifier of the pipeline for which the level is being added
     * @param level The data repository level to add (ownership is transferred)
     * @throws std::invalid_argument if a level already exists for the specified pipeline_id
    */
    void AddNewLevel(size_t pipeline_id, sirius::unique_ptr<IDataRepositoryLevel> level);

    /**
     * @brief Adds a new DataBatch to the repository
     * 
     * If a level was not previously initialized for the given pipeline_id, it will be
     * initialized with the default IDataRepositoryLevel implementation. For optimal
     * performance, it is recommended to call AddNewLevel for each pipeline in the
     * query plan before starting execution.
     * 
     * @param pipeline_id The identifier of the pipeline depositing the DataBatch
     * @param data_batch The DataBatch to add to the repository (ownership is transferred)
     */
    void AddNewDataBatch(size_t pipeline_id, sirius::unique_ptr<DataBatch> data_batch);

    /**
     * @brief Removes and returns a DataBatch from the repository
     * 
     * This method removes a DataBatch from the level corresponding to the specified
     * pipeline_id and transfers ownership to the caller.
     * 
     * @param pipeline_id The identifier of the pipeline where the DataBatch currently resides
     * @return sirius::unique_ptr<DataBatch> The evicted DataBatch
     * @throws std::invalid_argument if no level exists for the specified pipeline_id or if no data batch is available
     */
    sirius::unique_ptr<DataBatch> EvictDataBatch(size_t pipeline_id);

    /**
     * @brief Generates a new unique identifier for a DataBatch
     * 
     * This method atomically generates a new unique identifier that can be used
     * to identify a DataBatch within the repository system.
     * 
     * @return uint64_t A new unique identifier for a DataBatch
     */
    uint64_t GetNextDataBatchId() {
        return next_data_batch_id_++;
    }

    sirius::unordered_map<size_t, sirius::unique_ptr<IDataRepositoryLevel>> levels_;  ///< Map storing the different levels in the DataRepository
private:
    mutex mutex_;                                      ///< Mutex for thread-safe access to repository operations
    sirius::atomic<uint64_t> next_data_batch_id_ = 0;  ///< Atomic counter for generating unique data batch identifiers
};

}