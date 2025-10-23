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
 * @brief Abstract interface for a level within the data repository hierarchy.
 * 
 * Each level in the data repository is a thread-safe container designed to store
 * the output of a specific pipeline in the query execution plan. As data flows
 * through various stages of the query plan, it gets stored in different levels
 * of the data repository.
 * 
 * This design allows each level to implement its own data management policies
 * without needing to understand the broader query plan or execution DAG when
 * making decisions such as which DataBatch to prioritize for downgrading.
 */
class IDataRepositoryLevel {
public:
    /**
     * @brief Adds a new data batch to this level
     * 
     * @param data_batch The data batch to add to this level (ownership is transferred)
     */
    virtual void AddNewDataBatch(sirius::unique_ptr<DataBatch> data_batch) = 0;

    /**
     * @brief Removes and returns a data batch from this level
     * 
     * @return sirius::unique_ptr<DataBatch> The evicted data batch
     * @throws std::invalid_argument if no data batch is available for eviction
     */
    virtual sirius::unique_ptr<DataBatch> EvictDataBatch() = 0;

    /**
     * @brief Returns an ordered list of data batch IDs prioritized for downgrading
     * 
     * The returned IDs are ordered by downgrade priority, with the ID at index 0
     * having the highest priority for downgrading, followed by index 1, and so on.
     * Each derived class can implement its own logic for determining the priority
     * of downgrading batches within this level.
     * 
     * @param num_data_batches The maximum number of data batches to return (Top-K downgradable batches)
     * @return std::vector<uint64_t> Ordered list of data batch IDs for downgrading.
     *         Size = min(num_data_batches, number of downgradable batches)
     */
    virtual std::vector<uint64_t> GetDowngradableDataBatches(size_t num_data_batches) = 0;

    /**
     * @brief Safely casts this interface to a specific derived type
     * 
     * @tparam TARGET The target type to cast to
     * @return TARGET& Reference to the casted object
     */
	template <class TARGET>
	TARGET &Cast() {
		return reinterpret_cast<TARGET &>(*this);
	}

    /**
     * @brief Safely casts this interface to a specific derived type (const version)
     * 
     * @tparam TARGET The target type to cast to
     * @return const TARGET& Const reference to the casted object
     */
	template <class TARGET>
	const TARGET &Cast() const {
		return reinterpret_cast<const TARGET &>(*this);
	}
};

} // namespace sirius