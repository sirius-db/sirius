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

class DataRepositoryManager {
public:
    DataRepositoryManager() = default;

    void AddNewRepository(size_t pipeline_id, sirius::unique_ptr<IDataRepository> repository);

    void AddNewDataBatch(size_t pipeline_id, sirius::unique_ptr<DataBatchView> data_batch_view);

    sirius::unique_ptr<DataBatchView> EvictDataBatch(size_t pipeline_id);

    sirius::unique_ptr<IDataRepository>& GetRepository(size_t pipeline_id) {
        return repositories.at(pipeline_id);
    }

    uint64_t GetNextDataBatchId() {
        return next_data_batch_id_++;
    }

private:
    mutex mutex_;                                      ///< Mutex for thread-safe access to holder
    sirius::atomic<uint64_t> next_data_batch_id_ = 0;  ///< Atomic counter for generating unique data batch identifiers
    sirius::unordered_map<size_t, sirius::unique_ptr<IDataRepository>> repositories; ///< Map of pipeline ID to IDataRepository
    sirius::unordered_map<size_t, DataBatch*> holder;  ///< Map to hold the actual DataBatch
};

}