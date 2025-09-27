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

namespace sirius {

class DataRepository {
public:

    DataRepository(size_t num_pipelines) {
        data_batches_.resize(num_pipelines);
    }

    // Add a new DataBatch to the repository at the specified pipeline_id and idx
    void addNewDataBatch(size_t pipeline_id, size_t idx, std::unique_ptr<DataBatch> data_batch) {
    }

    // Get a DataBatch by pipeline_id and idx and transfer ownership
    std::unique_ptr<DataBatch> getDataBatch(size_t pipeline_id, size_t idx) {
        return nullptr;
    }

private:
    std::vector<std::vector<std::unique_ptr<DataBatch>>> data_batches_;
};

}