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

#include "data/data_repository_manager.hpp"
#include "data/data_batch_view.hpp"
#include "data/data_batch.hpp"

namespace sirius {

void DataRepositoryManager::AddNewRepository(size_t pipeline_id, sirius::unique_ptr<IDataRepository> repository) {
    sirius::lock_guard<sirius::mutex> lock(mutex_);
    repositories[pipeline_id] = std::move(repository);
}

void DataRepositoryManager::AddNewDataBatch(sirius::unique_ptr<DataBatch> data_batch, sirius::vector<size_t> pipeline_ids) {
    for (size_t pipeline_id : pipeline_ids) {
        auto data_batch_view = data_batch->CreateView();
        repositories[pipeline_id]->AddNewDataBatchView(std::move(data_batch_view));
    }
    sirius::lock_guard<sirius::mutex> lock(mutex_);
    holder.push_back(std::move(data_batch));
}

sirius::unique_ptr<IDataRepository>& DataRepositoryManager::GetRepository(size_t pipeline_id) {
    return repositories.at(pipeline_id);
}

uint64_t DataRepositoryManager::GetNextDataBatchId() {
    return next_data_batch_id_++;
}

} // namespace sirius
