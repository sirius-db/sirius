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

#include "data/data_repository.hpp"
#include "data/data_batch_view.hpp"

namespace sirius {

void IDataRepository::AddNewDataBatchView(sirius::unique_ptr<DataBatchView> data_batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_batches_.push_back(std::move(data_batch));
}

sirius::unique_ptr<DataBatchView> IDataRepository::PullDataBatchView() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_batches_.empty()) {
        return nullptr;
    }
    auto batch = std::move(data_batches_.front());
    data_batches_.erase(data_batches_.begin());
    return batch;
}

} // namespace sirius

