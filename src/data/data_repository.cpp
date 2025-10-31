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

void idata_repository::add_new_data_batch_view(sirius::unique_ptr<data_batch_view> batch_view) {
    std::lock_guard<std::mutex> lock(_mutex);
    _data_batches.push_back(std::move(batch_view));
}

sirius::unique_ptr<data_batch_view> idata_repository::pull_data_batch_view() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_data_batches.empty()) {
        return nullptr;
    }
    auto batch = std::move(_data_batches.front());
    _data_batches.erase(_data_batches.begin());
    return batch;
}

} // namespace sirius

