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

void data_repository_manager::add_new_repository(::duckdb::GPUPhysicalOperator* op, std::string_view port_id, sirius::unique_ptr<idata_repository> repository) {
    sirius::unique_ptr<idata_repository> old_repository;
    {
        sirius::lock_guard<sirius::mutex> lock(_mutex);
        // Move out the old repository before replacing to avoid holding the lock during destruction
        auto it = _repositories.find({op, std::string(port_id)});
        if (it != _repositories.end()) {
            old_repository = std::move(it->second);
        }
        _repositories[{op, std::string(port_id)}] = std::move(repository);
    }
    // old_repository is destroyed here, outside the locked section
    // This prevents deadlock when data_batch_view destructors call delete_data_batch()
}

void data_repository_manager::add_new_data_batch(sirius::unique_ptr<data_batch> batch, sirius::vector<std::pair<::duckdb::GPUPhysicalOperator*, std::string_view>> ops) {
    for (auto op : ops) {
        auto batch_view = batch->create_view();
        _repositories[{op.first, std::string(op.second)}]->add_new_data_batch_view(std::move(batch_view));
    }
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    _data_batches.insert({batch->get_batch_id(), std::move(batch)});
}

void data_repository_manager::delete_data_batch(size_t batch_id) {
    sirius::lock_guard<sirius::mutex> lock(_mutex);
    _data_batches.erase(batch_id);
}

sirius::unique_ptr<idata_repository>& data_repository_manager::get_repository(::duckdb::GPUPhysicalOperator* op, std::string_view port_id) {
    return _repositories.at({op, std::string(port_id)});
}

uint64_t data_repository_manager::get_next_data_batch_id() {
    return _next_data_batch_id++;
}

} // namespace sirius
