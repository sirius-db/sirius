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

#include "data/data_batch.hpp"
#include "data/data_batch_view.hpp"

namespace cucascade {

void data_repository_manager::add_new_repository(size_t operator_id,
                                                 std::string_view port_id,
                                                 std::unique_ptr<idata_repository> repository)
{
  std::unique_ptr<idata_repository> old_repository;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    // Move out the old repository before replacing to avoid holding the lock during destruction
    auto it = _repositories.find({operator_id, std::string(port_id)});
    if (it != _repositories.end()) { old_repository = std::move(it->second); }
    _repositories[{operator_id, std::string(port_id)}] = std::move(repository);
  }
  // old_repository is destroyed here, outside the locked section
  // This prevents deadlock when data_batch_view destructors call delete_data_batch()
}

void data_repository_manager::add_new_data_batch(
  std::unique_ptr<data_batch> batch, std::vector<std::pair<size_t, std::string_view>> ops)
{
  for (auto op : ops) {
    auto batch_view = batch->create_view();
    _repositories[{op.first, std::string(op.second)}]->add_new_data_batch_view(
      std::move(batch_view));
  }
  std::lock_guard<std::mutex> lock(_mutex);
  _data_batches.insert({batch->get_batch_id(), std::move(batch)});
}

void data_repository_manager::delete_data_batch(size_t batch_id)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _data_batches.erase(batch_id);
}

std::unique_ptr<idata_repository>& data_repository_manager::get_repository(size_t operator_id,
                                                                           std::string_view port_id)
{
  return _repositories.at({operator_id, std::string(port_id)});
}

std::vector<std::unique_ptr<data_batch>> data_repository_manager::get_data_batches_for_downgrade(
  cucascade::memory::memory_space_id memory_space_id, size_t amount_to_downgrade)
{
  std::vector<std::unique_ptr<data_batch>> data_batches;
  size_t downgrade_amount = 0;
  for (auto& [batch_id, batch] : _data_batches) {
    if (batch->get_memory_space()->get_id() == memory_space_id) {
      data_batches.push_back(std::move(batch));
    }
    downgrade_amount += batch->get_data()->get_size_in_bytes();
    if (downgrade_amount >= amount_to_downgrade) { break; }
  }
  return data_batches;
}

uint64_t data_repository_manager::get_next_data_batch_id() { return _next_data_batch_id++; }

}  // namespace cucascade
