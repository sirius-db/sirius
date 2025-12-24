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

namespace cucascade {

void idata_repository::add_new_data_batch_view(std::unique_ptr<data_batch_view> batch_view)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _data_batches.push_back(std::move(batch_view));
}

std::unique_ptr<data_batch_view> idata_repository::pull_data_batch_view()
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_data_batches.empty()) { return nullptr; }
  auto batch = std::move(_data_batches.front());
  _data_batches.erase(_data_batches.begin());
  return batch;
}

// THIS IS JUST A PLACEHOLDER FOR NOW, WE WILL ADJUST TO THE NEW DATA REPOSITORY INTERFACE IN
// cuCascade
void idata_repository::add_new_data_batch(std::shared_ptr<data_batch> batch)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _shared_data_batches.push_back(std::move(batch));
}

// THIS IS JUST A PLACEHOLDER FOR NOW, WE WILL ADJUST TO THE NEW DATA REPOSITORY INTERFACE IN
// cuCascade
std::shared_ptr<data_batch> idata_repository::pull_data_batch()
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_shared_data_batches.empty()) { return nullptr; }
  auto batch = std::move(_shared_data_batches.front());
  _shared_data_batches.erase(_shared_data_batches.begin());
  return batch;
}

bool idata_repository::check_data_batch_view_availability()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return !_data_batches.empty();
}

}  // namespace cucascade
