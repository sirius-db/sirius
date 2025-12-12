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

#include "data/data_batch.hpp"
#include "data/data_batch_view.hpp"
#include "data/gpu_data_representation.hpp"
#include "memory/memory_reservation.hpp"

namespace sirius {

data_batch::data_batch(uint64_t batch_id,
                       data_repository_manager& data_repo_mgr,
                       sirius::unique_ptr<idata_representation> data,
                       sirius::memory::memory_space& memory_space)
  : _batch_id(batch_id),
    _data(std::move(data)),
    _data_repo_mgr(&data_repo_mgr),
    _memory_space(&memory_space)
{
}

data_batch::data_batch(uint64_t batch_id,
                       data_repository_manager& data_repo_mgr,
                       sirius::unique_ptr<idata_representation> data)
  : _batch_id(batch_id),
    _data(std::move(data)),
    _data_repo_mgr(&data_repo_mgr),
    _memory_space(nullptr)
{
}

data_batch::data_batch(data_batch&& other)
  : _batch_id(other._batch_id),
    _data(std::move(other._data)),
    _data_repo_mgr(other._data_repo_mgr),
    _memory_space(other._memory_space)
{
  std::lock_guard<sirius::mutex> lock(other._mutex);
  size_t other_view_count = other._view_count;
  size_t other_pin_count  = other._pin_count;
  if (other_view_count != 0) {
    throw std::runtime_error("Cannot move data_batch with active views (view_count != 0)");
  }
  if (other_pin_count != 0) {
    throw std::runtime_error("Cannot move data_batch with active pins (pin_count != 0)");
  }
  other._batch_id     = 0;
  other._data         = nullptr;
  other._memory_space = nullptr;
}

data_batch& data_batch::operator=(data_batch&& other)
{
  if (this != &other) {
    std::lock_guard<sirius::mutex> lock(other._mutex);
    size_t other_view_count = other._view_count;
    size_t other_pin_count  = other._pin_count;
    if (other_view_count != 0) {
      throw std::runtime_error("Cannot move data_batch with active views (view_count != 0)");
    }
    if (other_pin_count != 0) {
      throw std::runtime_error("Cannot move data_batch with active pins (pin_count != 0)");
    }
    _batch_id           = other._batch_id;
    _data               = std::move(other._data);
    _data_repo_mgr      = other._data_repo_mgr;
    _memory_space       = other._memory_space;
    other._batch_id     = 0;
    other._data         = nullptr;
    other._memory_space = nullptr;
  }
  return *this;
}

memory::Tier data_batch::get_current_tier() const
{
  if (_memory_space != nullptr) { return _memory_space->get_tier(); }
  return _data->get_current_tier();
}

uint64_t data_batch::get_batch_id() const { return _batch_id; }

void data_batch::increment_view_ref_count()
{
  std::lock_guard<sirius::mutex> lock(_mutex);
  _view_count += 1;
}

size_t data_batch::decrement_view_ref_count()
{
  std::lock_guard<sirius::mutex> lock(_mutex);
  size_t old_count = _view_count;
  _view_count -= 1;
  return old_count;
}

void data_batch::decrement_pin_ref_count()
{
  std::lock_guard<sirius::mutex> lock(_mutex);
  const auto tier =
    (_memory_space != nullptr) ? _memory_space->get_tier() : _data->get_current_tier();
  if (tier != memory::Tier::GPU) {
    throw std::runtime_error("data_batch_view should always be in GPU tier");
  }
  _pin_count -= 1;
}

void data_batch::increment_pin_ref_count()
{
  std::lock_guard<sirius::mutex> lock(_mutex);
  const auto tier =
    (_memory_space != nullptr) ? _memory_space->get_tier() : _data->get_current_tier();
  if (tier != memory::Tier::GPU) {
    throw std::runtime_error(
      "data_batch data must be in GPU tier to create cudf_table_view_wrapper");
  }
  _pin_count += 1;
}

idata_representation* data_batch::get_data() const { return _data.get(); }

sirius::unique_ptr<data_batch_view> data_batch::create_view()
{
  return sirius::make_unique<data_batch_view>(this);
}

size_t data_batch::get_view_count() const
{
  std::lock_guard<sirius::mutex> lock(_mutex);
  return _view_count;
}

size_t data_batch::get_pin_count() const
{
  std::lock_guard<sirius::mutex> lock(_mutex);
  return _pin_count;
}

data_repository_manager* data_batch::get_data_repository_manager() const { return _data_repo_mgr; }

sirius::memory::memory_space* data_batch::get_memory_space() const
{
  if (_memory_space != nullptr) { return _memory_space; }
  if (_data == nullptr) { return nullptr; }
  auto& manager = sirius::memory::memory_reservation_manager::get_instance();
  auto* space   = manager.get_memory_space(_data->get_current_tier(), _data->get_device_id());
  return const_cast<sirius::memory::memory_space*>(space);
}

void data_batch::set_data(sirius::unique_ptr<idata_representation> data)
{
  std::lock_guard<sirius::mutex> lock(_mutex);
  size_t views = _view_count;
  size_t pins  = _pin_count;
  if (views != 0 || pins != 0) {
    throw std::runtime_error("Cannot set data while there are active views or pins");
  }
  _data = std::move(data);
  if (_data) {
    auto& manager = sirius::memory::memory_reservation_manager::get_instance();
    auto* space   = manager.get_memory_space(_data->get_current_tier(), _data->get_device_id());
    _memory_space = const_cast<sirius::memory::memory_space*>(space);
  } else {
    _memory_space = nullptr;
  }
}

void data_batch::convert_to_memory_space(const sirius::memory::memory_space* target_memory_space,
                                         rmm::cuda_stream_view stream)
{
  std::lock_guard<sirius::mutex> lock(_mutex);

  if (_pin_count != 0) {
    throw std::runtime_error("Cannot convert memory space while there are active pins");
  }
  auto new_representation = _data->convert_to_memory_space(target_memory_space, stream);
  _data                   = std::move(new_representation);
  _memory_space           = const_cast<sirius::memory::memory_space*>(target_memory_space);
}

}  // namespace sirius