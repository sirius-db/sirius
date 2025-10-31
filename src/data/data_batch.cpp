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

namespace sirius {

data_batch::data_batch(uint64_t batch_id, data_repository_manager& data_repo_mgr, sirius::unique_ptr<idata_representation> data)
    : _batch_id(batch_id), _data(std::move(data)), _data_repo_mgr(&data_repo_mgr) {}

data_batch::data_batch(data_batch&& other)
    : _batch_id(other._batch_id), 
      _data(std::move(other._data)),
      _data_repo_mgr(other._data_repo_mgr) {
    size_t other_view_count = other._view_count.load(std::memory_order_relaxed);
    size_t other_pin_count = other._pin_count.load(std::memory_order_relaxed);
    if (other_view_count != 0) {
        throw std::runtime_error("Cannot move data_batch with active views (view_count != 0)");
    }
    if (other_pin_count != 0) {
        throw std::runtime_error("Cannot move data_batch with active pins (pin_count != 0)");
    }
    other._batch_id = 0;
    other._data = nullptr;
}

data_batch& data_batch::operator=(data_batch&& other) {
    if (this != &other) {
        size_t other_view_count = other._view_count.load(std::memory_order_relaxed);
        size_t other_pin_count = other._pin_count.load(std::memory_order_relaxed);
        if (other_view_count != 0) {
            throw std::runtime_error("Cannot move data_batch with active views (view_count != 0)");
        }
        if (other_pin_count != 0) {
            throw std::runtime_error("Cannot move data_batch with active pins (pin_count != 0)");
        }
        _batch_id = other._batch_id;
        _data = std::move(other._data);
        _data_repo_mgr = other._data_repo_mgr;
        other._batch_id = 0;
        other._data = nullptr;
    }
    return *this;
}

Tier data_batch::get_current_tier() const {
    return _data->get_current_tier();
}

uint64_t data_batch::get_batch_id() const {
    return _batch_id;
}

void data_batch::increment_view_ref_count() {
    _view_count.fetch_add(1, std::memory_order_relaxed);
}

size_t data_batch::decrement_view_ref_count() {
    size_t old_count = _view_count.fetch_sub(1, std::memory_order_relaxed);
    return old_count;
}

void data_batch::decrement_pin_ref_count() {
    std::lock_guard<sirius::mutex> lock(_mutex);
    if (_data->get_current_tier() != Tier::GPU) {
        throw std::runtime_error("data_batch_view should always be in GPU tier");
    }
    _pin_count.fetch_sub(1, std::memory_order_relaxed);
}

void data_batch::increment_pin_ref_count() {
    std::lock_guard<sirius::mutex> lock(_mutex);
    if (_data->get_current_tier() != Tier::GPU) {
        throw std::runtime_error("data_batch data must be in GPU tier to create cudf_table_view_wrapper");
    }
    _pin_count.fetch_add(1, std::memory_order_relaxed);
}

idata_representation* data_batch::get_data() const {
    return _data.get();
}

sirius::unique_ptr<data_batch_view> data_batch::create_view() {
    return sirius::make_unique<data_batch_view>(this);
}

size_t data_batch::get_view_count() const {
    return _view_count.load(std::memory_order_relaxed);
}

size_t data_batch::get_pin_count() const {
    return _pin_count.load(std::memory_order_relaxed);
}

data_repository_manager* data_batch::get_data_repository_manager() const {
    return _data_repo_mgr;
}

} // namespace sirius