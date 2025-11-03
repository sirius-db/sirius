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

#include "data/data_batch_view.hpp"
#include "data/gpu_data_representation.hpp"
#include "data/data_repository_manager.hpp"

namespace sirius {

data_batch_view::data_batch_view(data_batch* batch)
    : _batch(batch) {
    // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
    _batch->increment_view_ref_count();
}

data_batch_view::data_batch_view(const data_batch_view& other)
    : _batch(other._batch) {
    // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
    _batch->increment_view_ref_count();
}

data_batch_view& data_batch_view::operator=(const data_batch_view& other) {
    if (this != &other) {
        if (_pinned) {
            unpin();
        }
        _batch->decrement_view_ref_count();
        _batch = other._batch;
        // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
        _batch->increment_view_ref_count();
    }
    return *this;
}

cudf::table_view
data_batch_view::get_cudf_table_view() const {
    if (!_pinned) {
        throw std::runtime_error("data_batch_view is not pinned");
    }
    return _batch->get_data()->cast<gpu_table_representation>().get_table().view();
}

void data_batch_view::pin() {
    if (_batch->get_data()->get_current_tier() != memory::Tier::GPU) {
        throw std::runtime_error("data_batch_view must be in GPU tier to be pinned");
        // TODO: later on we will add a method here to move the data to GPU tier
    }
    if (_pinned) {
        throw std::runtime_error("data_batch_view is already pinned");
    }
    _pinned = true;
    _batch->increment_pin_ref_count();
}

void data_batch_view::unpin() {
    if (!_pinned) {
        throw std::runtime_error("data_batch_view is not pinned");
    }
    _pinned = false;
    _batch->decrement_pin_ref_count();
}

data_batch_view::~data_batch_view() {
    if (_pinned) {
        unpin();
    }
    size_t old_count = _batch->decrement_view_ref_count();
    if (old_count == 1) {
        data_repository_manager* data_repo_mgr = _batch->get_data_repository_manager();
        data_repo_mgr->delete_data_batch(_batch->get_batch_id());
    }
}

} // namespace sirius

