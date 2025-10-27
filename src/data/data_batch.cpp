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

DataBatch::DataBatch(uint64_t batch_id, sirius::unique_ptr<IDataRepresentation> data)
    : batch_id_(batch_id), data_(std::move(data)) {}

DataBatch::DataBatch(DataBatch&& other) noexcept
    : batch_id_(other.batch_id_), data_(std::move(other.data_)) {
    other.batch_id_ = 0;
    other.data_ = nullptr;
}

DataBatch& DataBatch::operator=(DataBatch&& other) noexcept {
    if (this != &other) {
        batch_id_ = other.batch_id_;
        data_ = std::move(other.data_);
        other.batch_id_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

Tier DataBatch::GetCurrentTier() const {
    return data_->GetCurrentTier();
}

uint64_t DataBatch::GetBatchId() const {
    return batch_id_;
}

void DataBatch::IncrementViewRefCount() {
    view_count_.fetch_add(1, std::memory_order_relaxed);
}

void DataBatch::DecrementViewRefCount() {
    size_t old_count = view_count_.fetch_sub(1, std::memory_order_relaxed);
    if (old_count == 1) {
        delete this;
    }
}

void DataBatch::DecrementPinRefCount() {
    std::lock_guard<sirius::mutex> lock(mutex_);
    if (data_->GetCurrentTier() != Tier::GPU) {
        throw std::runtime_error("DataBatchView should always be in GPU tier");
    }
    pin_count_.fetch_sub(1, std::memory_order_relaxed);
}

void DataBatch::IncrementPinRefCount() {
    std::lock_guard<sirius::mutex> lock(mutex_);
    if (data_->GetCurrentTier() != Tier::GPU) {
        throw std::runtime_error("DataBatch data must be in GPU tier to create CuDFTableViewWrapper");
    }
    pin_count_.fetch_add(1, std::memory_order_relaxed);
}

sirius::unique_ptr<IDataRepresentation> DataBatch::GetData() const {
    return data_;
}

sirius::unique_ptr<DataBatchView> DataBatch::CreateView() {
    return sirius::make_unique<DataBatchView>(this, data_->Cast<GPUTableRepresentation>().GetTable().view());
}

size_t DataBatch::GetViewCount() const {
    return view_count_.load(std::memory_order_relaxed);
}

size_t DataBatch::GetPinCount() const {
    return pin_count_.load(std::memory_order_relaxed);
}

} // namespace sirius