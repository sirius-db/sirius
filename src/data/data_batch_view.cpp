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

namespace sirius {

DataBatchView::DataBatchView(DataBatch* batch)
    : batch_(batch) {
    // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
    batch_->IncrementViewRefCount();
}

DataBatchView::DataBatchView(const DataBatchView& other)
    : batch_(other.batch_) {
    // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
    batch_->IncrementViewRefCount();
}

DataBatchView& DataBatchView::operator=(const DataBatchView& other) {
    if (this != &other) {
        batch_ = other.batch_;
        // Thread-safe: acquire lock, validate GPU tier, increment ref count, release lock
        batch_->IncrementViewRefCount();
    }
    return *this;
}

cudf::table_view 
DataBatchView::GetCudfTableView() {
    if (!pinned_) {
        throw std::runtime_error("DataBatchView is not pinned");
    }
    return batch_->GetData()->Cast<GPUTableRepresentation>().GetTable().view();
}

void DataBatchView::Pin() {
    if (batch_->GetData()->GetCurrentTier() != Tier::GPU) {
        throw std::runtime_error("DataBatchView must be in GPU tier to be pinned");
        // TODO: later on we will add a method here to move the data to GPU tier
    }
    if (pinned_) {
        throw std::runtime_error("DataBatchView is already pinned");
    }
    pinned_ = true;
    batch_->IncrementPinRefCount();
}

void DataBatchView::Unpin() {
    if (!pinned_) {
        throw std::runtime_error("DataBatchView is not pinned");
    }
    pinned_ = false;
    batch_->DecrementPinRefCount();
}

DataBatchView::~DataBatchView() {
    if (pinned_) {
        Unpin();
    }
}

} // namespace sirius

